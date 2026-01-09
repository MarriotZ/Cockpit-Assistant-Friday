"""
Voice Interface
"""

import asyncio
import numpy as np
from typing import AsyncIterator, Optional, Callable, Any
from dataclasses import dataclass
import logging
import io
import wave
import struct

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    logger.warning("faster-whisper not installed")

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    logger.warning("edge-tts not installed")

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    logger.warning("sounddevice not installed")

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    logger.warning("webrtcvad not installed")


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 480  # 30ms at 16kHz
    dtype: str = "int16"


@dataclass
class ASRConfig:
    """ASR configuration"""
    model_size: str = "small"  # tiny, base, small, medium, large
    device: str = "cuda"       # cuda, cpu
    compute_type: str = "float16"  # float16, int8
    language: str = "zh"
    beam_size: int = 5


@dataclass
class TTSConfig:
    """TTS configuration"""
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"


# =============================================================================
# ASR
# =============================================================================

class ASREngine:
    """Speech recognition engine"""
    
    def __init__(self, config: ASRConfig = None):
        self.config = config or ASRConfig()
        self._model = None
        self._initialized = False
    
    def initialize(self, max_retries=3, retry_delay=2):
        """
        Initialize ASR model
    
        Args:
            max_retries: Maximum retry attempts
            retry_delay: Retry delay in seconds
        """
        if self._initialized:
            return
        
        if HAS_WHISPER:
            logger.info(f"Loading Whisper model: {self.config.model_size}")
            
            device = self.config.device
            compute_type = self.config.compute_type
            
            if device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA not available, falling back to CPU")
                        device = "cpu"
                        compute_type = "int8"  
                except ImportError:
                    logger.warning("PyTorch not available, cannot check CUDA. Using CPU")
                    device = "cpu"
                    compute_type = "int8" 
            
            if device == "cpu" and compute_type == "float16":
                logger.warning("CPU does not support float16, using int8 instead")
                compute_type = "int8"
            
            import time
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to load Whisper model (attempt {attempt + 1}/{max_retries})...")
                    self._model = WhisperModel(
                        self.config.model_size,
                        device=device,
                        compute_type=compute_type
                    )
                    logger.info(f"Whisper model loaded on {device} with compute_type={compute_type}")
                    break
                except (RuntimeError, ValueError) as e:
                    error_str = str(e).lower()
                    if "cuda" in error_str or "cublas" in error_str:
                        logger.warning(f"CUDA error: {e}, falling back to CPU")
                        device = "cpu"
                        compute_type = "int8"
                        try:
                            self._model = WhisperModel(
                                self.config.model_size,
                                device="cpu",
                                compute_type="int8"
                            )
                            logger.info("Whisper model loaded on CPU")
                            break
                        except Exception as fallback_error:
                            last_error = fallback_error
                            if attempt < max_retries - 1:
                                logger.warning(f"Fallback failed, retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            continue
                    elif "float16" in error_str or "compute type" in error_str:
                        logger.warning(f"Compute type error: {e}, trying int8")
                        compute_type = "int8"
                        try:
                            self._model = WhisperModel(
                                self.config.model_size,
                                device=device,
                                compute_type="int8"
                            )
                            logger.info(f"Whisper model loaded on {device} with int8 (fallback)")
                            break
                        except Exception as fallback_error:
                            last_error = fallback_error
                            if attempt < max_retries - 1:
                                logger.warning(f"Fallback failed, retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            continue
                    else:
                        last_error = e
                        error_str_lower = str(e).lower()
                        if any(keyword in error_str_lower for keyword in ["ssl", "connection", "network", "timeout", "retry", "huggingface"]):
                            if attempt < max_retries - 1:
                                logger.warning(f"Network error detected: {e}")
                                logger.warning(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                continue
                        raise
                except Exception as e:
                    last_error = e
                    error_str_lower = str(e).lower()
                    if any(keyword in error_str_lower for keyword in ["ssl", "connection", "network", "timeout", "retry", "huggingface", "max retries"]):
                        if attempt < max_retries - 1:
                            logger.warning(f"Network error detected: {e}")
                            logger.warning(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.error(f"Failed to load Whisper model after {max_retries} attempts: {e}")
                            raise Exception(f"Failed to load ASR model (network error, retried {max_retries} times): {str(e)}")
                    else:
                        raise
            
            if self._model is None:
                if last_error:
                    raise Exception(f"Failed to load ASR model: {str(last_error)}")
                else:
                    raise Exception("Failed to load ASR model: unknown error")
        else:
            logger.warning("Using mock ASR")
        
        self._initialized = True
    
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio
        
        Args:
            audio_data: Audio data (numpy array)
            sample_rate: Sample rate
            
        Returns:
            Recognized text
        """
        if not self._initialized:
            self.initialize()
        
        if self._model is None:
            return "Mock speech recognition result"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_data,
            sample_rate
        )
        return result
    
    def _transcribe_sync(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Synchronous transcription"""
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
        
        segments, info = self._model.transcribe(
            audio_data,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False
        )
        
        text = "".join(segment.text for segment in segments)
        return text.strip()
    
    def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio file"""
        if not self._initialized:
            self.initialize()
        
        if self._model is None:
            return "Mock speech recognition result"
        
        segments, info = self._model.transcribe(
            file_path,
            language=self.config.language,
            beam_size=self.config.beam_size
        )
        
        return "".join(segment.text for segment in segments).strip()


# =============================================================================
# TTS - Text to Speech
# =============================================================================

class TTSEngine:
    """Text to speech engine"""
    
    CHINESE_VOICES = {
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",
        "xiaoyi": "zh-CN-XiaoyiNeural",
        "yunjian": "zh-CN-YunjianNeural",
        "yunxi": "zh-CN-YunxiNeural",
        "yunxia": "zh-CN-YunxiaNeural",      # Male
        "yunyang": "zh-CN-YunyangNeural",    # Male, news broadcast style
    }
    
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
    
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech
        
        Args:
            text: Text to synthesize
            
        Returns:
            MP3 format audio data
        """
        if not HAS_EDGE_TTS:
            return b""
        
        communicate = edge_tts.Communicate(
            text,
            self.config.voice,
            rate=self.config.rate,
            volume=self.config.volume,
            pitch=self.config.pitch
        )
        
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        return audio_data
    
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Stream synthesize speech
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio data chunks
        """
        if not HAS_EDGE_TTS:
            yield b""
            return
        
        communicate = edge_tts.Communicate(
            text,
            self.config.voice,
            rate=self.config.rate,
            volume=self.config.volume,
            pitch=self.config.pitch
        )
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    
    def set_voice(self, voice_name: str):
        """Set voice"""
        if voice_name in self.CHINESE_VOICES:
            self.config.voice = self.CHINESE_VOICES[voice_name]
        else:
            self.config.voice = voice_name
    
    def set_rate(self, rate: int):
        """Set speech rate (-100 to +100)"""
        self.config.rate = f"{rate:+d}%"
    
    def set_volume(self, volume: int):
        """Set volume (-100 to +100)"""
        self.config.volume = f"{volume:+d}%"


# =============================================================================
# VAD - Voice Activity Detection
# =============================================================================

class VADEngine:
    """Voice activity detection"""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        """
        Args:
            aggressiveness: Aggressiveness level (0-3), higher = more aggressive filtering
            sample_rate: Sample rate (8000, 16000, 32000, 48000)
        """
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self._vad = None
        
        if HAS_VAD:
            self._vad = webrtcvad.Vad(aggressiveness)
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Detect if audio chunk contains speech
        
        Args:
            audio_chunk: Audio data (10, 20, or 30ms PCM data)
            
        Returns:
            Whether speech is detected
        """
        if self._vad is None:
            return True
        
        try:
            return self._vad.is_speech(audio_chunk, self.sample_rate)
        except:
            return True


# =============================================================================
# Audio Input/Output
# =============================================================================

class AudioRecorder:
    """Audio recorder"""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._stream = None
        self._recording = False
        self._audio_buffer = []
    
    async def record(
        self, 
        duration: float = None,
        vad_timeout: float = 2.0,
        on_audio: Callable[[np.ndarray], None] = None
    ) -> np.ndarray:
        """
        Record audio
        
        Args:
            duration: Recording duration in seconds, None for VAD auto-detection
            vad_timeout: VAD timeout (stop after this duration of silence)
            on_audio: Real-time audio callback
            
        Returns:
            Recorded audio data
        """
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice not available, returning mock audio")
            return np.zeros(int(self.config.sample_rate * 3), dtype=np.int16)
        
        self._audio_buffer = []
        self._recording = True
        
        vad = VADEngine(sample_rate=self.config.sample_rate)
        silence_frames = 0
        max_silence_frames = int(vad_timeout * self.config.sample_rate / self.config.chunk_size)
        
        def callback(indata, frames, time, status):
            if not self._recording:
                return
            
            self._audio_buffer.append(indata.copy())
            
            if on_audio:
                on_audio(indata)
        
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.chunk_size,
            callback=callback
        ):
            if duration:
                await asyncio.sleep(duration)
            else:
                while self._recording:
                    await asyncio.sleep(0.03)
                    
                    if len(self._audio_buffer) > 0:
                        last_chunk = self._audio_buffer[-1]
                        is_speech = vad.is_speech(last_chunk.tobytes())
                        
                        if is_speech:
                            silence_frames = 0
                        else:
                            silence_frames += 1
                            if silence_frames > max_silence_frames:
                                break
        
        self._recording = False
        
        if self._audio_buffer:
            return np.concatenate(self._audio_buffer, axis=0).flatten()
        return np.array([], dtype=np.int16)
    
    def stop(self):
        """Stop recording"""
        self._recording = False


class AudioPlayer:
    """Audio player"""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._playing = False
    
    async def play(self, audio_data: np.ndarray, sample_rate: int = None):
        """
        Play audio
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
        """
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice not available")
            return
        
        sample_rate = sample_rate or self.config.sample_rate
        self._playing = True
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sd.play, audio_data, sample_rate)
        await loop.run_in_executor(None, sd.wait)
        
        self._playing = False
    
    async def play_bytes(self, audio_bytes: bytes, format: str = "mp3"):
        """
        Play audio bytes
        
        Args:
            audio_bytes: Audio data
            format: Format (mp3, wav)
        """
        if not audio_bytes:
            return
        
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
            samples = np.array(audio.get_array_of_samples())
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            await self.play(samples, audio.frame_rate)
            
        except ImportError:
            logger.warning("pydub not installed, cannot play audio")
    
    def stop(self):
        """Stop playback"""
        if HAS_SOUNDDEVICE:
            sd.stop()
        self._playing = False


# =============================================================================
# Integrated Voice Interface
# =============================================================================

class VoiceInterface:
    """Integrated voice interface"""
    
    def __init__(
        self,
        asr_config: ASRConfig = None,
        tts_config: TTSConfig = None,
        audio_config: AudioConfig = None
    ):
        self.asr = ASREngine(asr_config)
        self.tts = TTSEngine(tts_config)
        self.recorder = AudioRecorder(audio_config)
        self.player = AudioPlayer(audio_config)
        
        self._wake_word = "Friday"
        self._is_listening = False
    
    def set_wake_word(self, wake_word: str):
        """Set wake word"""
        self._wake_word = wake_word
    
    async def listen(self, duration: float = None) -> str:
        """
        Listen and recognize speech
        
        Args:
            duration: Recording duration, None for VAD
            
        Returns:
            Recognized text
        """
        self._is_listening = True
        audio_data = await self.recorder.record(duration=duration)
        self._is_listening = False
        
        if len(audio_data) > 0:
            text = await self.asr.transcribe(audio_data)
            return text
        
        return ""
    
    async def speak(self, text: str):
        """
        Speak text
        
        Args:
            text: Text to speak
        """
        audio_bytes = await self.tts.synthesize(text)
        if audio_bytes:
            await self.player.play_bytes(audio_bytes, format="mp3")
    
    async def speak_stream(self, text: str):
        """Stream speak text"""
        async for audio_chunk in self.tts.synthesize_stream(text):
            await self.player.play_bytes(audio_chunk, format="mp3")
    
    def stop(self):
        """Stop current operation"""
        self.recorder.stop()
        self.player.stop()
        self._is_listening = False
    
    @property
    def is_listening(self) -> bool:
        return self._is_listening


# =============================================================================
# Cockpit Voice Assistant
# =============================================================================

class CockpitVoiceAssistant:
    """Cockpit assistant with voice interaction"""
    
    def __init__(
        self,
        model_path: str,
        asr_config: ASRConfig = None,
        tts_config: TTSConfig = None
    ):
        from cockpit_assistant import CockpitAssistant
        
        self.assistant = CockpitAssistant(model_path)
        self.voice = VoiceInterface(asr_config, tts_config)
        
        self._running = False
    
    async def process_voice(self) -> str:
        """
        Process one voice interaction
        
        Returns:
            Assistant's text response
        """
        logger.info("Listening...")
        user_text = await self.voice.listen()
        
        if not user_text:
            return ""
        
        logger.info(f"User: {user_text}")
        
        response_text = ""
        async for token in self.assistant.chat(user_text):
            response_text += token
        
        logger.info(f"Assistant: {response_text}")
        
        clean_response = self._clean_response_for_tts(response_text)
        if clean_response:
            await self.voice.speak(clean_response)
        
        return response_text
    
    def _clean_response_for_tts(self, response: str) -> str:
        """Clean response text for TTS (remove JSON)"""
        import re
        cleaned = re.sub(r'\{[^}]+\}', '', response)
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    async def run_loop(self):
        """Run interaction loop"""
        self._running = True
        logger.info("Voice assistant started. Say wake word to begin.")
        
        while self._running:
            try:
                await self.process_voice()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(1)
        
        logger.info("Voice assistant stopped.")
    
    def stop(self):
        """Stop assistant"""
        self._running = False
        self.voice.stop()

async def _test():
    """Test voice interface"""
    tts = TTSEngine()
    print("Testing TTS...")
    audio = await tts.synthesize("Hello, I am the intelligent cockpit assistant Friday")
    print(f"Generated {len(audio)} bytes of audio")
    
    if HAS_SOUNDDEVICE and audio:
        player = AudioPlayer()
        await player.play_bytes(audio)


if __name__ == "__main__":
    asyncio.run(_test())