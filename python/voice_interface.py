"""
Voice Interface - 语音交互接口

集成ASR（语音识别）和TTS（语音合成）功能
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

# 尝试导入语音相关库
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    logger.warning("faster-whisper not installed, ASR will be mocked")

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    logger.warning("edge-tts not installed, TTS will be mocked")

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    logger.warning("sounddevice not installed, audio I/O will be mocked")

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    logger.warning("webrtcvad not installed, VAD will be disabled")


@dataclass
class AudioConfig:
    """音频配置"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 480  # 30ms at 16kHz
    dtype: str = "int16"


@dataclass
class ASRConfig:
    """ASR配置"""
    model_size: str = "small"  # tiny, base, small, medium, large
    device: str = "cuda"       # cuda, cpu
    compute_type: str = "float16"  # float16, int8
    language: str = "zh"
    beam_size: int = 5


@dataclass
class TTSConfig:
    """TTS配置"""
    voice: str = "zh-CN-XiaoxiaoNeural"  # 中文女声
    rate: str = "+0%"      # 语速
    volume: str = "+0%"    # 音量
    pitch: str = "+0Hz"    # 音调


# =============================================================================
# ASR - 语音识别
# =============================================================================

class ASREngine:
    """语音识别引擎"""
    
    def __init__(self, config: ASRConfig = None):
        self.config = config or ASRConfig()
        self._model = None
        self._initialized = False
    
    def initialize(self):
        """初始化ASR模型"""
        if self._initialized:
            return
        
        if HAS_WHISPER:
            logger.info(f"Loading Whisper model: {self.config.model_size}")
            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type
            )
            logger.info("Whisper model loaded")
        else:
            logger.warning("Using mock ASR")
        
        self._initialized = True
    
    async def transcribe(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int = 16000
    ) -> str:
        """
        转录音频
        
        Args:
            audio_data: 音频数据 (numpy array)
            sample_rate: 采样率
            
        Returns:
            识别的文本
        """
        if not self._initialized:
            self.initialize()
        
        if self._model is None:
            # 模拟ASR
            return "模拟语音识别结果"
        
        # 在线程池中运行转录
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_data,
            sample_rate
        )
        return result
    
    def _transcribe_sync(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """同步转录"""
        # 确保音频格式正确
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
        
        segments, info = self._model.transcribe(
            audio_data,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=True
        )
        
        text = "".join(segment.text for segment in segments)
        return text.strip()
    
    def transcribe_file(self, file_path: str) -> str:
        """转录音频文件"""
        if not self._initialized:
            self.initialize()
        
        if self._model is None:
            return "模拟语音识别结果"
        
        segments, info = self._model.transcribe(
            file_path,
            language=self.config.language,
            beam_size=self.config.beam_size
        )
        
        return "".join(segment.text for segment in segments).strip()


# =============================================================================
# TTS - 语音合成
# =============================================================================

class TTSEngine:
    """语音合成引擎"""
    
    # 可用的中文语音
    CHINESE_VOICES = {
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",      # 女声，温柔
        "xiaoyi": "zh-CN-XiaoyiNeural",          # 女声，活泼
        "yunjian": "zh-CN-YunjianNeural",        # 男声，沉稳
        "yunxi": "zh-CN-YunxiNeural",            # 男声，阳光
        "yunxia": "zh-CN-YunxiaNeural",          # 男声
        "yunyang": "zh-CN-YunyangNeural",        # 男声，新闻播报
    }
    
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
    
    async def synthesize(self, text: str) -> bytes:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            
        Returns:
            MP3格式的音频数据
        """
        if not HAS_EDGE_TTS:
            # 返回空的音频数据
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
        流式合成语音
        
        Args:
            text: 要合成的文本
            
        Yields:
            音频数据块
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
        """设置语音"""
        if voice_name in self.CHINESE_VOICES:
            self.config.voice = self.CHINESE_VOICES[voice_name]
        else:
            self.config.voice = voice_name
    
    def set_rate(self, rate: int):
        """设置语速 (-100 到 +100)"""
        self.config.rate = f"{rate:+d}%"
    
    def set_volume(self, volume: int):
        """设置音量 (-100 到 +100)"""
        self.config.volume = f"{volume:+d}%"


# =============================================================================
# VAD - 语音活动检测
# =============================================================================

class VADEngine:
    """语音活动检测"""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        """
        Args:
            aggressiveness: 激进程度 (0-3)，越高越激进地过滤非语音
            sample_rate: 采样率 (8000, 16000, 32000, 48000)
        """
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self._vad = None
        
        if HAS_VAD:
            self._vad = webrtcvad.Vad(aggressiveness)
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        检测音频块是否包含语音
        
        Args:
            audio_chunk: 音频数据 (10, 20, 或 30ms的PCM数据)
            
        Returns:
            是否包含语音
        """
        if self._vad is None:
            return True  # 如果没有VAD，假设所有音频都是语音
        
        try:
            return self._vad.is_speech(audio_chunk, self.sample_rate)
        except:
            return True


# =============================================================================
# 音频输入/输出
# =============================================================================

class AudioRecorder:
    """音频录制器"""
    
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
        录制音频
        
        Args:
            duration: 录制时长（秒），None表示使用VAD自动检测
            vad_timeout: VAD超时时间（连续无语音多久后停止）
            on_audio: 实时音频回调
            
        Returns:
            录制的音频数据
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
                # VAD模式
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
        """停止录制"""
        self._recording = False


class AudioPlayer:
    """音频播放器"""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._playing = False
    
    async def play(self, audio_data: np.ndarray, sample_rate: int = None):
        """
        播放音频
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
        """
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice not available")
            return
        
        sample_rate = sample_rate or self.config.sample_rate
        self._playing = True
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            sd.play,
            audio_data,
            sample_rate
        )
        await loop.run_in_executor(None, sd.wait)
        
        self._playing = False
    
    async def play_bytes(self, audio_bytes: bytes, format: str = "mp3"):
        """
        播放音频字节
        
        Args:
            audio_bytes: 音频数据
            format: 格式 (mp3, wav)
        """
        if not audio_bytes:
            return
        
        # 需要安装pydub来解码mp3
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
        """停止播放"""
        if HAS_SOUNDDEVICE:
            sd.stop()
        self._playing = False


# =============================================================================
# 集成语音接口
# =============================================================================

class VoiceInterface:
    """集成语音接口"""
    
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
        
        self._wake_word = "小智"
        self._is_listening = False
    
    def set_wake_word(self, wake_word: str):
        """设置唤醒词"""
        self._wake_word = wake_word
    
    async def listen(self, duration: float = None) -> str:
        """
        监听并识别语音
        
        Args:
            duration: 录制时长，None使用VAD
            
        Returns:
            识别的文本
        """
        self._is_listening = True
        
        # 录制音频
        audio_data = await self.recorder.record(duration=duration)
        
        self._is_listening = False
        
        # 识别
        if len(audio_data) > 0:
            text = await self.asr.transcribe(audio_data)
            return text
        
        return ""
    
    async def speak(self, text: str):
        """
        语音播报
        
        Args:
            text: 要播报的文本
        """
        audio_bytes = await self.tts.synthesize(text)
        if audio_bytes:
            await self.player.play_bytes(audio_bytes, format="mp3")
    
    async def speak_stream(self, text: str):
        """流式语音播报"""
        async for audio_chunk in self.tts.synthesize_stream(text):
            await self.player.play_bytes(audio_chunk, format="mp3")
    
    def stop(self):
        """停止当前操作"""
        self.recorder.stop()
        self.player.stop()
        self._is_listening = False
    
    @property
    def is_listening(self) -> bool:
        return self._is_listening


# =============================================================================
# 带语音的座舱助手
# =============================================================================

class CockpitVoiceAssistant:
    """带语音交互的座舱助手"""
    
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
        处理一次语音交互
        
        Returns:
            助手的文本响应
        """
        # 1. 监听用户语音
        logger.info("Listening...")
        user_text = await self.voice.listen()
        
        if not user_text:
            return ""
        
        logger.info(f"User: {user_text}")
        
        # 2. 获取LLM响应
        response_text = ""
        async for token in self.assistant.chat(user_text):
            response_text += token
        
        logger.info(f"Assistant: {response_text}")
        
        # 3. 语音播报
        # 清理响应文本（移除JSON部分）
        clean_response = self._clean_response_for_tts(response_text)
        if clean_response:
            await self.voice.speak(clean_response)
        
        return response_text
    
    def _clean_response_for_tts(self, response: str) -> str:
        """清理响应文本用于TTS"""
        import re
        # 移除JSON
        cleaned = re.sub(r'\{[^}]+\}', '', response)
        # 移除多余空白
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    async def run_loop(self):
        """运行交互循环"""
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
        """停止"""
        self._running = False
        self.voice.stop()


# =============================================================================
# 测试
# =============================================================================

async def _test():
    """测试语音接口"""
    # 测试TTS
    tts = TTSEngine()
    print("Testing TTS...")
    audio = await tts.synthesize("你好，我是智能座舱助手小智")
    print(f"Generated {len(audio)} bytes of audio")
    
    # 如果有音频设备，播放
    if HAS_SOUNDDEVICE and audio:
        player = AudioPlayer()
        await player.play_bytes(audio)


if __name__ == "__main__":
    asyncio.run(_test())
