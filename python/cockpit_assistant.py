"""
Cockpit Assistant - Intelligent Cockpit Assistant Class

"""

import asyncio
import json
import os
import logging
from typing import AsyncIterator, Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from pathlib import Path

import os
from pathlib import Path

def _add_dll_dirs():
    # Current python directory (for pyd files, you may also place dll here)
    py_dir = Path(__file__).resolve().parent
    root = py_dir.parent  # Project root directory: .../Cockpit-Assistant-Friday

    # Your llama.cpp dll directory
    llama_bin = root / "third_party" / "llama.cpp" / "build" / "bin" / "Release"

    # Your project build dll directory (if cockpit_engine.dll is here)
    build_bin = root / "build" / "bin" / "Release"

    # CUDA directory (actual location on your machine)
    cuda_bin_x64 = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")

    for p in [cuda_bin_x64, llama_bin, build_bin, py_dir]:
        if p.exists():
            os.add_dll_directory(str(p))

_add_dll_dirs()


# Import C++ engine, use mock if import fails
try:
    from cockpit_engine_py import LLMEngine, Message, GenerationConfig, FunctionCall
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False  
    print("Warning: C++ engine not available, using mock engine")

from vehicle_controller import VehicleController
from function_registry import FunctionRegistry, get_function_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Type Definitions
# =============================================================================

@dataclass
class ChatMessage:
    """Chat message"""
    role: str           # system, user, assistant
    content: str
    function_call: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "function_call": self.function_call
        }


# =============================================================================
# Mock Classes (for testing when C++ engine is unavailable)
# =============================================================================

class MockMessage:
    def __init__(self, role: str = "", content: str = ""):
        self.role = role
        self.content = content


class MockGenerationConfig:
    def __init__(self):
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 40
        self.max_tokens = 512
        self.repeat_penalty = 1.1
        self.stop_sequences = ["<|im_end|>"]


class MockFunctionCall:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments


class MockLLMEngine:
    """Mock LLM Engine (for testing)"""
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._context_usage = 0
        logger.info(f"MockLLMEngine initialized (model: {model_path})")
    
    def is_initialized(self) -> bool:
        return True
    
    def generate(self, messages: List[MockMessage], config: MockGenerationConfig = None) -> str:
        """Non-streaming generation"""
        return self._mock_response(messages)
    
    def generate_stream(self, messages: List[MockMessage], callback: Callable, 
                       config: MockGenerationConfig = None) -> str:
        """Streaming generation"""
        response = self._mock_response(messages)
        
        # Simulate streaming output
        for char in response:
            callback(char, False)
        callback("", True)
        
        return response
    
    def _mock_response(self, messages: List[MockMessage]) -> str:
        """Generate mock response"""
        if not messages:
            return "Hello! How can I help you today?"
        
        last_msg = messages[-1].content.lower() if messages else ""
        
        # Simple keyword matching
        if "air" in last_msg or "ac" in last_msg or "climate" in last_msg:
            if "on" in last_msg or "turn on" in last_msg or "start" in last_msg:
                temp = 24
                if "degree" in last_msg or "°" in last_msg:
                    import re
                    match = re.search(r'(\d+)', last_msg)
                    if match:
                        temp = int(match.group(1))
                return f'{{"name": "control_air_conditioner", "arguments": {{"action": "on", "temperature": {temp}}}}}\nSure, turning on the air conditioner.'
            elif "off" in last_msg or "turn off" in last_msg:
                return '{"name": "control_air_conditioner", "arguments": {"action": "off"}}\nOkay, air conditioner turned off.'
        
        elif "window" in last_msg:
            action = "open" if "open" in last_msg else "close"
            return f'{{"name": "control_window", "arguments": {{"position": "all", "action": "{action}"}}}}\nSure, operating the windows.'
        
        elif "navigate" in last_msg or "navigation" in last_msg or "go to" in last_msg:
            # Extract destination
            dest = "destination"
            keywords = ["to ", "go to", "navigate to"]
            for kw in keywords:
                if kw in last_msg:
                    idx = last_msg.find(kw) + len(kw)
                    dest = last_msg[idx:].strip()[:20]
                    break
            return f'{{"name": "navigate_to", "arguments": {{"destination": "{dest}"}}}}\nOkay, planning route for you.'
        
        elif "play" in last_msg or "music" in last_msg or "song" in last_msg:
            query = "pop music"
            if "play" in last_msg:
                idx = last_msg.find("play") + 4
                query = last_msg[idx:].strip()[:20] or "pop music"
            return f'{{"name": "play_music", "arguments": {{"query": "{query}", "action": "play"}}}}\nSure, playing music.'
        
        elif "status" in last_msg or "battery" in last_msg or "range" in last_msg:
            return '{"name": "get_vehicle_status", "arguments": {"info_type": "all"}}\nOkay, checking vehicle status.'
        
        elif "weather" in last_msg:
            return '{"name": "get_weather", "arguments": {"type": "current"}}\nSure, checking the weather.'
        
        return "Understood. Anything else I can help with?"
    
    def parse_function_call(self, response: str) -> Optional[MockFunctionCall]:
        """Parse function call"""
        try:
            # Find JSON
            import re
            pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
            match = re.search(pattern, response)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                return MockFunctionCall(
                    name=data.get("name", ""),
                    arguments=json.dumps(data.get("arguments", {}))
                )
        except:
            pass
        return None
    
    def clear_cache(self):
        self._context_usage = 0
    
    def get_context_usage(self) -> int:
        return self._context_usage
    
    def get_max_context(self) -> int:
        return self.n_ctx
    
    def get_model_info(self) -> str:
        return f"MockEngine: {self.model_path}"
    
    def stop_generation(self):
        pass


# Select classes based on C++ engine availability
if HAS_CPP_ENGINE:
    EngineClass = LLMEngine
    MessageClass = Message
    ConfigClass = GenerationConfig
else:
    EngineClass = MockLLMEngine
    MessageClass = MockMessage
    ConfigClass = MockGenerationConfig


# =============================================================================
# Intelligent Cockpit Assistant
# =============================================================================

class CockpitAssistant:
    """
    Intelligent Cockpit Assistant
    
    Integrates LLM inference and vehicle control functions
    """
    
    # System prompt template
    SYSTEM_PROMPT_TEMPLATE = '''You are an intelligent automotive cockpit assistant named Friday, responsible for helping drivers control vehicle functions.

## Your Capabilities
1. Control air conditioning (on/off, adjust temperature and fan speed)
2. Control windows (open, close, half-open)
3. Set navigation destination
4. Play music
5. Query vehicle status (battery, tire pressure, mileage, etc.)
6. Control lights
7. Control seats (heating, ventilation, massage)
8. Make phone calls
9. Query weather

## Response Rules
1. Keep replies brief and suitable for voice playback (generally no more than 50 words)
2. When vehicle control is needed, return function call in JSON format at the beginning of the reply
3. Function call format: {{"name": "function_name", "arguments": {{"parameter": "value"}}}}
4. If user request is unclear, politely ask for more information

## Available Functions
{functions}

## Example Conversations
User: Turn on the AC and set temperature to 26 degrees
Assistant: {{"name": "control_air_conditioner", "arguments": {{"action": "on", "temperature": 26}}}}
Sure, I've turned on the air conditioner and set the temperature to 26 degrees.

User: Navigate to Times Square New York
Assistant: {{"name": "navigate_to", "arguments": {{"destination": "Times Square New York"}}}}
Okay, planning route to Times Square New York.

User: Check how much battery is left
Assistant: {{"name": "get_vehicle_status", "arguments": {{"info_type": "battery"}}}}
Sure, let me check the battery status.
'''

    def __init__(
        self, 
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 35,
        max_history: int = 20,
        detailed_functions: bool = False
    ):
        """
        Initialize cockpit assistant
        
        Args:
            model_path: Model file path
            n_ctx: Context length
            n_gpu_layers: Number of GPU layers (0 means CPU mode)
            max_history: Maximum number of conversation history rounds to keep
            detailed_functions: Whether to use detailed function list
                               True: Full format (suitable for 7B+ large models)
                               False: Concise format (suitable for 3B small models)
        """
        # Initialize LLM engine
        logger.info(f"Loading model: {model_path}")
        self.engine = EngineClass(model_path, n_ctx, n_gpu_layers)
        
        # Initialize vehicle controller
        self.controller = VehicleController()
        
        # Initialize function registry
        self.function_registry = FunctionRegistry()
        
        # Conversation history
        self.conversation_history: List[ChatMessage] = []
        self.max_history = max_history
        
        # Generation config
        self.gen_config = ConfigClass()
        self.gen_config.temperature = 0.7
        self.gen_config.max_tokens = 256
        
        # Build system prompt
        self._system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            functions=get_function_prompt(detailed=detailed_functions)
        )
        
        logger.info("CockpitAssistant initialized successfully")
    
    async def chat(self, user_input: str) -> AsyncIterator[str]:
        """
        Process user input, return response in streaming manner
        
        Args:
            user_input: User input text
            
        Yields:
            Response text fragments
        """
        # Add user message
        self.conversation_history.append(ChatMessage(role="user", content=user_input))
        
        # Build message list
        messages = self._build_messages()
        
        # Streaming generation
        full_response = ""
        
        # Use queue to pass tokens between threads
        token_queue: Queue = Queue()
        
        def stream_callback(token: str, is_end: bool):
            token_queue.put((token, is_end))
        
        # Run inference in background thread
        def run_inference():
            try:
                result = self.engine.generate_stream(messages, stream_callback, self.gen_config)
                return result
            except Exception as e:
                logger.error(f"Inference error: {e}")
                token_queue.put(("", True))
                return ""
        
        inference_thread = Thread(target=run_inference)
        inference_thread.start()
        
        # Stream tokens
        from queue import Empty as QueueEmpty
        
        while True:
            try:
                # Use run_in_executor to execute blocking queue.get operation in thread pool
                loop = asyncio.get_event_loop()
                token, is_end = await loop.run_in_executor(
                    None,
                    lambda: token_queue.get(timeout=30.0)
                )
                
                if is_end:
                    break
                
                full_response += token
                yield token
                
            except QueueEmpty:
                # Timeout, check if thread is still running
                if not inference_thread.is_alive():
                    break
                continue
            except Exception as e:
                logger.error(f"Error in chat stream: {e}", exc_info=True)
                break
        
        inference_thread.join()
        
        # Check for function call
        function_call = self.engine.parse_function_call(full_response)
        
        if function_call:
            # Execute function
            try:
                args = json.loads(function_call.arguments) if isinstance(function_call.arguments, str) else function_call.arguments
                result = await self.controller.execute(function_call.name, args)
                
                # Return execution result
                yield f"\n\n✅ {result}"
                full_response += f"\n\n{result}"
                
            except Exception as e:
                error_msg = f"\n\n❌ Execution failed: {str(e)}"
                yield error_msg
                full_response += error_msg
        
        # Save assistant reply
        self.conversation_history.append(ChatMessage(
            role="assistant", 
            content=full_response,
            function_call={"name": function_call.name, "arguments": function_call.arguments} if function_call else None
        ))
        
        # Limit history length
        self._trim_history()
    
    async def chat_sync(self, user_input: str) -> str:
        """
        Synchronous version of chat (non-streaming)
        
        Args:
            user_input: User input text
            
        Returns:
            Complete response text
        """
        full_response = ""
        async for token in self.chat(user_input):
            full_response += token
        return full_response
    
    def _build_messages(self) -> List:
        """Build message list"""
        messages = [MessageClass("system", self._system_prompt)]
        
        # Add history messages
        for msg in self.conversation_history[-self.max_history:]:
            messages.append(MessageClass(msg.role, msg.content))
        
        return messages
    
    def _trim_history(self):
        """Trim conversation history"""
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def reset_conversation(self):
        """Reset conversation"""
        self.conversation_history.clear()
        self.engine.clear_cache()
        logger.info("Conversation reset")
    
    def get_vehicle_state(self) -> Dict[str, Any]:
        """Get current vehicle status"""
        return self.controller.get_state_summary()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        if hasattr(self.engine, 'get_stats'):
            stats = self.engine.get_stats()
            return {
                "tokens_generated": stats.tokens_generated,
                "generation_time_ms": stats.generation_time_ms,
                "tokens_per_second": stats.tokens_per_second,
                "context_usage": self.engine.get_context_usage(),
                "max_context": self.engine.get_max_context()
            }
        return {
            "context_usage": self.engine.get_context_usage(),
            "max_context": self.engine.get_max_context()
        }


# =============================================================================
# Async Assistant Wrapper (for simpler async calls)
# =============================================================================

class AsyncCockpitAssistant:
    """Async cockpit assistant wrapper"""
    
    def __init__(self, *args, **kwargs):
        self._assistant = CockpitAssistant(*args, **kwargs)
    
    async def chat(self, user_input: str) -> AsyncIterator[str]:
        """Streaming chat"""
        async for token in self._assistant.chat(user_input):
            yield token
    
    async def chat_complete(self, user_input: str) -> str:
        """Complete response"""
        return await self._assistant.chat_sync(user_input)
    
    def reset(self):
        """Reset"""
        self._assistant.reset_conversation()
    
    @property
    def vehicle_state(self) -> Dict[str, Any]:
        """Vehicle status"""
        return self._assistant.get_vehicle_state()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Statistics"""
        return self._assistant.get_stats()


# =============================================================================
# Test Code
# =============================================================================

async def _test():
    """Test function"""
    # Use actual model path (modify according to your actual situation)
    model_path = r"E:/projects/Cockpit-Assistant-Friday/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    # If file doesn't exist, use relative path
    if not os.path.exists(model_path):
        model_path = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    # If still doesn't exist, use mock engine
    if not os.path.exists(model_path):
        print(f"Warning: Model file does not exist: {model_path}")
        print("Will use mock engine for testing")
        model_path = "mock_model.gguf"
    
    assistant = CockpitAssistant(model_path)
    
    test_inputs = [
        "Hello",
        "Turn on the AC and set temperature to 26 degrees",
        "Navigate to Times Square New York",
        "Check vehicle status",
        "Play some music by Taylor Swift",
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        print("Assistant: ", end="", flush=True)
        async for token in assistant.chat(user_input):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(_test())