"""
Cockpit Assistant Class
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


def _add_dll_dirs():
    """Add DLL directories for (Windows) CUDA support"""
    py_dir = Path(__file__).resolve().parent
    root = py_dir.parent

    llama_bin = root / "third_party" / "llama.cpp" / "build" / "bin" / "Release"
    build_bin = root / "build" / "bin" / "Release"
    cuda_bin_x64 = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")

    for p in [cuda_bin_x64, llama_bin, build_bin, py_dir]:
        if p.exists():
            os.add_dll_directory(str(p))
# if your system has CUDA and use Windows, please uncomment the following line
#_add_dll_dirs()


try:
    from cockpit_engine_py import LLMEngine, Message, GenerationConfig, FunctionCall
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False  
    print("Warning: C++ engine not available, using mock engine")

from vehicle_controller import VehicleController
from function_registry import FunctionRegistry, get_function_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
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


if HAS_CPP_ENGINE:
    EngineClass = LLMEngine
    MessageClass = Message
    ConfigClass = GenerationConfig


# =============================================================================
# Cockpit Assistant
# =============================================================================

class CockpitAssistant:
    
    SYSTEM_PROMPT_TEMPLATE = '''You are an intelligent car cockpit assistant named Friday, helping drivers control vehicle functions.

## Capabilities
1. Control AC (on/off, temperature, fan speed)
2. Control windows (open, close, half-open)
3. Set navigation destination
4. Play music
5. Query vehicle status (battery, tire pressure, mileage, etc.)
6. Control lights
7. Control seats (heating, ventilation, massage)
8. Make phone calls
9. Query weather

## Response Rules
1. Keep responses brief, suitable for voice (under 50 words)
2. When executing vehicle control, return function call in JSON format at the beginning
3. Function call format: {{"name": "function_name", "arguments": {{"param": "value"}}}}
4. If user request is unclear, politely ask for more information

## Available Functions
{functions}

## Example Dialogues
User: Turn on the AC, set temperature to 26
Assistant: {{"name": "control_air_conditioner", "arguments": {{"action": "on", "temperature": 26}}}}
OK, AC is on and temperature set to 26°C.

User: Navigate to Shanghai Oriental Pearl Tower
Assistant: {{"name": "navigate_to", "arguments": {{"destination": "Shanghai Oriental Pearl Tower"}}}}
OK, planning route to Shanghai Oriental Pearl Tower.

User: Check how much battery is left
Assistant: {{"name": "get_vehicle_status", "arguments": {{"info_type": "battery"}}}}
OK, checking battery status.
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
            n_gpu_layers: GPU layers (0 for CPU mode)
            max_history: Maximum conversation history rounds to keep
            detailed_functions: Whether to use detailed function list
        """
        logger.info(f"Loading model: {model_path}")
        self.engine = EngineClass(model_path, n_ctx, n_gpu_layers)
        
        self.controller = VehicleController()
        self.function_registry = FunctionRegistry()
        
        self.conversation_history: List[ChatMessage] = []
        self.max_history = max_history
        
        self.gen_config = ConfigClass()
        self.gen_config.temperature = 0.7
        self.gen_config.max_tokens = 256
        
        self._system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            functions=get_function_prompt(detailed=detailed_functions)
        )
        
        logger.info("CockpitAssistant initialized successfully")
    
    async def chat(self, user_input: str) -> AsyncIterator[str]:
        """
        Process user input, stream response
        
        Args:
            user_input: User input text
            
        Yields:
            Response text fragments
        """
        self.conversation_history.append(ChatMessage(role="user", content=user_input))
        
        messages = self._build_messages()
        
        full_response = ""
        token_queue: Queue = Queue()
        
        def stream_callback(token: str, is_end: bool):
            token_queue.put((token, is_end))
        
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
        
        from queue import Empty as QueueEmpty
        
        while True:
            try:
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
                if not inference_thread.is_alive():
                    break
                continue
            except Exception as e:
                logger.error(f"Error in chat stream: {e}", exc_info=True)
                break
        
        inference_thread.join()
        
        function_call = self.engine.parse_function_call(full_response)
        
        if function_call:
            try:
                args = json.loads(function_call.arguments) if isinstance(function_call.arguments, str) else function_call.arguments
                result = await self.controller.execute(function_call.name, args)
                
                yield f"\n\n✅ {result}"
                full_response += f"\n\n{result}"
                
            except Exception as e:
                error_msg = f"\n\n❌ Execution failed: {str(e)}"
                yield error_msg
                full_response += error_msg
        
        self.conversation_history.append(ChatMessage(
            role="assistant", 
            content=full_response,
            function_call={"name": function_call.name, "arguments": function_call.arguments} if function_call else None
        ))
        
        self._trim_history()
    
    async def chat_sync(self, user_input: str) -> str:
        """
        Synchronous chat (non-streaming)
        
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
        """Get current vehicle state"""
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


class AsyncCockpitAssistant:
    """Async wrapper for CockpitAssistant"""
    
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
        """Reset conversation"""
        self._assistant.reset_conversation()
    
    @property
    def vehicle_state(self) -> Dict[str, Any]:
        """Vehicle state"""
        return self._assistant.get_vehicle_state()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Statistics"""
        return self._assistant.get_stats()


async def _test():
    """Test the cockpit assistant"""
    model_path = r"../models/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    if not os.path.exists(model_path):
        model_path = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    assistant = CockpitAssistant(model_path)
    
    test_inputs = [
        "Hello",
        "Turn on the AC, set temperature to 26",
        "Navigate to Shanghai Oriental Pearl Tower",
        "Check vehicle status",
        "Play Jay Chou's songs",
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        print("Assistant: ", end="", flush=True)
        async for token in assistant.chat(user_input):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(_test())