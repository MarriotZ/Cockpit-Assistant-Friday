"""
Cockpit Assistant - 智能座舱助手主类

整合LLM推理引擎、函数调用和车辆控制
"""

import asyncio
import json
import os
from typing import AsyncIterator, Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from queue import Queue
from threading import Thread
import logging

# 尝试导入C++引擎，如果失败则使用模拟引擎
try:
    from cockpit_engine_py import LLMEngine, Message, GenerationConfig, FunctionCall
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False
    print("Warning: C++ engine not available, using mock engine")

from vehicle_controller import VehicleController
from function_registry import FunctionRegistry, get_function_prompt

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据类型定义
# =============================================================================

@dataclass
class ChatMessage:
    """聊天消息"""
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
# 模拟引擎（用于测试，当C++引擎不可用时）
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
    """模拟LLM引擎（用于测试）"""
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 35):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._context_usage = 0
        logger.info(f"MockLLMEngine initialized (model: {model_path})")
    
    def is_initialized(self) -> bool:
        return True
    
    def generate(self, messages: List[MockMessage], config: MockGenerationConfig = None) -> str:
        """非流式生成"""
        return self._mock_response(messages)
    
    def generate_stream(self, messages: List[MockMessage], callback: Callable, 
                       config: MockGenerationConfig = None) -> str:
        """流式生成"""
        response = self._mock_response(messages)
        
        # 模拟流式输出
        for char in response:
            callback(char, False)
        callback("", True)
        
        return response
    
    def _mock_response(self, messages: List[MockMessage]) -> str:
        """生成模拟响应"""
        if not messages:
            return "你好！有什么可以帮助你的吗？"
        
        last_msg = messages[-1].content.lower() if messages else ""
        
        # 简单的关键词匹配
        if "空调" in last_msg:
            if "打开" in last_msg or "开" in last_msg:
                temp = 24
                if "度" in last_msg:
                    import re
                    match = re.search(r'(\d+)', last_msg)
                    if match:
                        temp = int(match.group(1))
                return f'{{"name": "control_air_conditioner", "arguments": {{"action": "on", "temperature": {temp}}}}}\n好的，我来帮您打开空调。'
            elif "关" in last_msg:
                return '{"name": "control_air_conditioner", "arguments": {"action": "off"}}\n好的，已关闭空调。'
        
        elif "车窗" in last_msg:
            action = "open" if "打开" in last_msg or "开" in last_msg else "close"
            return f'{{"name": "control_window", "arguments": {{"position": "all", "action": "{action}"}}}}\n好的，正在操作车窗。'
        
        elif "导航" in last_msg or "去" in last_msg:
            # 提取目的地
            dest = "目的地"
            keywords = ["去", "到", "导航到", "带我去"]
            for kw in keywords:
                if kw in last_msg:
                    idx = last_msg.find(kw) + len(kw)
                    dest = last_msg[idx:].strip()[:20]
                    break
            return f'{{"name": "navigate_to", "arguments": {{"destination": "{dest}"}}}}\n好的，正在为您规划路线。'
        
        elif "播放" in last_msg or "音乐" in last_msg or "歌" in last_msg:
            query = "流行音乐"
            if "播放" in last_msg:
                idx = last_msg.find("播放") + 2
                query = last_msg[idx:].strip()[:20] or "流行音乐"
            return f'{{"name": "play_music", "arguments": {{"query": "{query}", "action": "play"}}}}\n好的，正在播放音乐。'
        
        elif "状态" in last_msg or "电量" in last_msg or "续航" in last_msg:
            return '{"name": "get_vehicle_status", "arguments": {"info_type": "all"}}\n好的，我来查询车辆状态。'
        
        elif "天气" in last_msg:
            return '{"name": "get_weather", "arguments": {"type": "current"}}\n好的，我来查询天气。'
        
        return "好的，我明白了。还有什么需要帮助的吗？"
    
    def parse_function_call(self, response: str) -> Optional[MockFunctionCall]:
        """解析函数调用"""
        try:
            # 查找JSON
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


# 根据是否有C++引擎选择使用的类
if HAS_CPP_ENGINE:
    EngineClass = LLMEngine
    MessageClass = Message
    ConfigClass = GenerationConfig
else:
    EngineClass = MockLLMEngine
    MessageClass = MockMessage
    ConfigClass = MockGenerationConfig


# =============================================================================
# 智能座舱助手
# =============================================================================

class CockpitAssistant:
    """
    智能座舱助手
    
    整合LLM推理和车辆控制功能
    """
    
    # 系统提示词模板
    SYSTEM_PROMPT_TEMPLATE = '''你是一个智能汽车座舱助手，名叫小智，负责帮助驾驶员控制车辆功能。

## 你的能力
1. 控制空调（开关、调节温度和风量）
2. 控制车窗（打开、关闭、半开）
3. 设置导航目的地
4. 播放音乐
5. 查询车辆状态（电量、胎压、里程等）
6. 控制车灯
7. 控制座椅（加热、通风、按摩）
8. 拨打电话
9. 查询天气

## 响应规则
1. 用简洁友好的语气回复用户
2. 回复要简短，适合语音播报（一般不超过50字）
3. 当需要执行车辆控制时，在回复的最前面以JSON格式返回函数调用
4. 函数调用格式: {{"name": "函数名", "arguments": {{"参数": "值"}}}}
5. 如果用户请求不清楚，礼貌地询问更多信息
6. 安全第一，如果检测到危险操作请提醒用户

## 可用函数
{functions}

## 示例对话
用户: 把空调打开，温度调到26度
助手: {{"name": "control_air_conditioner", "arguments": {{"action": "on", "temperature": 26}}}}
好的，已为您打开空调，温度设置为26度。

用户: 导航到北京天安门
助手: {{"name": "navigate_to", "arguments": {{"destination": "北京天安门"}}}}
好的，正在为您规划前往北京天安门的路线。

用户: 查一下车还有多少电
助手: {{"name": "get_vehicle_status", "arguments": {{"info_type": "battery"}}}}
好的，我来查看电池状态。'''

    def __init__(
        self, 
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 35,
        max_history: int = 20
    ):
        """
        初始化座舱助手
        
        Args:
            model_path: 模型文件路径
            n_ctx: 上下文长度
            n_gpu_layers: GPU层数
            max_history: 保留的最大对话历史轮数
        """
        # 初始化LLM引擎
        logger.info(f"Loading model: {model_path}")
        self.engine = EngineClass(model_path, n_ctx, n_gpu_layers)
        
        # 初始化车辆控制器
        self.controller = VehicleController()
        
        # 初始化函数注册表
        self.function_registry = FunctionRegistry()
        
        # 对话历史
        self.conversation_history: List[ChatMessage] = []
        self.max_history = max_history
        
        # 生成配置
        self.gen_config = ConfigClass()
        self.gen_config.temperature = 0.7
        self.gen_config.max_tokens = 256
        
        # 构建系统提示词
        self._system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            functions=get_function_prompt()
        )
        
        logger.info("CockpitAssistant initialized successfully")
    
    async def chat(self, user_input: str) -> AsyncIterator[str]:
        """
        处理用户输入，流式返回响应
        
        Args:
            user_input: 用户输入文本
            
        Yields:
            响应文本片段
        """
        # 添加用户消息
        self.conversation_history.append(ChatMessage(role="user", content=user_input))
        
        # 构建消息列表
        messages = self._build_messages()
        
        # 流式生成
        full_response = ""
        
        # 使用队列在线程间传递token
        token_queue: Queue = Queue()
        
        def stream_callback(token: str, is_end: bool):
            token_queue.put((token, is_end))
        
        # 在后台线程运行推理
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
        
        # 流式返回token
        while True:
            try:
                token, is_end = await asyncio.get_event_loop().run_in_executor(
                    None, token_queue.get, True, 30.0
                )
                
                if is_end:
                    break
                
                full_response += token
                yield token
                
            except Exception:
                break
        
        inference_thread.join()
        
        # 检查是否有函数调用
        function_call = self.engine.parse_function_call(full_response)
        
        if function_call:
            # 执行函数
            try:
                args = json.loads(function_call.arguments) if isinstance(function_call.arguments, str) else function_call.arguments
                result = await self.controller.execute(function_call.name, args)
                
                # 返回执行结果
                yield f"\n\n✅ {result}"
                full_response += f"\n\n{result}"
                
            except Exception as e:
                error_msg = f"\n\n❌ 执行失败: {str(e)}"
                yield error_msg
                full_response += error_msg
        
        # 保存助手回复
        self.conversation_history.append(ChatMessage(
            role="assistant", 
            content=full_response,
            function_call={"name": function_call.name, "arguments": function_call.arguments} if function_call else None
        ))
        
        # 限制历史长度
        self._trim_history()
    
    async def chat_sync(self, user_input: str) -> str:
        """
        同步版本的聊天（非流式）
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            完整的响应文本
        """
        full_response = ""
        async for token in self.chat(user_input):
            full_response += token
        return full_response
    
    def _build_messages(self) -> List:
        """构建消息列表"""
        messages = [MessageClass("system", self._system_prompt)]
        
        # 添加历史消息
        for msg in self.conversation_history[-self.max_history:]:
            messages.append(MessageClass(msg.role, msg.content))
        
        return messages
    
    def _trim_history(self):
        """裁剪对话历史"""
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def reset_conversation(self):
        """重置对话"""
        self.conversation_history.clear()
        self.engine.clear_cache()
        logger.info("Conversation reset")
    
    def get_vehicle_state(self) -> Dict[str, Any]:
        """获取当前车辆状态"""
        return self.controller.get_state_summary()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
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
# 异步助手包装器（用于更简单的异步调用）
# =============================================================================

class AsyncCockpitAssistant:
    """异步座舱助手包装器"""
    
    def __init__(self, *args, **kwargs):
        self._assistant = CockpitAssistant(*args, **kwargs)
    
    async def chat(self, user_input: str) -> AsyncIterator[str]:
        """流式聊天"""
        async for token in self._assistant.chat(user_input):
            yield token
    
    async def chat_complete(self, user_input: str) -> str:
        """完整响应"""
        return await self._assistant.chat_sync(user_input)
    
    def reset(self):
        """重置"""
        self._assistant.reset_conversation()
    
    @property
    def vehicle_state(self) -> Dict[str, Any]:
        """车辆状态"""
        return self._assistant.get_vehicle_state()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        return self._assistant.get_stats()


# =============================================================================
# 测试代码
# =============================================================================

async def _test():
    """测试函数"""
    # 使用模拟引擎测试
    assistant = CockpitAssistant("mock_model.gguf")
    
    test_inputs = [
        "你好",
        "把空调打开，温度调到26度",
        "帮我导航到北京天安门",
        "查一下车辆状态",
        "播放周杰伦的歌",
    ]
    
    for user_input in test_inputs:
        print(f"\n用户: {user_input}")
        print("助手: ", end="", flush=True)
        async for token in assistant.chat(user_input):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(_test())
