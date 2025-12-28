#!/usr/bin/env python3
"""
测试套件 - 智能座舱助手

运行: pytest tests/test_assistant.py -v
"""

import pytest
import asyncio
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from vehicle_controller import VehicleController
from function_registry import FunctionRegistry, FunctionDefinition, FunctionParameter
from cockpit_assistant import CockpitAssistant


# ==============================================================================
# 车辆控制器测试
# ==============================================================================

class TestVehicleController:
    """测试车辆控制器"""
    
    @pytest.fixture
    def controller(self):
        return VehicleController()
    
    @pytest.mark.asyncio
    async def test_air_conditioner_on(self, controller):
        """测试打开空调"""
        result = await controller.execute(
            "control_air_conditioner",
            {"action": "on", "temperature": 26}
        )
        
        assert "打开" in result or "开启" in result
        assert controller.state.ac.is_on
        assert controller.state.ac.temperature == 26
    
    @pytest.mark.asyncio
    async def test_air_conditioner_off(self, controller):
        """测试关闭空调"""
        # 先打开
        await controller.execute("control_air_conditioner", {"action": "on"})
        
        # 再关闭
        result = await controller.execute(
            "control_air_conditioner",
            {"action": "off"}
        )
        
        assert "关闭" in result
        assert not controller.state.ac.is_on
    
    @pytest.mark.asyncio
    async def test_window_control(self, controller):
        """测试车窗控制"""
        result = await controller.execute(
            "control_window",
            {"position": "front_left", "action": "open"}
        )
        
        assert "左前" in result
        assert "打开" in result
        assert controller.state.windows.front_left == "open"
    
    @pytest.mark.asyncio
    async def test_window_all(self, controller):
        """测试控制所有车窗"""
        result = await controller.execute(
            "control_window",
            {"position": "all", "action": "close"}
        )
        
        assert "全部" in result
        assert controller.state.windows.front_left == "close"
        assert controller.state.windows.front_right == "close"
        assert controller.state.windows.rear_left == "close"
        assert controller.state.windows.rear_right == "close"
    
    @pytest.mark.asyncio
    async def test_navigation(self, controller):
        """测试导航"""
        result = await controller.execute(
            "navigate_to",
            {"destination": "北京天安门"}
        )
        
        assert "北京天安门" in result
        assert controller.state.destination == "北京天安门"
        assert controller.state.navigation_active
    
    @pytest.mark.asyncio
    async def test_music_play(self, controller):
        """测试播放音乐"""
        result = await controller.execute(
            "play_music",
            {"query": "周杰伦", "action": "play"}
        )
        
        assert "周杰伦" in result or "播放" in result
        assert controller.state.music_playing
    
    @pytest.mark.asyncio
    async def test_music_pause(self, controller):
        """测试暂停音乐"""
        # 先播放
        await controller.execute("play_music", {"query": "test", "action": "play"})
        
        # 再暂停
        result = await controller.execute("play_music", {"action": "pause"})
        
        assert "暂停" in result
        assert not controller.state.music_playing
    
    @pytest.mark.asyncio
    async def test_vehicle_status(self, controller):
        """测试车辆状态查询"""
        result = await controller.execute(
            "get_vehicle_status",
            {"info_type": "battery"}
        )
        
        assert "电量" in result or "电池" in result
        assert "%" in result
    
    @pytest.mark.asyncio
    async def test_vehicle_status_all(self, controller):
        """测试查询所有状态"""
        result = await controller.execute(
            "get_vehicle_status",
            {"info_type": "all"}
        )
        
        assert "电量" in result or "电池" in result
        assert "胎压" in result
    
    @pytest.mark.asyncio
    async def test_lights_control(self, controller):
        """测试灯光控制"""
        result = await controller.execute(
            "control_lights",
            {"light_type": "headlight", "action": "on"}
        )
        
        assert "大灯" in result
        assert controller.state.lights.headlight == "on"
    
    @pytest.mark.asyncio
    async def test_seat_heating(self, controller):
        """测试座椅加热"""
        result = await controller.execute(
            "control_seat",
            {"seat": "driver", "function": "heating", "level": 2}
        )
        
        assert "加热" in result
        assert controller.state.driver_seat.heating == 2
    
    @pytest.mark.asyncio
    async def test_phone_call(self, controller):
        """测试电话拨打"""
        result = await controller.execute(
            "make_phone_call",
            {"action": "call", "contact": "张三"}
        )
        
        assert "张三" in result or "拨打" in result
        assert controller.state.call_active
    
    @pytest.mark.asyncio
    async def test_unknown_function(self, controller):
        """测试未知函数"""
        result = await controller.execute("unknown_function", {})
        
        assert "未知" in result
    
    def test_state_summary(self, controller):
        """测试状态摘要"""
        summary = controller.get_state_summary()
        
        assert "ac" in summary
        assert "windows" in summary
        assert "navigation" in summary
        assert "music" in summary
        assert "battery" in summary


# ==============================================================================
# 函数注册表测试
# ==============================================================================

class TestFunctionRegistry:
    """测试函数注册表"""
    
    @pytest.fixture
    def registry(self):
        return FunctionRegistry()
    
    def test_default_functions(self, registry):
        """测试默认函数已注册"""
        assert "control_air_conditioner" in registry.functions
        assert "control_window" in registry.functions
        assert "navigate_to" in registry.functions
        assert "play_music" in registry.functions
        assert "get_vehicle_status" in registry.functions
    
    def test_get_function(self, registry):
        """测试获取函数定义"""
        func = registry.get("control_air_conditioner")
        
        assert func is not None
        assert func.name == "control_air_conditioner"
        assert len(func.parameters) > 0
    
    def test_to_schema(self, registry):
        """测试转换为JSON Schema"""
        func = registry.get("control_air_conditioner")
        schema = func.to_schema()
        
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert "properties" in schema["parameters"]
    
    def test_json_schema(self, registry):
        """测试生成完整JSON Schema"""
        json_str = registry.to_json_schema()
        data = json.loads(json_str)
        
        assert "functions" in data
        assert len(data["functions"]) > 0
    
    def test_register_custom_function(self, registry):
        """测试注册自定义函数"""
        custom_func = FunctionDefinition(
            name="test_function",
            description="测试函数",
            parameters=[
                FunctionParameter(
                    name="test_param",
                    type="string",
                    description="测试参数",
                    required=True
                )
            ]
        )
        
        registry.register(custom_func)
        
        assert "test_function" in registry.functions
        assert registry.get("test_function").name == "test_function"


# ==============================================================================
# 座舱助手测试
# ==============================================================================

class TestCockpitAssistant:
    """测试座舱助手"""
    
    @pytest.fixture
    def assistant(self):
        # 使用模拟模型
        return CockpitAssistant("mock_model.gguf")
    
    @pytest.mark.asyncio
    async def test_chat_basic(self, assistant):
        """测试基本对话"""
        response = ""
        async for token in assistant.chat("你好"):
            response += token
        
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_chat_air_conditioner(self, assistant):
        """测试空调控制对话"""
        response = ""
        async for token in assistant.chat("把空调打开"):
            response += token
        
        assert len(response) > 0
        # 应该触发空调控制
        assert assistant.controller.state.ac.is_on
    
    @pytest.mark.asyncio
    async def test_chat_sync(self, assistant):
        """测试同步对话"""
        response = await assistant.chat_sync("你好")
        
        assert len(response) > 0
    
    def test_reset_conversation(self, assistant):
        """测试重置对话"""
        assistant.conversation_history.append(
            type(assistant.conversation_history[0] if assistant.conversation_history else object)
        ) if assistant.conversation_history else None
        
        assistant.reset_conversation()
        
        assert len(assistant.conversation_history) == 0
    
    def test_get_vehicle_state(self, assistant):
        """测试获取车辆状态"""
        state = assistant.get_vehicle_state()
        
        assert "ac" in state
        assert "windows" in state
    
    def test_get_stats(self, assistant):
        """测试获取统计信息"""
        stats = assistant.get_stats()
        
        assert "context_usage" in stats
        assert "max_context" in stats


# ==============================================================================
# 运行测试
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
