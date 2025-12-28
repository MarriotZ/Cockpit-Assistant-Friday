"""
Function Registry - 车辆控制函数定义

定义智能座舱助手可调用的所有函数及其参数schema
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class FunctionParameter:
    """函数参数定义"""
    name: str
    type: str
    description: str
    required: bool = False
    enum: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Any = None

@dataclass
class FunctionDefinition:
    """函数定义"""
    name: str
    description: str
    parameters: List[FunctionParameter] = field(default_factory=list)
    
    def to_schema(self) -> Dict[str, Any]:
        """转换为JSON Schema格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.minimum is not None:
                prop["minimum"] = param.minimum
            if param.maximum is not None:
                prop["maximum"] = param.maximum
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class FunctionRegistry:
    """函数注册表"""
    
    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
        self._register_default_functions()
    
    def _register_default_functions(self):
        """注册默认的车辆控制函数"""
        
        # 空调控制
        self.register(FunctionDefinition(
            name="control_air_conditioner",
            description="控制车内空调，包括开关、温度调节、风量调节",
            parameters=[
                FunctionParameter(
                    name="action",
                    type="string",
                    description="操作类型：on(打开), off(关闭), adjust(调节)",
                    required=True,
                    enum=["on", "off", "adjust"]
                ),
                FunctionParameter(
                    name="temperature",
                    type="number",
                    description="目标温度（摄氏度）",
                    minimum=16,
                    maximum=30,
                    default=24
                ),
                FunctionParameter(
                    name="fan_speed",
                    type="integer",
                    description="风量档位（1-5）",
                    minimum=1,
                    maximum=5,
                    default=3
                ),
                FunctionParameter(
                    name="mode",
                    type="string",
                    description="空调模式",
                    enum=["auto", "cool", "heat", "fan", "dry"]
                )
            ]
        ))
        
        # 车窗控制
        self.register(FunctionDefinition(
            name="control_window",
            description="控制车窗的开关状态",
            parameters=[
                FunctionParameter(
                    name="position",
                    type="string",
                    description="车窗位置：front_left(左前), front_right(右前), rear_left(左后), rear_right(右后), all(全部)",
                    required=True,
                    enum=["front_left", "front_right", "rear_left", "rear_right", "all"]
                ),
                FunctionParameter(
                    name="action",
                    type="string",
                    description="操作类型：open(打开), close(关闭), half_open(半开)",
                    required=True,
                    enum=["open", "close", "half_open"]
                ),
                FunctionParameter(
                    name="percentage",
                    type="integer",
                    description="开启百分比（0-100）",
                    minimum=0,
                    maximum=100
                )
            ]
        ))
        
        # 导航
        self.register(FunctionDefinition(
            name="navigate_to",
            description="设置导航目的地，支持地点名称或地址",
            parameters=[
                FunctionParameter(
                    name="destination",
                    type="string",
                    description="目的地名称或地址",
                    required=True
                ),
                FunctionParameter(
                    name="via_points",
                    type="array",
                    description="途经点列表"
                ),
                FunctionParameter(
                    name="avoid",
                    type="array",
                    description="需要避开的路况（toll:收费站, highway:高速, ferry:轮渡）"
                ),
                FunctionParameter(
                    name="route_preference",
                    type="string",
                    description="路线偏好",
                    enum=["fastest", "shortest", "economical"]
                )
            ]
        ))
        
        # 音乐控制
        self.register(FunctionDefinition(
            name="play_music",
            description="播放音乐，支持搜索歌曲或控制播放",
            parameters=[
                FunctionParameter(
                    name="query",
                    type="string",
                    description="搜索关键词（歌曲名、歌手名、专辑名等）"
                ),
                FunctionParameter(
                    name="action",
                    type="string",
                    description="播放操作",
                    enum=["play", "pause", "stop", "next", "previous", "shuffle", "repeat"]
                ),
                FunctionParameter(
                    name="volume",
                    type="integer",
                    description="音量（0-100）",
                    minimum=0,
                    maximum=100
                ),
                FunctionParameter(
                    name="source",
                    type="string",
                    description="音乐来源",
                    enum=["local", "bluetooth", "usb", "online"]
                )
            ]
        ))
        
        # 车辆状态查询
        self.register(FunctionDefinition(
            name="get_vehicle_status",
            description="查询车辆各项状态信息",
            parameters=[
                FunctionParameter(
                    name="info_type",
                    type="string",
                    description="查询类型：battery(电池/油量), tire_pressure(胎压), oil(机油), mileage(里程), temperature(车内温度), all(全部)",
                    required=True,
                    enum=["battery", "tire_pressure", "oil", "mileage", "temperature", "doors", "lights", "all"]
                )
            ]
        ))
        
        # 车灯控制
        self.register(FunctionDefinition(
            name="control_lights",
            description="控制车辆灯光",
            parameters=[
                FunctionParameter(
                    name="light_type",
                    type="string",
                    description="灯光类型",
                    required=True,
                    enum=["headlight", "highbeam", "fog", "interior", "hazard", "turn_left", "turn_right"]
                ),
                FunctionParameter(
                    name="action",
                    type="string",
                    description="操作类型",
                    required=True,
                    enum=["on", "off", "auto"]
                ),
                FunctionParameter(
                    name="brightness",
                    type="integer",
                    description="亮度（仅适用于内饰灯）",
                    minimum=0,
                    maximum=100
                )
            ]
        ))
        
        # 座椅控制
        self.register(FunctionDefinition(
            name="control_seat",
            description="控制座椅位置和功能",
            parameters=[
                FunctionParameter(
                    name="seat",
                    type="string",
                    description="座椅位置",
                    required=True,
                    enum=["driver", "passenger", "rear_left", "rear_right"]
                ),
                FunctionParameter(
                    name="function",
                    type="string",
                    description="功能类型",
                    required=True,
                    enum=["heating", "cooling", "massage", "position", "memory"]
                ),
                FunctionParameter(
                    name="level",
                    type="integer",
                    description="强度/档位（1-3）",
                    minimum=0,
                    maximum=3
                ),
                FunctionParameter(
                    name="memory_slot",
                    type="integer",
                    description="记忆位置（1-3）",
                    minimum=1,
                    maximum=3
                )
            ]
        ))
        
        # 电话控制
        self.register(FunctionDefinition(
            name="make_phone_call",
            description="拨打电话或管理通话",
            parameters=[
                FunctionParameter(
                    name="action",
                    type="string",
                    description="操作类型",
                    required=True,
                    enum=["call", "answer", "hangup", "reject", "mute"]
                ),
                FunctionParameter(
                    name="contact",
                    type="string",
                    description="联系人姓名或电话号码"
                )
            ]
        ))
        
        # 天气查询
        self.register(FunctionDefinition(
            name="get_weather",
            description="查询天气信息",
            parameters=[
                FunctionParameter(
                    name="location",
                    type="string",
                    description="查询地点（默认当前位置）"
                ),
                FunctionParameter(
                    name="type",
                    type="string",
                    description="查询类型",
                    enum=["current", "forecast", "hourly"]
                )
            ]
        ))
    
    def register(self, func_def: FunctionDefinition):
        """注册函数"""
        self.functions[func_def.name] = func_def
    
    def get(self, name: str) -> Optional[FunctionDefinition]:
        """获取函数定义"""
        return self.functions.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """获取所有函数的schema"""
        return [f.to_schema() for f in self.functions.values()]
    
    def to_json_schema(self) -> str:
        """生成完整的JSON schema字符串"""
        return json.dumps({
            "functions": self.get_all_schemas()
        }, ensure_ascii=False, indent=2)
    
    def get_system_prompt_functions(self) -> str:
        """生成用于系统提示词的函数说明"""
        lines = ["可用的函数："]
        for func in self.functions.values():
            params = []
            for p in func.parameters:
                param_str = f"{p.name}: {p.type}"
                if p.enum:
                    param_str += f" ({'/'.join(p.enum)})"
                if p.required:
                    param_str += " [必需]"
                params.append(param_str)
            
            lines.append(f"- {func.name}: {func.description}")
            if params:
                lines.append(f"  参数: {', '.join(params)}")
        
        return "\n".join(lines)


# 创建默认注册表实例
default_registry = FunctionRegistry()


def get_function_schema() -> str:
    """获取默认函数schema"""
    return default_registry.to_json_schema()


def get_function_prompt() -> str:
    """获取函数说明提示词"""
    return default_registry.get_system_prompt_functions()
