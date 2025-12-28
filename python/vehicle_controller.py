"""
Vehicle Controller - 车辆控制器

模拟车辆各项功能的控制接口
实际应用中可替换为CAN总线通信或车辆API调用
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
import random


@dataclass
class AirConditionerState:
    """空调状态"""
    is_on: bool = False
    temperature: float = 24.0
    fan_speed: int = 3
    mode: str = "auto"


@dataclass
class WindowState:
    """车窗状态"""
    front_left: str = "closed"      # closed, open, half_open
    front_right: str = "closed"
    rear_left: str = "closed"
    rear_right: str = "closed"
    
    def get_position(self, position: str) -> str:
        return getattr(self, position, "closed")
    
    def set_position(self, position: str, state: str):
        if position == "all":
            self.front_left = state
            self.front_right = state
            self.rear_left = state
            self.rear_right = state
        else:
            setattr(self, position, state)


@dataclass
class SeatState:
    """座椅状态"""
    heating: int = 0        # 0-3
    cooling: int = 0        # 0-3
    massage: int = 0        # 0-3
    memory_slot: int = 0    # 1-3, 0表示未设置


@dataclass
class LightState:
    """灯光状态"""
    headlight: str = "off"   # on, off, auto
    highbeam: bool = False
    fog: bool = False
    interior: int = 0        # 0-100 亮度
    hazard: bool = False


@dataclass
class VehicleState:
    """车辆完整状态"""
    # 空调
    ac: AirConditionerState = field(default_factory=AirConditionerState)
    
    # 车窗
    windows: WindowState = field(default_factory=WindowState)
    
    # 座椅
    driver_seat: SeatState = field(default_factory=SeatState)
    passenger_seat: SeatState = field(default_factory=SeatState)
    
    # 灯光
    lights: LightState = field(default_factory=LightState)
    
    # 导航
    current_location: str = "未知位置"
    destination: Optional[str] = None
    navigation_active: bool = False
    
    # 媒体
    music_playing: bool = False
    current_track: str = ""
    volume: int = 50
    
    # 通话
    call_active: bool = False
    current_contact: str = ""
    
    # 车辆信息
    battery_percentage: int = 78
    estimated_range: int = 320     # km
    tire_pressure: Dict[str, float] = field(default_factory=lambda: {
        "front_left": 2.4,
        "front_right": 2.4,
        "rear_left": 2.3,
        "rear_right": 2.3
    })
    oil_life: int = 85             # %
    total_mileage: int = 15680     # km
    interior_temperature: float = 25.0
    exterior_temperature: float = 28.0


class VehicleController:
    """
    车辆控制器
    
    处理所有车辆控制相关的函数调用
    """
    
    def __init__(self):
        self.state = VehicleState()
        self._handlers: Dict[str, Callable[..., Awaitable[str]]] = {
            "control_air_conditioner": self._handle_ac,
            "control_window": self._handle_window,
            "navigate_to": self._handle_navigation,
            "play_music": self._handle_music,
            "get_vehicle_status": self._handle_status_query,
            "control_lights": self._handle_lights,
            "control_seat": self._handle_seat,
            "make_phone_call": self._handle_phone,
            "get_weather": self._handle_weather,
        }
        
        # 事件回调（用于UI更新等）
        self.on_state_changed: Optional[Callable[[str, Any], None]] = None
    
    async def execute(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """
        执行函数调用
        
        Args:
            function_name: 函数名称
            arguments: 函数参数（字典）
            
        Returns:
            执行结果的文本描述
        """
        handler = self._handlers.get(function_name)
        if handler:
            try:
                result = await handler(arguments)
                self._notify_state_changed(function_name, arguments)
                return result
            except Exception as e:
                return f"执行失败: {str(e)}"
        return f"未知的控制指令: {function_name}"
    
    def _notify_state_changed(self, function_name: str, arguments: Dict[str, Any]):
        """通知状态变更"""
        if self.on_state_changed:
            self.on_state_changed(function_name, arguments)
    
    # =========================================================================
    # 处理函数
    # =========================================================================
    
    async def _handle_ac(self, args: Dict[str, Any]) -> str:
        """处理空调控制"""
        action = args.get("action", "")
        
        if action == "on":
            self.state.ac.is_on = True
            temp = args.get("temperature", 24)
            fan = args.get("fan_speed", 3)
            mode = args.get("mode", "auto")
            
            self.state.ac.temperature = temp
            self.state.ac.fan_speed = fan
            self.state.ac.mode = mode
            
            return f"已打开空调，温度设置为{temp}°C，风量{fan}档，{mode}模式"
            
        elif action == "off":
            self.state.ac.is_on = False
            return "已关闭空调"
            
        elif action == "adjust":
            changes = []
            
            if "temperature" in args:
                temp = args["temperature"]
                self.state.ac.temperature = temp
                changes.append(f"温度{temp}°C")
            
            if "fan_speed" in args:
                fan = args["fan_speed"]
                self.state.ac.fan_speed = fan
                changes.append(f"风量{fan}档")
            
            if "mode" in args:
                mode = args["mode"]
                self.state.ac.mode = mode
                changes.append(f"{mode}模式")
            
            if changes:
                return f"已调整空调: {', '.join(changes)}"
            return "请指定需要调整的参数"
        
        return "未知的空调操作"
    
    async def _handle_window(self, args: Dict[str, Any]) -> str:
        """处理车窗控制"""
        position = args.get("position", "")
        action = args.get("action", "")
        
        position_names = {
            "front_left": "左前",
            "front_right": "右前",
            "rear_left": "左后",
            "rear_right": "右后",
            "all": "全部"
        }
        
        action_names = {
            "open": "打开",
            "close": "关闭",
            "half_open": "半开"
        }
        
        self.state.windows.set_position(position, action)
        
        pos_name = position_names.get(position, position)
        act_name = action_names.get(action, action)
        
        return f"已{act_name}{pos_name}车窗"
    
    async def _handle_navigation(self, args: Dict[str, Any]) -> str:
        """处理导航"""
        destination = args.get("destination", "")
        via_points = args.get("via_points", [])
        route_pref = args.get("route_preference", "fastest")
        
        if not destination:
            return "请指定目的地"
        
        self.state.destination = destination
        self.state.navigation_active = True
        
        # 模拟计算路线
        await asyncio.sleep(0.5)
        
        # 生成模拟信息
        distance = random.randint(5, 50)
        time_mins = distance * random.randint(2, 4)
        
        result = f"正在为您导航至{destination}"
        if via_points:
            result += f"，途经{', '.join(via_points)}"
        result += f"\n预计距离{distance}公里，约需{time_mins}分钟"
        
        return result
    
    async def _handle_music(self, args: Dict[str, Any]) -> str:
        """处理音乐控制"""
        action = args.get("action", "play")
        query = args.get("query", "")
        volume = args.get("volume")
        
        if volume is not None:
            self.state.volume = volume
            if action == "volume":
                return f"音量已调整至{volume}"
        
        if action == "play":
            if query:
                self.state.music_playing = True
                self.state.current_track = query
                return f"正在播放: {query}"
            elif self.state.current_track:
                self.state.music_playing = True
                return "继续播放"
            else:
                return "请告诉我您想听什么"
                
        elif action == "pause":
            self.state.music_playing = False
            return "音乐已暂停"
            
        elif action == "stop":
            self.state.music_playing = False
            self.state.current_track = ""
            return "已停止播放"
            
        elif action == "next":
            # 模拟下一首
            self.state.current_track = "下一首歌曲"
            return "已切换到下一首"
            
        elif action == "previous":
            self.state.current_track = "上一首歌曲"
            return "已切换到上一首"
            
        elif action == "shuffle":
            return "已开启随机播放"
            
        elif action == "repeat":
            return "已开启单曲循环"
        
        return "已执行音乐操作"
    
    async def _handle_status_query(self, args: Dict[str, Any]) -> str:
        """处理状态查询"""
        info_type = args.get("info_type", "all")
        
        status_info = {
            "battery": f"电池电量: {self.state.battery_percentage}%，剩余续航约{self.state.estimated_range}公里",
            "tire_pressure": self._format_tire_pressure(),
            "oil": f"机油寿命: {self.state.oil_life}%，状态良好",
            "mileage": f"总里程: {self.state.total_mileage:,}公里",
            "temperature": f"车内温度: {self.state.interior_temperature}°C，车外温度: {self.state.exterior_temperature}°C",
            "doors": "所有车门已锁定",
            "lights": self._format_lights_status(),
        }
        
        if info_type == "all":
            return "\n".join([
                "📊 车辆状态报告",
                f"🔋 {status_info['battery']}",
                f"🛞 {status_info['tire_pressure']}",
                f"🛢️ {status_info['oil']}",
                f"📍 {status_info['mileage']}",
                f"🌡️ {status_info['temperature']}",
            ])
        
        return status_info.get(info_type, f"未知的查询类型: {info_type}")
    
    def _format_tire_pressure(self) -> str:
        """格式化胎压信息"""
        tp = self.state.tire_pressure
        return (f"胎压正常 - 左前:{tp['front_left']}bar 右前:{tp['front_right']}bar "
                f"左后:{tp['rear_left']}bar 右后:{tp['rear_right']}bar")
    
    def _format_lights_status(self) -> str:
        """格式化灯光状态"""
        lights = self.state.lights
        status = []
        if lights.headlight != "off":
            status.append(f"大灯{lights.headlight}")
        if lights.highbeam:
            status.append("远光灯开启")
        if lights.fog:
            status.append("雾灯开启")
        if lights.interior > 0:
            status.append(f"内饰灯{lights.interior}%")
        
        return "灯光: " + (", ".join(status) if status else "全部关闭")
    
    async def _handle_lights(self, args: Dict[str, Any]) -> str:
        """处理灯光控制"""
        light_type = args.get("light_type", "")
        action = args.get("action", "")
        brightness = args.get("brightness")
        
        light_names = {
            "headlight": "大灯",
            "highbeam": "远光灯",
            "fog": "雾灯",
            "interior": "内饰灯",
            "hazard": "双闪",
            "turn_left": "左转向灯",
            "turn_right": "右转向灯"
        }
        
        if light_type == "headlight":
            self.state.lights.headlight = action
        elif light_type == "highbeam":
            self.state.lights.highbeam = (action == "on")
        elif light_type == "fog":
            self.state.lights.fog = (action == "on")
        elif light_type == "interior":
            if brightness is not None:
                self.state.lights.interior = brightness
            else:
                self.state.lights.interior = 100 if action == "on" else 0
        elif light_type == "hazard":
            self.state.lights.hazard = (action == "on")
        
        name = light_names.get(light_type, light_type)
        action_text = "打开" if action == "on" else ("关闭" if action == "off" else "自动")
        
        return f"已{action_text}{name}"
    
    async def _handle_seat(self, args: Dict[str, Any]) -> str:
        """处理座椅控制"""
        seat = args.get("seat", "driver")
        function = args.get("function", "")
        level = args.get("level", 0)
        memory_slot = args.get("memory_slot")
        
        seat_state = self.state.driver_seat if seat == "driver" else self.state.passenger_seat
        seat_name = "主驾" if seat == "driver" else "副驾"
        
        if function == "heating":
            seat_state.heating = level
            return f"{seat_name}座椅加热已设置为{level}档" if level > 0 else f"已关闭{seat_name}座椅加热"
            
        elif function == "cooling":
            seat_state.cooling = level
            return f"{seat_name}座椅通风已设置为{level}档" if level > 0 else f"已关闭{seat_name}座椅通风"
            
        elif function == "massage":
            seat_state.massage = level
            return f"{seat_name}座椅按摩已设置为{level}档" if level > 0 else f"已关闭{seat_name}座椅按摩"
            
        elif function == "memory":
            if memory_slot:
                seat_state.memory_slot = memory_slot
                return f"已恢复{seat_name}座椅记忆位置{memory_slot}"
        
        return f"已调整{seat_name}座椅"
    
    async def _handle_phone(self, args: Dict[str, Any]) -> str:
        """处理电话控制"""
        action = args.get("action", "")
        contact = args.get("contact", "")
        
        if action == "call":
            if not contact:
                return "请告诉我您要拨打给谁"
            self.state.call_active = True
            self.state.current_contact = contact
            return f"正在拨打{contact}..."
            
        elif action == "answer":
            self.state.call_active = True
            return "已接听来电"
            
        elif action == "hangup":
            self.state.call_active = False
            self.state.current_contact = ""
            return "已挂断电话"
            
        elif action == "reject":
            return "已拒绝来电"
            
        elif action == "mute":
            return "已静音"
        
        return "已执行电话操作"
    
    async def _handle_weather(self, args: Dict[str, Any]) -> str:
        """处理天气查询"""
        location = args.get("location", "当前位置")
        query_type = args.get("type", "current")
        
        # 模拟天气数据
        weather_data = {
            "condition": random.choice(["晴", "多云", "阴", "小雨"]),
            "temperature": random.randint(15, 35),
            "humidity": random.randint(40, 80),
            "wind": random.choice(["微风", "东风3级", "西北风4级"]),
        }
        
        if query_type == "current":
            return (f"{location}当前天气: {weather_data['condition']}，"
                   f"温度{weather_data['temperature']}°C，"
                   f"湿度{weather_data['humidity']}%，{weather_data['wind']}")
                   
        elif query_type == "forecast":
            return f"{location}未来三天: 明天多云转晴，后天晴，大后天多云"
        
        return f"{location}天气信息"
    
    # =========================================================================
    # 状态访问
    # =========================================================================
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "ac": {
                "on": self.state.ac.is_on,
                "temperature": self.state.ac.temperature,
                "fan_speed": self.state.ac.fan_speed
            },
            "windows": {
                "front_left": self.state.windows.front_left,
                "front_right": self.state.windows.front_right,
                "rear_left": self.state.windows.rear_left,
                "rear_right": self.state.windows.rear_right
            },
            "navigation": {
                "active": self.state.navigation_active,
                "destination": self.state.destination
            },
            "music": {
                "playing": self.state.music_playing,
                "track": self.state.current_track,
                "volume": self.state.volume
            },
            "battery": self.state.battery_percentage,
            "range": self.state.estimated_range
        }


# 测试代码
if __name__ == "__main__":
    async def test():
        controller = VehicleController()
        
        # 测试空调
        print(await controller.execute("control_air_conditioner", 
                                       {"action": "on", "temperature": 26}))
        
        # 测试车窗
        print(await controller.execute("control_window", 
                                       {"position": "front_left", "action": "half_open"}))
        
        # 测试导航
        print(await controller.execute("navigate_to", 
                                       {"destination": "北京天安门"}))
        
        # 测试状态查询
        print(await controller.execute("get_vehicle_status", 
                                       {"info_type": "all"}))
    
    asyncio.run(test())
