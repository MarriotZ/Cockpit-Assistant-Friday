"""
Vehicle Controller

Simulates vehicle control interfaces.
Replace with CAN bus communication or vehicle API calls in production.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
import random


@dataclass
class AirConditionerState:
    """Air conditioner state"""
    is_on: bool = False
    temperature: float = 24.0
    fan_speed: int = 3
    mode: str = "auto"


@dataclass
class WindowState:
    """Window state"""
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
    """Seat state"""
    heating: int = 0        # 0-3
    cooling: int = 0        # 0-3
    massage: int = 0        # 0-3
    memory_slot: int = 0    # 1-3, 0 means not set


@dataclass
class LightState:
    """Light state"""
    headlight: str = "off"   # on, off, auto
    highbeam: bool = False
    fog: bool = False
    interior: int = 0        # 0-100 brightness
    hazard: bool = False


@dataclass
class VehicleState:
    """Complete vehicle state"""
    ac: AirConditionerState = field(default_factory=AirConditionerState)
    windows: WindowState = field(default_factory=WindowState)
    driver_seat: SeatState = field(default_factory=SeatState)
    passenger_seat: SeatState = field(default_factory=SeatState)
    lights: LightState = field(default_factory=LightState)
    
    # Navigation
    current_location: str = "Unknown"
    destination: Optional[str] = None
    navigation_active: bool = False
    
    # Media
    music_playing: bool = False
    current_track: str = ""
    volume: int = 50
    
    # Phone
    call_active: bool = False
    current_contact: str = ""
    
    # Vehicle info
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
    """Vehicle controller that handles all vehicle control function calls"""
    
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
        self.on_state_changed: Optional[Callable[[str, Any], None]] = None
    
    async def execute(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute function call
        
        Args:
            function_name: Function name
            arguments: Function arguments (dict)
            
        Returns:
            Text description of execution result
        """
        handler = self._handlers.get(function_name)
        if handler:
            try:
                result = await handler(arguments)
                self._notify_state_changed(function_name, arguments)
                return result
            except Exception as e:
                return f"Execution failed: {str(e)}"
        return f"Unknown control command: {function_name}"
    
    def _notify_state_changed(self, function_name: str, arguments: Dict[str, Any]):
        """Notify state change"""
        if self.on_state_changed:
            self.on_state_changed(function_name, arguments)
    
    # =========================================================================
    # Handler Functions
    # =========================================================================
    
    async def _handle_ac(self, args: Dict[str, Any]) -> str:
        """Handle air conditioner control"""
        action = args.get("action", "")
        
        if action == "on":
            self.state.ac.is_on = True
            temp = args.get("temperature", 24)
            fan = args.get("fan_speed", 3)
            mode = args.get("mode", "auto")
            
            self.state.ac.temperature = temp
            self.state.ac.fan_speed = fan
            self.state.ac.mode = mode
            
            return f"AC turned on, temperature set to {temp}°C, fan level {fan}, {mode} mode"
            
        elif action == "off":
            self.state.ac.is_on = False
            return "AC turned off"
            
        elif action == "adjust":
            changes = []
            
            if "temperature" in args:
                temp = args["temperature"]
                self.state.ac.temperature = temp
                changes.append(f"temperature {temp}°C")
            
            if "fan_speed" in args:
                fan = args["fan_speed"]
                self.state.ac.fan_speed = fan
                changes.append(f"fan level {fan}")
            
            if "mode" in args:
                mode = args["mode"]
                self.state.ac.mode = mode
                changes.append(f"{mode} mode")
            
            if changes:
                return f"AC adjusted: {', '.join(changes)}"
            return "Please specify parameters to adjust"
        
        return "Unknown AC operation"
    
    async def _handle_window(self, args: Dict[str, Any]) -> str:
        """Handle window control"""
        position = args.get("position", "")
        action = args.get("action", "")
        
        position_names = {
            "front_left": "front left",
            "front_right": "front right",
            "rear_left": "rear left",
            "rear_right": "rear right",
            "all": "all"
        }
        
        action_names = {
            "open": "opened",
            "close": "closed",
            "half_open": "half opened"
        }
        
        self.state.windows.set_position(position, action)
        
        pos_name = position_names.get(position, position)
        act_name = action_names.get(action, action)
        
        return f"{pos_name.capitalize()} window {act_name}"
    
    async def _handle_navigation(self, args: Dict[str, Any]) -> str:
        """Handle navigation"""
        destination = args.get("destination", "")
        via_points = args.get("via_points", [])
        route_pref = args.get("route_preference", "fastest")
        
        if not destination:
            return "Please specify a destination"
        
        self.state.destination = destination
        self.state.navigation_active = True
        
        await asyncio.sleep(0.5)  # Simulate route calculation
        
        distance = random.randint(5, 50)
        time_mins = distance * random.randint(2, 4)
        
        result = f"Navigating to {destination}"
        if via_points:
            result += f", via {', '.join(via_points)}"
        result += f"\nEstimated distance {distance}km, about {time_mins} minutes"
        
        return result
    
    async def _handle_music(self, args: Dict[str, Any]) -> str:
        """Handle music control"""
        action = args.get("action", "play")
        query = args.get("query", "")
        volume = args.get("volume")
        
        if volume is not None:
            self.state.volume = volume
            if action == "volume":
                return f"Volume set to {volume}"
        
        if action == "play":
            if query:
                self.state.music_playing = True
                self.state.current_track = query
                return f"Now playing: {query}"
            elif self.state.current_track:
                self.state.music_playing = True
                return "Resuming playback"
            else:
                return "What would you like to listen to?"
                
        elif action == "pause":
            self.state.music_playing = False
            return "Music paused"
            
        elif action == "stop":
            self.state.music_playing = False
            self.state.current_track = ""
            return "Playback stopped"
            
        elif action == "next":
            self.state.current_track = "Next track"
            return "Skipped to next track"
            
        elif action == "previous":
            self.state.current_track = "Previous track"
            return "Skipped to previous track"
            
        elif action == "shuffle":
            return "Shuffle enabled"
            
        elif action == "repeat":
            return "Repeat enabled"
        
        return "Music operation executed"
    
    async def _handle_status_query(self, args: Dict[str, Any]) -> str:
        """Handle status query"""
        info_type = args.get("info_type", "all")
        
        status_info = {
            "battery": f"Battery: {self.state.battery_percentage}%, estimated range {self.state.estimated_range}km",
            "tire_pressure": self._format_tire_pressure(),
            "oil": f"Oil life: {self.state.oil_life}%, good condition",
            "mileage": f"Total mileage: {self.state.total_mileage:,}km",
            "temperature": f"Interior: {self.state.interior_temperature}°C, Exterior: {self.state.exterior_temperature}°C",
            "doors": "All doors locked",
            "lights": self._format_lights_status(),
        }
        
        if info_type == "all":
            return "\n".join([
                "📊 Vehicle Status Report",
                f"🔋 {status_info['battery']}",
                f"🛞 {status_info['tire_pressure']}",
                f"🛢️ {status_info['oil']}",
                f"📍 {status_info['mileage']}",
                f"🌡️ {status_info['temperature']}",
            ])
        
        return status_info.get(info_type, f"Unknown query type: {info_type}")
    
    def _format_tire_pressure(self) -> str:
        """Format tire pressure info"""
        tp = self.state.tire_pressure
        return (f"Tire pressure normal - FL:{tp['front_left']}bar FR:{tp['front_right']}bar "
                f"RL:{tp['rear_left']}bar RR:{tp['rear_right']}bar")
    
    def _format_lights_status(self) -> str:
        """Format lights status"""
        lights = self.state.lights
        status = []
        if lights.headlight != "off":
            status.append(f"headlight {lights.headlight}")
        if lights.highbeam:
            status.append("high beam on")
        if lights.fog:
            status.append("fog light on")
        if lights.interior > 0:
            status.append(f"interior {lights.interior}%")
        
        return "Lights: " + (", ".join(status) if status else "all off")
    
    async def _handle_lights(self, args: Dict[str, Any]) -> str:
        """Handle lights control"""
        light_type = args.get("light_type", "")
        action = args.get("action", "")
        brightness = args.get("brightness")
        
        light_names = {
            "headlight": "headlight",
            "highbeam": "high beam",
            "fog": "fog light",
            "interior": "interior light",
            "hazard": "hazard light",
            "turn_left": "left turn signal",
            "turn_right": "right turn signal"
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
        action_text = "on" if action == "on" else ("off" if action == "off" else "auto")
        
        return f"{name.capitalize()} turned {action_text}"
    
    async def _handle_seat(self, args: Dict[str, Any]) -> str:
        """Handle seat control"""
        seat = args.get("seat", "driver")
        function = args.get("function", "")
        level = args.get("level", 0)
        memory_slot = args.get("memory_slot")
        
        seat_state = self.state.driver_seat if seat == "driver" else self.state.passenger_seat
        seat_name = "Driver" if seat == "driver" else "Passenger"
        
        if function == "heating":
            seat_state.heating = level
            return f"{seat_name} seat heating set to level {level}" if level > 0 else f"{seat_name} seat heating turned off"
            
        elif function == "cooling":
            seat_state.cooling = level
            return f"{seat_name} seat ventilation set to level {level}" if level > 0 else f"{seat_name} seat ventilation turned off"
            
        elif function == "massage":
            seat_state.massage = level
            return f"{seat_name} seat massage set to level {level}" if level > 0 else f"{seat_name} seat massage turned off"
            
        elif function == "memory":
            if memory_slot:
                seat_state.memory_slot = memory_slot
                return f"{seat_name} seat restored to memory position {memory_slot}"
        
        return f"{seat_name} seat adjusted"
    
    async def _handle_phone(self, args: Dict[str, Any]) -> str:
        """Handle phone control"""
        action = args.get("action", "")
        contact = args.get("contact", "")
        
        if action == "call":
            if not contact:
                return "Who would you like to call?"
            self.state.call_active = True
            self.state.current_contact = contact
            return f"Calling {contact}..."
            
        elif action == "answer":
            self.state.call_active = True
            return "Call answered"
            
        elif action == "hangup":
            self.state.call_active = False
            self.state.current_contact = ""
            return "Call ended"
            
        elif action == "reject":
            return "Call rejected"
            
        elif action == "mute":
            return "Call muted"
        
        return "Phone operation executed"
    
    async def _handle_weather(self, args: Dict[str, Any]) -> str:
        """Handle weather query"""
        location = args.get("location", "current location")
        query_type = args.get("type", "current")
        
        weather_data = {
            "condition": random.choice(["Sunny", "Cloudy", "Overcast", "Light rain"]),
            "temperature": random.randint(15, 35),
            "humidity": random.randint(40, 80),
            "wind": random.choice(["Light breeze", "East 3", "Northwest 4"]),
        }
        
        if query_type == "current":
            return (f"Weather at {location}: {weather_data['condition']}, "
                   f"{weather_data['temperature']}°C, "
                   f"humidity {weather_data['humidity']}%, {weather_data['wind']}")
                   
        elif query_type == "forecast":
            return f"Forecast for {location}: Tomorrow cloudy to sunny, day after sunny, then cloudy"
        
        return f"Weather info for {location}"
    
    # =========================================================================
    # State Access
    # =========================================================================
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary"""
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


if __name__ == "__main__":
    async def test():
        controller = VehicleController()
        
        print(await controller.execute("control_air_conditioner", {"action": "on", "temperature": 26}))
        
        print(await controller.execute("control_window", {"position": "front_left", "action": "half_open"}))
        
        print(await controller.execute("navigate_to", {"destination": "Beijing Tiananmen"}))
        
        print(await controller.execute("get_vehicle_status", {"info_type": "all"}))
    
    asyncio.run(test())