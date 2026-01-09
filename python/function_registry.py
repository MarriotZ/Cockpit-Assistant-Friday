"""
Function Registry

Defines all callable functions and their parameter schemas for the cockpit assistant.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class FunctionParameter:
    """Function parameter definition"""
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
    """Function definition"""
    name: str
    description: str
    parameters: List[FunctionParameter] = field(default_factory=list)
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format"""
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
    """Function registry"""
    
    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
        self._register_default_functions()
    
    def _register_default_functions(self):
        """Register default vehicle control functions"""
        
        # Air conditioner control
        self.register(FunctionDefinition(
            name="control_air_conditioner",
            description="Control vehicle AC: on/off, temperature, fan speed",
            parameters=[
                FunctionParameter(
                    name="action",
                    type="string",
                    description="Action type: on, off, adjust",
                    required=True,
                    enum=["on", "off", "adjust"]
                ),
                FunctionParameter(
                    name="temperature",
                    type="number",
                    description="Target temperature (Celsius)",
                    minimum=16,
                    maximum=30,
                    default=24
                ),
                FunctionParameter(
                    name="fan_speed",
                    type="integer",
                    description="Fan level (1-5)",
                    minimum=1,
                    maximum=5,
                    default=3
                ),
                FunctionParameter(
                    name="mode",
                    type="string",
                    description="AC mode",
                    enum=["auto", "cool", "heat", "fan", "dry"]
                )
            ]
        ))
        
        # Window control
        self.register(FunctionDefinition(
            name="control_window",
            description="Control vehicle windows",
            parameters=[
                FunctionParameter(
                    name="position",
                    type="string",
                    description="Window position: front_left, front_right, rear_left, rear_right, all",
                    required=True,
                    enum=["front_left", "front_right", "rear_left", "rear_right", "all"]
                ),
                FunctionParameter(
                    name="action",
                    type="string",
                    description="Action type: open, close, half_open",
                    required=True,
                    enum=["open", "close", "half_open"]
                ),
                FunctionParameter(
                    name="percentage",
                    type="integer",
                    description="Open percentage (0-100)",
                    minimum=0,
                    maximum=100
                )
            ]
        ))
        
        # Navigation
        self.register(FunctionDefinition(
            name="navigate_to",
            description="Set navigation destination by name or address",
            parameters=[
                FunctionParameter(
                    name="destination",
                    type="string",
                    description="Destination name or address",
                    required=True
                ),
                FunctionParameter(
                    name="via_points",
                    type="array",
                    description="List of waypoints"
                ),
                FunctionParameter(
                    name="avoid",
                    type="array",
                    description="Conditions to avoid: toll, highway, ferry"
                ),
                FunctionParameter(
                    name="route_preference",
                    type="string",
                    description="Route preference",
                    enum=["fastest", "shortest", "economical"]
                )
            ]
        ))
        
        # Music control
        self.register(FunctionDefinition(
            name="play_music",
            description="Play music: search or control playback",
            parameters=[
                FunctionParameter(
                    name="query",
                    type="string",
                    description="Search keywords (song, artist, album)"
                ),
                FunctionParameter(
                    name="action",
                    type="string",
                    description="Playback action",
                    enum=["play", "pause", "stop", "next", "previous", "shuffle", "repeat"]
                ),
                FunctionParameter(
                    name="volume",
                    type="integer",
                    description="Volume (0-100)",
                    minimum=0,
                    maximum=100
                ),
                FunctionParameter(
                    name="source",
                    type="string",
                    description="Music source",
                    enum=["local", "bluetooth", "usb", "online"]
                )
            ]
        ))
        
        # Vehicle status query
        self.register(FunctionDefinition(
            name="get_vehicle_status",
            description="Query vehicle status information",
            parameters=[
                FunctionParameter(
                    name="info_type",
                    type="string",
                    description="Query type: battery, tire_pressure, oil, mileage, temperature, doors, lights, all",
                    required=True,
                    enum=["battery", "tire_pressure", "oil", "mileage", "temperature", "doors", "lights", "all"]
                )
            ]
        ))
        
        # Light control
        self.register(FunctionDefinition(
            name="control_lights",
            description="Control vehicle lights",
            parameters=[
                FunctionParameter(
                    name="light_type",
                    type="string",
                    description="Light type",
                    required=True,
                    enum=["headlight", "highbeam", "fog", "interior", "hazard", "turn_left", "turn_right"]
                ),
                FunctionParameter(
                    name="action",
                    type="string",
                    description="Action type",
                    required=True,
                    enum=["on", "off", "auto"]
                ),
                FunctionParameter(
                    name="brightness",
                    type="integer",
                    description="Brightness (interior light only)",
                    minimum=0,
                    maximum=100
                )
            ]
        ))
        
        # Seat control
        self.register(FunctionDefinition(
            name="control_seat",
            description="Control seat position and functions",
            parameters=[
                FunctionParameter(
                    name="seat",
                    type="string",
                    description="Seat position",
                    required=True,
                    enum=["driver", "passenger", "rear_left", "rear_right"]
                ),
                FunctionParameter(
                    name="function",
                    type="string",
                    description="Function type",
                    required=True,
                    enum=["heating", "cooling", "massage", "position", "memory"]
                ),
                FunctionParameter(
                    name="level",
                    type="integer",
                    description="Intensity level (0-3)",
                    minimum=0,
                    maximum=3
                ),
                FunctionParameter(
                    name="memory_slot",
                    type="integer",
                    description="Memory position (1-3)",
                    minimum=1,
                    maximum=3
                )
            ]
        ))
        
        # Phone control
        self.register(FunctionDefinition(
            name="make_phone_call",
            description="Make or manage phone calls",
            parameters=[
                FunctionParameter(
                    name="action",
                    type="string",
                    description="Action type",
                    required=True,
                    enum=["call", "answer", "hangup", "reject", "mute"]
                ),
                FunctionParameter(
                    name="contact",
                    type="string",
                    description="Contact name or phone number"
                )
            ]
        ))
        
        # Weather query
        self.register(FunctionDefinition(
            name="get_weather",
            description="Query weather information",
            parameters=[
                FunctionParameter(
                    name="location",
                    type="string",
                    description="Location (default: current location)"
                ),
                FunctionParameter(
                    name="type",
                    type="string",
                    description="Query type",
                    enum=["current", "forecast", "hourly"]
                )
            ]
        ))
    
    def register(self, func_def: FunctionDefinition):
        """Register a function"""
        self.functions[func_def.name] = func_def
    
    def get(self, name: str) -> Optional[FunctionDefinition]:
        """Get function definition by name"""
        return self.functions.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all functions"""
        return [f.to_schema() for f in self.functions.values()]
    
    def to_json_schema(self) -> str:
        """Generate complete JSON schema string"""
        return json.dumps({
            "functions": self.get_all_schemas()
        }, ensure_ascii=False, indent=2)
    
    def get_system_prompt_functions(self, detailed: bool = True) -> str:
        """
        Generate function descriptions for system prompt
        
        Args:
            detailed: Include detailed parameter info
                     True: Full format (name + description + parameters)
                     False: Compact format (name + brief description only)
        """
        lines = ["Available functions:"]
        for func in self.functions.values():
            if detailed:
                params = []
                for p in func.parameters:
                    param_str = f"{p.name}: {p.type}"
                    if p.enum:
                        param_str += f" ({'/'.join(p.enum)})"
                    if p.required:
                        param_str += " [required]"
                    params.append(param_str)
                
                lines.append(f"- {func.name}: {func.description}")
                if params:
                    lines.append(f"  Parameters: {', '.join(params)}")
            else:
                lines.append(f"- {func.name}: {func.description}")
        
        return "\n".join(lines)


default_registry = FunctionRegistry()


def get_function_schema() -> str:
    """Get JSON schema for all functions"""
    return default_registry.to_json_schema()


def get_function_prompt(detailed: bool = True) -> str:
    """
    Get function description prompt
    
    Args:
        detailed: Include detailed parameter info (default True for large models)
                  False: Compact format (suitable for smaller models like 3B)
    """
    return default_registry.get_system_prompt_functions(detailed=detailed)