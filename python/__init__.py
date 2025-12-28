"""
Cockpit Assistant - 智能座舱助手

基于大语言模型的智能汽车座舱助手系统
"""

__version__ = "1.0.0"
__author__ = "Cockpit Assistant Team"

from .cockpit_assistant import CockpitAssistant, AsyncCockpitAssistant
from .vehicle_controller import VehicleController
from .function_registry import FunctionRegistry, get_function_schema

__all__ = [
    "CockpitAssistant",
    "AsyncCockpitAssistant",
    "VehicleController",
    "FunctionRegistry",
    "get_function_schema",
]
