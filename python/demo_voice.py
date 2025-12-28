#!/usr/bin/env python3
"""
Demo Voice - 语音交互演示

支持语音输入和语音输出的完整交互演示
"""

import asyncio
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

console = Console()


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🎤  智能座舱语音交互系统  🔊                          ║
║            Voice Cockpit Assistant                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


async def main(model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 35,
               asr_model: str = "small", tts_voice: str = "xiaoxiao"):
    """主函数"""
    print_banner()
    
    # 检查依赖
    try:
        from voice_interface import (
            CockpitVoiceAssistant, VoiceInterface,
            ASRConfig, TTSConfig, HAS_WHISPER, HAS_EDGE_TTS, HAS_SOUNDDEVICE
        )
    except ImportError as e:
        console.print(f"[red]导入错误: {e}[/red]")
        console.print("[yellow]请确保已安装所有依赖[/yellow]")
        return
    
    # 检查可用功能
    console.print("\n[bold]系统检查:[/bold]")
    console.print(f"  ASR (Whisper): {'✅ 可用' if HAS_WHISPER else '❌ 不可用'}")
    console.print(f"  TTS (Edge TTS): {'✅ 可用' if HAS_EDGE_TTS else '❌ 不可用'}")
    console.print(f"  音频设备: {'✅ 可用' if HAS_SOUNDDEVICE else '❌ 不可用'}")
    
    if not HAS_SOUNDDEVICE:
        console.print("\n[yellow]警告: 音频设备不可用，将使用文本模式[/yellow]")
        console.print("[yellow]请安装: pip install sounddevice[/yellow]\n")
        
        # 回退到文本模式
        from demo_text import main as text_main
        await text_main(model_path, n_ctx, n_gpu_layers)
        return
    
    # 配置
    asr_config = ASRConfig(
        model_size=asr_model,
        device="cuda" if n_gpu_layers > 0 else "cpu",
        language="zh"
    )
    
    tts_config = TTSConfig()
    
    # TTS语音映射
    voice_map = {
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",
        "xiaoyi": "zh-CN-XiaoyiNeural",
        "yunjian": "zh-CN-YunjianNeural",
        "yunxi": "zh-CN-YunxiNeural",
    }
    tts_config.voice = voice_map.get(tts_voice, tts_voice)
    
    console.print(f"\n[dim]模型路径: {model_path}[/dim]")
    console.print(f"[dim]ASR模型: {asr_model}[/dim]")
    console.print(f"[dim]TTS语音: {tts_voice}[/dim]\n")
    
    # 初始化
    with console.status("[bold green]正在加载模型...", spinner="dots"):
        try:
            assistant = CockpitVoiceAssistant(
                model_path=model_path,
                asr_config=asr_config,
                tts_config=tts_config
            )
            
            # 预热ASR模型
            if HAS_WHISPER:
                assistant.voice.asr.initialize()
                
        except Exception as e:
            console.print(f"[red]初始化失败: {e}[/red]")
            return
    
    console.print("[green]✓ 系统初始化完成！[/green]\n")
    
    # 使用说明
    console.print(Panel(
        "[bold]使用说明:[/bold]\n\n"
        "• 按 [bold]Enter[/bold] 开始说话\n"
        "• 说完后等待自动识别\n"
        "• 输入 [bold]quit[/bold] 退出\n"
        "• 输入 [bold]text[/bold] 切换到文本模式",
        title="语音模式",
        border_style="cyan"
    ))
    
    text_mode = False
    
    # 主循环
    while True:
        try:
            if text_mode:
                # 文本输入模式
                user_input = console.input("\n[bold blue]You (文本):[/bold blue] ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                
                if user_input.lower() == "voice":
                    text_mode = False
                    console.print("[green]已切换到语音模式[/green]")
                    continue
                
                # 处理文本输入
                console.print("[bold green]Assistant:[/bold green] ", end="")
                
                response_text = ""
                async for token in assistant.assistant.chat(user_input):
                    console.print(token, end="", highlight=False)
                    response_text += token
                
                console.print()
                
                # TTS播报
                if HAS_EDGE_TTS:
                    clean_text = _clean_for_tts(response_text)
                    if clean_text:
                        with console.status("[dim]正在播报...[/dim]"):
                            await assistant.voice.speak(clean_text)
                
            else:
                # 语音输入模式
                console.print("\n[bold cyan]按 Enter 开始说话...[/bold cyan]", end="")
                cmd = console.input("")
                
                if cmd.lower() in ["quit", "exit", "q"]:
                    break
                
                if cmd.lower() == "text":
                    text_mode = True
                    console.print("[green]已切换到文本模式[/green]")
                    continue
                
                # 开始录音
                console.print("[yellow]🎤 正在听...[/yellow]")
                
                try:
                    # 录制并识别
                    user_text = await assistant.voice.listen(duration=5.0)
                    
                    if not user_text:
                        console.print("[dim]未检测到语音[/dim]")
                        continue
                    
                    console.print(f"[bold blue]You:[/bold blue] {user_text}")
                    
                    # 获取响应
                    console.print("[bold green]Assistant:[/bold green] ", end="")
                    
                    response_text = ""
                    async for token in assistant.assistant.chat(user_text):
                        console.print(token, end="", highlight=False)
                        response_text += token
                    
                    console.print()
                    
                    # TTS播报
                    if HAS_EDGE_TTS:
                        clean_text = _clean_for_tts(response_text)
                        if clean_text:
                            with console.status("[dim]正在播报...[/dim]"):
                                await assistant.voice.speak(clean_text)
                    
                except Exception as e:
                    console.print(f"[red]错误: {e}[/red]")
                    
        except KeyboardInterrupt:
            console.print("\n\n[yellow]已中断[/yellow]")
            break
    
    console.print("\n[yellow]再见！祝您行车安全！🚗[/yellow]\n")


def _clean_for_tts(text: str) -> str:
    """清理文本用于TTS"""
    import re
    # 移除JSON
    cleaned = re.sub(r'\{[^}]+\}', '', text)
    # 移除特殊符号
    cleaned = re.sub(r'[✅❌🔧📊🔋🛞🛢️📍🌡️]', '', cleaned)
    # 移除多余空白
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


def run():
    """运行入口"""
    parser = argparse.ArgumentParser(
        description="智能座舱助手 - 语音交互演示"
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/qwen2.5-7b-instruct-q4_k_m.gguf",
        help="模型文件路径"
    )
    parser.add_argument(
        "-c", "--ctx",
        type=int,
        default=4096,
        help="上下文长度 (默认: 4096)"
    )
    parser.add_argument(
        "-g", "--gpu-layers",
        type=int,
        default=35,
        help="GPU层数 (默认: 35)"
    )
    parser.add_argument(
        "--asr-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper模型大小 (默认: small)"
    )
    parser.add_argument(
        "--tts-voice",
        default="xiaoxiao",
        choices=["xiaoxiao", "xiaoyi", "yunjian", "yunxi"],
        help="TTS语音 (默认: xiaoxiao)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(
        args.model_path,
        args.ctx,
        args.gpu_layers,
        args.asr_model,
        args.tts_voice
    ))


if __name__ == "__main__":
    run()
