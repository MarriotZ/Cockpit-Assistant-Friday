#!/usr/bin/env python3
"""
Demo Text - 文本交互演示

命令行文本交互演示程序
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from cockpit_assistant import CockpitAssistant
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

console = Console()


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🚗  智能座舱多轮对话系统  🚗                          ║
║            Cockpit Assistant Demo                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_help():
    """打印帮助信息"""
    help_text = """
## 可用命令

| 命令 | 说明 |
|------|------|
| `quit` / `exit` | 退出程序 |
| `clear` / `reset` | 清除对话历史 |
| `status` | 查看车辆状态 |
| `stats` | 查看引擎统计 |
| `help` | 显示帮助 |

## 示例对话

- "把空调打开，温度调到26度"
- "导航到北京天安门"
- "播放周杰伦的歌"
- "查一下车还有多少电"
- "把车窗打开一半"
- "打开座椅加热"
    """
    console.print(Markdown(help_text))


async def main(model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 35):
    """主函数"""
    print_banner()
    
    console.print(f"\n[dim]模型路径: {model_path}[/dim]")
    console.print(f"[dim]上下文长度: {n_ctx}[/dim]")
    console.print(f"[dim]GPU层数: {n_gpu_layers}[/dim]\n")
    
    # 初始化助手
    with console.status("[bold green]正在加载模型...", spinner="dots"):
        try:
            assistant = CockpitAssistant(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
        except Exception as e:
            console.print(f"[red]加载模型失败: {e}[/red]")
            return
    
    console.print("[green]✓ 模型加载成功！[/green]\n")
    console.print("输入 [bold]help[/bold] 查看帮助，输入 [bold]quit[/bold] 退出\n")
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("\n[yellow]再见！祝您行车安全！🚗[/yellow]\n")
                break
            
            if user_input.lower() in ["clear", "reset"]:
                assistant.reset_conversation()
                console.print("[green]✓ 对话已重置[/green]\n")
                continue
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            if user_input.lower() == "status":
                state = assistant.get_vehicle_state()
                console.print(Panel(
                    str(state),
                    title="车辆状态",
                    border_style="cyan"
                ))
                continue
            
            if user_input.lower() == "stats":
                stats = assistant.get_stats()
                console.print(Panel(
                    str(stats),
                    title="引擎统计",
                    border_style="green"
                ))
                continue
            
            # 获取响应
            console.print("[bold green]Assistant:[/bold green] ", end="")
            
            response_text = ""
            async for token in assistant.chat(user_input):
                console.print(token, end="", highlight=False)
                response_text += token
            
            console.print("\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]已中断[/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]\n")


def run():
    """运行入口"""
    parser = argparse.ArgumentParser(
        description="智能座舱对话系统 - 文本演示"
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
        help="GPU层数 (默认: 35, -1表示全部)"
    )
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        console.print(f"[yellow]警告: 模型文件不存在: {args.model_path}[/yellow]")
        console.print("[yellow]将使用模拟引擎运行演示[/yellow]\n")
    
    asyncio.run(main(args.model_path, args.ctx, args.gpu_layers))


if __name__ == "__main__":
    run()
