#!/bin/bash
# ==============================================================================
# 快速启动脚本
# Quick start script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_banner() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║        🚗  智能座舱助手快速启动  🚗                          ║"
    echo "║            Cockpit Assistant Quick Start                     ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_menu() {
    echo ""
    echo -e "${BLUE}请选择启动模式:${NC}"
    echo ""
    echo "  1) 文本交互演示 (demo_text.py)"
    echo "  2) Web界面演示 (demo_web.py)"
    echo "  3) 语音交互演示 (demo_voice.py)"
    echo "  4) 安装依赖"
    echo "  5) 下载模型"
    echo "  6) 运行测试"
    echo "  0) 退出"
    echo ""
}

install_dependencies() {
    echo -e "${GREEN}正在安装 Python 依赖...${NC}"
    cd "$PROJECT_DIR"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
}

download_model() {
    echo ""
    echo -e "${BLUE}可用模型:${NC}"
    echo "  1) qwen2.5-3b  (~2GB, 适合低配设备)"
    echo "  2) qwen2.5-7b  (~4GB, 推荐)"
    echo "  3) qwen2.5-14b (~8GB, 高性能)"
    echo ""
    read -p "请选择模型 (1-3): " model_choice
    
    case $model_choice in
        1) model_name="qwen2.5-3b" ;;
        2) model_name="qwen2.5-7b" ;;
        3) model_name="qwen2.5-14b" ;;
        *) echo -e "${RED}无效选择${NC}"; return ;;
    esac
    
    bash "$SCRIPT_DIR/download_model.sh" "$model_name"
}

run_demo() {
    local demo_type=$1
    local model_path=""
    
    # 查找模型
    if [ -f "$PROJECT_DIR/models/qwen2.5-7b.gguf" ]; then
        model_path="$PROJECT_DIR/models/qwen2.5-7b.gguf"
    elif [ -f "$PROJECT_DIR/models/qwen2.5-3b.gguf" ]; then
        model_path="$PROJECT_DIR/models/qwen2.5-3b.gguf"
    else
        echo -e "${YELLOW}未找到模型文件，将使用模拟模式${NC}"
        model_path="mock_model.gguf"
    fi
    
    cd "$PROJECT_DIR"
    
    case $demo_type in
        "text")
            python python/demo_text.py "$model_path"
            ;;
        "web")
            echo -e "${GREEN}启动 Web 服务器...${NC}"
            echo -e "${BLUE}访问: http://localhost:8000${NC}"
            python python/demo_web.py "$model_path"
            ;;
        "voice")
            python python/demo_voice.py "$model_path"
            ;;
    esac
}

run_tests() {
    cd "$PROJECT_DIR"
    echo -e "${GREEN}运行 Python 测试...${NC}"
    python -m pytest tests/test_assistant.py -v
}

# 主程序
print_banner

while true; do
    print_menu
    read -p "请输入选择 (0-6): " choice
    
    case $choice in
        1)
            run_demo "text"
            ;;
        2)
            run_demo "web"
            ;;
        3)
            run_demo "voice"
            ;;
        4)
            install_dependencies
            ;;
        5)
            download_model
            ;;
        6)
            run_tests
            ;;
        0)
            echo -e "${GREEN}再见！${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选择，请重试${NC}"
            ;;
    esac
done
