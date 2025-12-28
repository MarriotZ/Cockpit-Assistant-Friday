#!/bin/bash
# ==============================================================================
# 模型下载脚本
# Model download script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 可用模型列表
declare -A MODELS
MODELS["qwen2.5-3b"]="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
MODELS["qwen2.5-7b"]="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
MODELS["qwen2.5-14b"]="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf"

print_usage() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  模型下载脚本${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "用法: $0 <model_name>"
    echo ""
    echo "可用模型:"
    echo -e "  ${BLUE}qwen2.5-3b${NC}   - Qwen2.5 3B (推荐用于低配设备, ~2GB)"
    echo -e "  ${BLUE}qwen2.5-7b${NC}   - Qwen2.5 7B (推荐, ~4GB)"
    echo -e "  ${BLUE}qwen2.5-14b${NC}  - Qwen2.5 14B (高性能, ~8GB)"
    echo ""
    echo "示例:"
    echo "  $0 qwen2.5-7b"
    echo ""
}

download_model() {
    local model_name=$1
    local url=${MODELS[$model_name]}
    
    if [ -z "$url" ]; then
        echo -e "${RED}错误: 未知的模型名称 '$model_name'${NC}"
        print_usage
        exit 1
    fi
    
    # 从URL提取文件名
    local filename=$(basename "$url")
    local filepath="$MODELS_DIR/$filename"
    
    # 创建目录
    mkdir -p "$MODELS_DIR"
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  下载模型: $model_name${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "URL: $url"
    echo "保存路径: $filepath"
    echo ""
    
    # 检查是否已存在
    if [ -f "$filepath" ]; then
        echo -e "${YELLOW}模型文件已存在: $filepath${NC}"
        read -p "是否重新下载? (y/N): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo "跳过下载"
            return
        fi
    fi
    
    # 下载
    echo -e "${GREEN}正在下载...${NC}"
    
    # 检查下载工具
    if command -v wget &> /dev/null; then
        wget -c --progress=bar:force "$url" -O "$filepath"
    elif command -v curl &> /dev/null; then
        curl -L -C - --progress-bar "$url" -o "$filepath"
    else
        echo -e "${RED}错误: 需要 wget 或 curl${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✓ 下载完成: $filepath${NC}"
    
    # 创建符号链接
    local link_name="$MODELS_DIR/${model_name}.gguf"
    if [ ! -e "$link_name" ]; then
        ln -sf "$filename" "$link_name"
        echo -e "${GREEN}✓ 创建符号链接: $link_name${NC}"
    fi
}

# 主程序
if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

model_name=$1

# 检查是否是列出所有模型
if [ "$model_name" == "list" ] || [ "$model_name" == "-l" ]; then
    print_usage
    exit 0
fi

download_model "$model_name"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "运行演示:"
echo "  python python/demo_text.py models/${model_name}.gguf"
echo ""
