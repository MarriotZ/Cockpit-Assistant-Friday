#!/bin/bash
# ==============================================================================
# llama.cpp 安装脚本
# Setup script for llama.cpp
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_DIR="$PROJECT_DIR/third_party"
LLAMA_CPP_DIR="$THIRD_PARTY_DIR/llama.cpp"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  llama.cpp 安装脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 创建目录
mkdir -p "$THIRD_PARTY_DIR"

# 检查是否已存在
if [ -d "$LLAMA_CPP_DIR" ]; then
    echo -e "${YELLOW}llama.cpp 已存在，正在更新...${NC}"
    cd "$LLAMA_CPP_DIR"
    git pull
else
    echo -e "${GREEN}正在克隆 llama.cpp...${NC}"
    cd "$THIRD_PARTY_DIR"
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd "$LLAMA_CPP_DIR"

# 检测CUDA
USE_CUDA="OFF"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}检测到 CUDA，将启用 GPU 加速${NC}"
    USE_CUDA="ON"
else
    echo -e "${YELLOW}未检测到 CUDA，将使用 CPU 模式${NC}"
fi

# 检测Metal (macOS)
USE_METAL="OFF"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}检测到 macOS，将启用 Metal 加速${NC}"
    USE_METAL="ON"
fi

# 创建构建目录
mkdir -p build
cd build

# 配置CMake
echo -e "${GREEN}正在配置 CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=$USE_CUDA \
    -DGGML_METAL=$USE_METAL \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON

# 编译
echo -e "${GREEN}正在编译...${NC}"
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  llama.cpp 安装完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "安装路径: $LLAMA_CPP_DIR"
echo ""
echo "可执行文件:"
echo "  - $LLAMA_CPP_DIR/build/bin/llama-cli"
echo "  - $LLAMA_CPP_DIR/build/bin/llama-server"
echo ""
echo "下一步: 运行 ./scripts/download_model.sh 下载模型"
