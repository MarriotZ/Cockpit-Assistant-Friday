# 智能座舱语音助手 - FRIDAY / Cockpit Assistant - FRIDAY

基于大语言模型的智能汽车座舱助手，支持语音交互和车辆控制。

## 功能特性

- 🎙️ **语音交互**: 集成Whisper ASR和Edge TTS
- 🚗 **车辆控制**: 空调、车窗、导航、音乐等Function Calling
- 💬 **多轮对话**: 支持上下文记忆的连续对话
- ⚡ **流式输出**: 低延迟的流式文本生成
- 🔧 **高性能推理**: C++推理引擎 + Python业务层

## 系统架构

<img width="600" height="800" alt="architecture" src="./diagrams/architecture.png" />

## 环境要求

- Ubuntu 20.04+ / macOS 12+
- Python 3.10+
- CMake 3.18+
- CUDA 11.8+ (可选，GPU加速)
- GCC 11+ / Clang 14+

## 快速开始

### 1. 克隆项目并安装依赖

```bash
# 克隆项目
git clone https://github.com/MarriotZ/Cockpit-Assistant-Friday
cd cockpit-assistant-friday

# 安装Python依赖
pip install -r requirements.txt

# 下载并编译llama.cpp
./scripts/setup_llama_cpp.sh
```

### 2. 下载模型

```bash
# 下载推荐的Qwen2.5-7B-Instruct模型
./scripts/download_model.sh qwen2.5-7b

# 或者使用较小的3B模型（适合低配设备）
./scripts/download_model.sh qwen2.5-3b
```

### 3. 编译C++推理引擎

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON  # 使用GPU加速
# cmake .. -DGGML_CUDA=OFF  # 仅CPU
make -j$(nproc)
cd ..
```

### 4. 运行演示

```bash
# 文本交互模式
python python/demo_text.py

# 语音交互模式
python python/demo_voice.py

# Web界面模式
python python/demo_web.py
```

## 项目结构

```
cockpit-assistant/
├── CMakeLists.txt              # CMake构建配置
├── README.md                   # 项目说明
├── requirements.txt            # Python依赖
├── cpp/                        # C++代码
│   ├── include/                # 头文件
│   │   ├── inference_engine.h  # 推理引擎接口
│   │   ├── kv_cache.h          # KV缓存管理
│   │   ├── sampler.h           # 采样策略
│   │   └── tokenizer.h         # 分词器
│   ├── src/                    # 源文件
│   │   ├── inference_engine.cpp
│   │   ├── kv_cache.cpp
│   │   ├── sampler.cpp
│   │   └── tokenizer.cpp
│   └── bindings/               # Python绑定
│       └── pybind_engine.cpp
├── python/                     # Python代码
│   ├── __init__.py
│   ├── cockpit_assistant.py    # 座舱助手主类
│   ├── vehicle_controller.py   # 车辆控制器
│   ├── function_registry.py    # 函数注册表
│   ├── voice_interface.py      # 语音接口
│   ├── demo_text.py            # 文本演示
│   ├── demo_voice.py           # 语音演示
│   └── demo_web.py             # Web演示
├── models/                     # 模型存放目录
├── config/                     # 配置文件
│   └── config.yaml
├── tests/                      # 测试
│   ├── test_engine.cpp
│   └── test_assistant.py
└── scripts/                    # 脚本
    ├── setup_llama_cpp.sh
    └── download_model.sh
```

## 配置说明

编辑 `config/config.yaml` 自定义配置：

```yaml
model:
  path: "models/qwen2.5-7b-instruct-q4_k_m.gguf"
  n_ctx: 4096
  n_gpu_layers: 35
  
inference:
  temperature: 0.7
  max_tokens: 512
  top_p: 0.9
  
voice:
  asr_model: "small"
  tts_voice: "zh-CN-XiaoxiaoNeural"
  wake_word: "Hey Friday"
```

## API使用示例

### Python API

```python
from cockpit_assistant import CockpitAssistant

# 初始化助手
assistant = CockpitAssistant("models/qwen2.5-7b-instruct-q4_k_m.gguf")

# 文本对话
async for token in assistant.chat("把空调打开，温度调到26度"):
    print(token, end="", flush=True)
```

### 带语音的使用

```python
from voice_interface import CockpitVoiceAssistant

assistant = CockpitVoiceAssistant("models/qwen2.5-7b-instruct-q4_k_m.gguf")

# 处理语音输入
async for audio_chunk in assistant.process_voice_input(audio_data):
    play_audio(audio_chunk)
```

## Function Calling

系统支持以下车辆控制函数：

| 函数名 | 描述 | 参数 |
|--------|------|------|
| `control_air_conditioner` | 控制空调 | action, temperature, fan_speed |
| `control_window` | 控制车窗 | position, action |
| `navigate_to` | 设置导航 | destination, via_points |
| `play_music` | 播放音乐 | query, action |
| `get_vehicle_status` | 查询车辆状态 | info_type |
| `control_lights` | 控制车灯 | light_type, action |
| `make_phone_call` | 拨打电话 | contact |

## 性能指标
待重测，代码进行了优化

~~在 NVIDIA RTX 4090 上使用 Qwen2.5-7B-Instruct-Q4_K_M：

| 指标 | 数值 |
|------|------|
| 首Token延迟 | ~150ms |
| 生成速度 | ~45 tokens/s |
| 内存占用 | ~6GB |
| ASR延迟 | ~200ms |~~


## 扩展开发

### 添加新的控制函数

1. 在 `python/function_registry.py` 中定义函数schema
2. 在 `python/vehicle_controller.py` 中实现处理逻辑
3. 更新系统提示词

### 适配新硬件

修改 `CMakeLists.txt` 中的编译选项以适配不同硬件：

- NVIDIA Jetson: `-DGGML_CUDA=ON`
- Apple Silicon: `-DGGML_METAL=ON`
- 高通平台: 需要使用QNN后端

## 正在进行的工作
### 手机端互联控制

**目前正在开发iOS和Android两端对该助手进行远程控制的方案**

## 许可证

MIT License

## 致谢
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen](https://github.com/QwenLM/Qwen)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [edge-tts](https://github.com/rany2/edge-tts)






