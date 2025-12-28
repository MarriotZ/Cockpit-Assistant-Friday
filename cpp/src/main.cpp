#include "inference_engine.h"

#include <iostream>
#include <string>
#include <sstream>

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> [options]\n"
              << "\nOptions:\n"
              << "  -c, --ctx <size>      Context size (default: 4096)\n"
              << "  -g, --gpu <layers>    GPU layers (default: 35, -1 for all)\n"
              << "  -t, --temp <value>    Temperature (default: 0.7)\n"
              << "  --top-p <value>       Top-P (default: 0.9)\n"
              << "  --top-k <value>       Top-K (default: 40)\n"
              << "  -h, --help            Show this help\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string model_path;
    int n_ctx = 4096;
    int n_gpu_layers = 35;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    
    // 解析参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-c" || arg == "--ctx") && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--temp") && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (model_path.empty()) {
            model_path = arg;
        }
    }
    
    if (model_path.empty()) {
        std::cerr << "Error: Model path is required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "=== Cockpit Assistant CLI ===\n";
    std::cout << "Loading model: " << model_path << "\n";
    std::cout << "Context size: " << n_ctx << "\n";
    std::cout << "GPU layers: " << n_gpu_layers << "\n";
    std::cout << std::endl;
    
    try {
        // 创建引擎
        cockpit::LLMEngine engine(model_path, n_ctx, n_gpu_layers);
        
        std::cout << "Model loaded successfully!\n";
        std::cout << engine.get_model_info() << "\n";
        
        // 系统提示
        const std::string system_prompt = R"(你是一个智能汽车座舱助手，负责帮助驾驶员控制车辆功能。

你可以执行以下操作：
1. 控制空调（开关、调节温度和风量）
2. 控制车窗（打开、关闭、半开）
3. 设置导航目的地
4. 播放音乐
5. 查询车辆状态

请用简洁友好的语气回复用户。当需要执行车辆控制时，请以JSON格式返回函数调用：
{"name": "函数名", "arguments": {"参数名": "参数值"}}

可用的函数：
- control_air_conditioner: 控制空调 (action: on/off/adjust, temperature: 16-30, fan_speed: 1-5)
- control_window: 控制车窗 (position: front_left/front_right/rear_left/rear_right/all, action: open/close/half_open)
- navigate_to: 设置导航 (destination: 目的地名称)
- play_music: 播放音乐 (query: 搜索词, action: play/pause/next/previous)
- get_vehicle_status: 查询状态 (info_type: battery/tire_pressure/oil/mileage/all)

回复要简洁，适合语音播报。)";
        
        std::vector<cockpit::Message> messages;
        messages.emplace_back("system", system_prompt);
        
        // 配置生成参数
        cockpit::GenerationConfig gen_config;
        gen_config.temperature = temperature;
        gen_config.top_p = top_p;
        gen_config.top_k = top_k;
        gen_config.max_tokens = 512;
        
        std::cout << "Type 'quit' to exit, 'clear' to reset conversation\n\n";
        
        // 主循环
        while (true) {
            std::cout << "User: ";
            std::string input;
            std::getline(std::cin, input);
            
            if (input.empty()) continue;
            
            if (input == "quit" || input == "exit") {
                break;
            }
            
            if (input == "clear" || input == "reset") {
                messages.clear();
                messages.emplace_back("system", system_prompt);
                engine.clear_cache();
                std::cout << "Conversation cleared.\n\n";
                continue;
            }
            
            if (input == "stats") {
                auto stats = engine.get_stats();
                std::cout << "Stats:\n";
                std::cout << "  Tokens generated: " << stats.tokens_generated << "\n";
                std::cout << "  Generation time: " << stats.generation_time_ms << "ms\n";
                std::cout << "  Tokens/sec: " << stats.tokens_per_second << "\n";
                std::cout << "  Context usage: " << engine.get_context_usage() 
                         << "/" << engine.get_max_context() << "\n\n";
                continue;
            }
            
            // 添加用户消息
            messages.emplace_back("user", input);
            
            std::cout << "Assistant: ";
            std::cout.flush();
            
            // 流式生成
            std::string response = engine.generate_stream(
                messages,
                [](const std::string& token, bool is_end) {
                    if (!is_end) {
                        std::cout << token;
                        std::cout.flush();
                    }
                },
                gen_config
            );
            
            std::cout << "\n\n";
            
            // 检查函数调用
            auto func_call = engine.parse_function_call(response);
            if (func_call) {
                std::cout << "[Function Call] " << func_call->name 
                         << "(" << func_call->arguments << ")\n\n";
            }
            
            // 添加助手消息
            messages.emplace_back("assistant", response);
            
            // 打印统计
            auto stats = engine.get_stats();
            std::cout << "[" << stats.tokens_per_second << " tokens/s, "
                     << engine.get_context_usage() << "/" << engine.get_max_context() 
                     << " ctx]\n\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Goodbye!\n";
    return 0;
}
