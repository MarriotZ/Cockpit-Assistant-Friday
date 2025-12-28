#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <atomic>

namespace cockpit {

/**
 * 消息结构体
 */
struct Message {
    std::string role;      // "system", "user", "assistant"
    std::string content;   // 消息内容
    
    Message() = default;
    Message(const std::string& r, const std::string& c) : role(r), content(c) {}
};

/**
 * 函数调用结构体
 */
struct FunctionCall {
    std::string name;       // 函数名
    std::string arguments;  // JSON格式的参数
    
    FunctionCall() = default;
    FunctionCall(const std::string& n, const std::string& args) : name(n), arguments(args) {}
};

/**
 * 生成配置
 */
struct GenerationConfig {
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    int max_tokens = 512;
    float repeat_penalty = 1.1f;
    std::vector<std::string> stop_sequences;
    
    GenerationConfig() {
        stop_sequences = {"<|im_end|>", "<|endoftext|>", "</s>"};
    }
};

/**
 * 引擎配置
 */
struct EngineConfig {
    std::string model_path;
    int n_ctx = 4096;           // 上下文长度
    int n_batch = 512;          // 批处理大小
    int n_gpu_layers = 35;      // GPU层数 (-1表示全部)
    int n_threads = 4;          // CPU线程数
    bool use_mmap = true;       // 使用内存映射
    bool use_mlock = false;     // 锁定内存
    std::string chat_template;  // 聊天模板（为空则自动检测）
};

/**
 * 引擎统计信息
 */
struct EngineStats {
    int tokens_generated = 0;
    float generation_time_ms = 0.0f;
    float tokens_per_second = 0.0f;
    int prompt_tokens = 0;
    int context_tokens = 0;
};

// 流式输出回调类型
// token: 生成的token文本
// is_end: 是否是最后一个token
using StreamCallback = std::function<void(const std::string& token, bool is_end)>;

/**
 * LLM推理引擎
 * 
 * 基于llama.cpp的高性能推理引擎，支持流式生成和Function Calling
 */
class LLMEngine {
public:
    /**
     * 构造函数
     * @param config 引擎配置
     */
    explicit LLMEngine(const EngineConfig& config);
    
    /**
     * 简化构造函数
     * @param model_path 模型路径
     * @param n_ctx 上下文长度
     * @param n_gpu_layers GPU层数
     */
    LLMEngine(const std::string& model_path, int n_ctx = 4096, int n_gpu_layers = 35);
    
    ~LLMEngine();
    
    // 禁止拷贝
    LLMEngine(const LLMEngine&) = delete;
    LLMEngine& operator=(const LLMEngine&) = delete;
    
    // 允许移动
    LLMEngine(LLMEngine&&) noexcept;
    LLMEngine& operator=(LLMEngine&&) noexcept;
    
    /**
     * 检查引擎是否已初始化
     */
    bool is_initialized() const;
    
    /**
     * 流式生成响应
     * @param messages 对话消息列表
     * @param callback 流式输出回调
     * @param config 生成配置
     * @return 完整的生成文本
     */
    std::string generate_stream(
        const std::vector<Message>& messages,
        StreamCallback callback,
        const GenerationConfig& config = GenerationConfig()
    );
    
    /**
     * 非流式生成响应
     * @param messages 对话消息列表
     * @param config 生成配置
     * @return 生成的文本
     */
    std::string generate(
        const std::vector<Message>& messages,
        const GenerationConfig& config = GenerationConfig()
    );
    
    /**
     * 解析函数调用
     * @param response LLM响应文本
     * @return 如果包含函数调用则返回FunctionCall，否则返回nullopt
     */
    std::optional<FunctionCall> parse_function_call(const std::string& response);
    
    /**
     * 设置函数定义（用于Function Calling）
     * @param function_schema JSON格式的函数定义
     */
    void set_function_schema(const std::string& function_schema);
    
    /**
     * 清除KV缓存
     */
    void clear_cache();
    
    /**
     * 保存会话状态
     * @param path 保存路径
     * @return 是否成功
     */
    bool save_session(const std::string& path);
    
    /**
     * 加载会话状态
     * @param path 加载路径
     * @return 是否成功
     */
    bool load_session(const std::string& path);
    
    /**
     * 获取统计信息
     */
    EngineStats get_stats() const;
    
    /**
     * 重置统计信息
     */
    void reset_stats();
    
    /**
     * 停止当前生成
     */
    void stop_generation();
    
    /**
     * 获取模型信息
     */
    std::string get_model_info() const;
    
    /**
     * 获取当前上下文使用量
     */
    int get_context_usage() const;
    
    /**
     * 获取最大上下文长度
     */
    int get_max_context() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    
    std::atomic<bool> stop_flag_{false};
    EngineStats stats_;
    std::string function_schema_;
};

} // namespace cockpit
