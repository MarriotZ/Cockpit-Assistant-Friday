#include "inference_engine.h"
#include "sampler.h"
#include "tokenizer.h"
#include "kv_cache.h"

#include <llama.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <sstream>
#include <chrono>
#include <regex>
#include <fstream>

using json = nlohmann::json;

namespace cockpit {

// ============================================================================
// 实现细节
// ============================================================================

struct LLMEngine::Impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    EngineConfig config;
    Tokenizer tokenizer;
    Sampler sampler;
    
    std::vector<int32_t> token_history;
    int n_past = 0;
    
    ~Impl() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
    }
    
    bool initialize() {
        // 初始化llama后端
        llama_backend_init();
        
        // 加载模型
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.n_gpu_layers;
        model_params.use_mmap = config.use_mmap;
        model_params.use_mlock = config.use_mlock;
        
        model = llama_load_model_from_file(config.model_path.c_str(), model_params);
        if (!model) {
            std::cerr << "Failed to load model: " << config.model_path << std::endl;
            return false;
        }
        
        // 创建上下文
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = config.n_ctx;
        ctx_params.n_batch = config.n_batch;
        ctx_params.n_threads = config.n_threads;
        ctx_params.n_threads_batch = config.n_threads;
        
        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            std::cerr << "Failed to create context" << std::endl;
            llama_free_model(model);
            model = nullptr;
            return false;
        }
        
        // 初始化分词器
        tokenizer.init_from_llama_model(model);
        
        std::cout << "Model loaded successfully: " << config.model_path << std::endl;
        std::cout << "  Context size: " << config.n_ctx << std::endl;
        std::cout << "  GPU layers: " << config.n_gpu_layers << std::endl;
        
        return true;
    }
    
    std::string format_messages(const std::vector<Message>& messages) {
        std::vector<std::pair<std::string, std::string>> msg_pairs;
        for (const auto& msg : messages) {
            msg_pairs.emplace_back(msg.role, msg.content);
        }
        return tokenizer.apply_chat_template(msg_pairs, true);
    }
};

// ============================================================================
// 构造函数和析构函数
// ============================================================================

LLMEngine::LLMEngine(const EngineConfig& config) : pimpl_(std::make_unique<Impl>()) {
    pimpl_->config = config;
    if (!pimpl_->initialize()) {
        throw std::runtime_error("Failed to initialize LLM engine");
    }
}

LLMEngine::LLMEngine(const std::string& model_path, int n_ctx, int n_gpu_layers)
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->config.model_path = model_path;
    pimpl_->config.n_ctx = n_ctx;
    pimpl_->config.n_gpu_layers = n_gpu_layers;
    pimpl_->config.n_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
    
    if (!pimpl_->initialize()) {
        throw std::runtime_error("Failed to initialize LLM engine");
    }
}

LLMEngine::~LLMEngine() = default;

LLMEngine::LLMEngine(LLMEngine&&) noexcept = default;
LLMEngine& LLMEngine::operator=(LLMEngine&&) noexcept = default;

bool LLMEngine::is_initialized() const {
    return pimpl_ && pimpl_->model && pimpl_->ctx;
}

// ============================================================================
// 生成函数
// ============================================================================

std::string LLMEngine::generate_stream(
    const std::vector<Message>& messages,
    StreamCallback callback,
    const GenerationConfig& config
) {
    if (!is_initialized()) {
        throw std::runtime_error("Engine not initialized");
    }
    
    stop_flag_ = false;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 格式化消息
    std::string prompt = pimpl_->format_messages(messages);
    
    // 分词
    std::vector<int32_t> tokens = pimpl_->tokenizer.encode(prompt, false, true);
    
    stats_.prompt_tokens = tokens.size();
    
    // 检查上下文长度
    if (tokens.size() >= (size_t)pimpl_->config.n_ctx) {
        throw std::runtime_error("Prompt too long for context window");
    }
    
    // 计算可复用的缓存
    int n_reuse = 0;
    for (size_t i = 0; i < std::min(tokens.size(), pimpl_->token_history.size()); i++) {
        if (tokens[i] == pimpl_->token_history[i]) {
            n_reuse++;
        } else {
            break;
        }
    }
    
    // 如果需要清除部分缓存
    if (n_reuse < pimpl_->n_past) {
        llama_kv_cache_seq_rm(pimpl_->ctx, 0, n_reuse, -1);
        pimpl_->n_past = n_reuse;
    }
    
    // 处理新的prompt tokens
    if ((int)tokens.size() > pimpl_->n_past) {
        std::vector<int32_t> new_tokens(tokens.begin() + pimpl_->n_past, tokens.end());
        
        // 批量处理
        llama_batch batch = llama_batch_init(pimpl_->config.n_batch, 0, 1);
        
        for (size_t i = 0; i < new_tokens.size(); i++) {
            llama_batch_add(batch, new_tokens[i], pimpl_->n_past + i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;
        
        if (llama_decode(pimpl_->ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to decode prompt");
        }
        
        llama_batch_free(batch);
        pimpl_->n_past = tokens.size();
    }
    
    // 更新token历史
    pimpl_->token_history = tokens;
    
    // 配置采样器
    SamplerConfig sampler_config;
    sampler_config.temperature = config.temperature;
    sampler_config.top_p = config.top_p;
    sampler_config.top_k = config.top_k;
    sampler_config.repeat_penalty = config.repeat_penalty;
    pimpl_->sampler.update_config(sampler_config);
    
    // 生成
    std::string result;
    std::vector<int32_t> generated_tokens;
    
    for (int i = 0; i < config.max_tokens; i++) {
        if (stop_flag_) {
            break;
        }
        
        // 获取logits
        float* logits = llama_get_logits_ith(pimpl_->ctx, -1);
        int vocab_size = llama_n_vocab(pimpl_->model);
        
        // 采样
        int32_t new_token = pimpl_->sampler.sample(logits, vocab_size, generated_tokens);
        
        // 检查是否是结束token
        if (pimpl_->tokenizer.is_eos_token(new_token)) {
            break;
        }
        
        // 检查停止序列
        std::string token_text = pimpl_->tokenizer.decode_token(new_token);
        result += token_text;
        
        bool should_stop = false;
        for (const auto& stop_seq : config.stop_sequences) {
            if (result.find(stop_seq) != std::string::npos) {
                // 移除停止序列
                size_t pos = result.find(stop_seq);
                result = result.substr(0, pos);
                should_stop = true;
                break;
            }
        }
        
        if (should_stop) {
            break;
        }
        
        // 调用回调
        if (callback) {
            callback(token_text, false);
        }
        
        generated_tokens.push_back(new_token);
        pimpl_->token_history.push_back(new_token);
        
        // 解码下一个token
        llama_batch batch = llama_batch_init(1, 0, 1);
        llama_batch_add(batch, new_token, pimpl_->n_past, {0}, true);
        
        if (llama_decode(pimpl_->ctx, batch) != 0) {
            llama_batch_free(batch);
            break;
        }
        
        llama_batch_free(batch);
        pimpl_->n_past++;
    }
    
    // 调用结束回调
    if (callback) {
        callback("", true);
    }
    
    // 更新统计
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    stats_.tokens_generated = generated_tokens.size();
    stats_.generation_time_ms = duration.count();
    stats_.tokens_per_second = stats_.tokens_generated / (stats_.generation_time_ms / 1000.0f);
    stats_.context_tokens = pimpl_->n_past;
    
    return result;
}

std::string LLMEngine::generate(
    const std::vector<Message>& messages,
    const GenerationConfig& config
) {
    return generate_stream(messages, nullptr, config);
}

// ============================================================================
// Function Calling
// ============================================================================

void LLMEngine::set_function_schema(const std::string& function_schema) {
    function_schema_ = function_schema;
}

std::optional<FunctionCall> LLMEngine::parse_function_call(const std::string& response) {
    // 尝试解析JSON格式的函数调用
    // 格式1: {"name": "func_name", "arguments": {...}}
    // 格式2: <function_call>{"name": "func_name", "arguments": {...}}</function_call>
    // 格式3: <tool_call>...</tool_call>
    
    std::vector<std::regex> patterns = {
        std::regex(R"(<function_call>\s*(\{.*?\})\s*</function_call>)", std::regex::dotall),
        std::regex(R"(<tool_call>\s*(\{.*?\})\s*</tool_call>)", std::regex::dotall),
        std::regex(R"(\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\})"),
    };
    
    for (const auto& pattern : patterns) {
        std::smatch match;
        if (std::regex_search(response, match, pattern)) {
            std::string json_str = match.size() > 1 ? match[1].str() : match[0].str();
            
            try {
                json j = json::parse(json_str);
                
                if (j.contains("name")) {
                    FunctionCall fc;
                    fc.name = j["name"].get<std::string>();
                    
                    if (j.contains("arguments")) {
                        if (j["arguments"].is_string()) {
                            fc.arguments = j["arguments"].get<std::string>();
                        } else {
                            fc.arguments = j["arguments"].dump();
                        }
                    }
                    
                    return fc;
                }
            } catch (const json::exception& e) {
                // 解析失败，继续尝试下一个模式
                continue;
            }
        }
    }
    
    return std::nullopt;
}

// ============================================================================
// 缓存管理
// ============================================================================

void LLMEngine::clear_cache() {
    if (pimpl_->ctx) {
        llama_kv_cache_clear(pimpl_->ctx);
        pimpl_->n_past = 0;
        pimpl_->token_history.clear();
    }
}

bool LLMEngine::save_session(const std::string& path) {
    if (!is_initialized()) return false;
    
    // 保存token历史
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    
    size_t size = pimpl_->token_history.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(pimpl_->token_history.data()), 
               size * sizeof(int32_t));
    
    // TODO: 保存KV缓存状态
    
    return true;
}

bool LLMEngine::load_session(const std::string& path) {
    if (!is_initialized()) return false;
    
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    pimpl_->token_history.resize(size);
    file.read(reinterpret_cast<char*>(pimpl_->token_history.data()), 
              size * sizeof(int32_t));
    
    // 重新处理tokens
    clear_cache();
    // TODO: 恢复KV缓存状态
    
    return true;
}

// ============================================================================
// 工具函数
// ============================================================================

EngineStats LLMEngine::get_stats() const {
    return stats_;
}

void LLMEngine::reset_stats() {
    stats_ = EngineStats();
}

void LLMEngine::stop_generation() {
    stop_flag_ = true;
}

std::string LLMEngine::get_model_info() const {
    if (!is_initialized()) return "Not initialized";
    
    std::stringstream ss;
    ss << "Model: " << pimpl_->config.model_path << "\n";
    ss << "Context size: " << pimpl_->config.n_ctx << "\n";
    ss << "Vocab size: " << llama_n_vocab(pimpl_->model) << "\n";
    ss << "Embedding size: " << llama_n_embd(pimpl_->model) << "\n";
    
    return ss.str();
}

int LLMEngine::get_context_usage() const {
    return pimpl_ ? pimpl_->n_past : 0;
}

int LLMEngine::get_max_context() const {
    return pimpl_ ? pimpl_->config.n_ctx : 0;
}

} // namespace cockpit
