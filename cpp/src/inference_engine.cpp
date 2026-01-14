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
#include <thread>
#include <algorithm>

using json = nlohmann::json;

namespace cockpit {

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

// 获取vocab指针
// 获取vocab指针
static const llama_vocab* get_vocab(llama_model* model) {
    return llama_model_get_vocab(model);
}

// 获取vocab大小
static int get_vocab_size(llama_model* model) {
    const llama_vocab* vocab = get_vocab(model);
    return llama_vocab_n_tokens(vocab);
}

// 向batch添加token
static void batch_add(llama_batch& batch, llama_token token, llama_pos pos, 
                      const std::vector<llama_seq_id>& seq_ids, bool logits) {
    batch.token[batch.n_tokens] = token;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = static_cast<int32_t>(seq_ids.size());
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

// ============================================================================
// KV Cache
// ============================================================================

static void kv_cache_clear(llama_context* ctx) {
    llama_memory_clear(llama_get_memory(ctx), true);
}

static void kv_cache_seq_rm(llama_context* ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);
}


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
    unsigned int hw_threads = std::thread::hardware_concurrency();
    // 这里可以自己配
    pimpl_->config.n_threads = static_cast<int>((std::max)(1u, hw_threads / 2));
    
    if (!pimpl_->initialize()) {
        throw std::runtime_error("Failed to initialize LLM engine");
    }
}

LLMEngine::~LLMEngine() = default;

// 手动实现移动构造函数 (因为 std::atomic 不能被移动)
LLMEngine::LLMEngine(LLMEngine&& other) noexcept
    : pimpl_(std::move(other.pimpl_)),
      stop_flag_(other.stop_flag_.load()),
      stats_(other.stats_),
      function_schema_(std::move(other.function_schema_)) {
}

// 手动实现移动赋值运算符
LLMEngine& LLMEngine::operator=(LLMEngine&& other) noexcept {
    if (this != &other) {
        pimpl_ = std::move(other.pimpl_);
        stop_flag_.store(other.stop_flag_.load());
        stats_ = other.stats_;
        function_schema_ = std::move(other.function_schema_);
    }
    return *this;
}

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
    
    stats_.prompt_tokens = static_cast<int>(tokens.size());
    
    // 检查上下文长度
    if (tokens.size() >= static_cast<size_t>(pimpl_->config.n_ctx)) {
        throw std::runtime_error("Prompt too long for context window");
    }
    
    // 计算可复用的缓存
    int n_reuse = 0;
    for (size_t i = 0; i < (std::min)(tokens.size(), pimpl_->token_history.size()); i++) {
        if (tokens[i] == pimpl_->token_history[i]) {
            n_reuse++;
        } else {
            break;
        }
    }
    
    // 如果需要清除部分缓存
    if (n_reuse < pimpl_->n_past) {
        kv_cache_seq_rm(pimpl_->ctx, 0, n_reuse, -1);
        pimpl_->n_past = n_reuse;
    }
    
    // 处理新的prompt tokens
    if (static_cast<int>(tokens.size()) > pimpl_->n_past) {
        std::vector<int32_t> new_tokens(tokens.begin() + pimpl_->n_past, tokens.end());
        
        // 批量处理
        llama_batch batch = llama_batch_init(pimpl_->config.n_batch, 0, 1);
        
        for (size_t i = 0; i < new_tokens.size(); i++) {
            batch_add(batch, new_tokens[i], pimpl_->n_past + static_cast<int>(i), {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;
        
        if (llama_decode(pimpl_->ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to decode prompt");
        }
        
        llama_batch_free(batch);
        pimpl_->n_past = static_cast<int>(tokens.size());
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
    
    // 获取vocab大小
    int vocab_size = get_vocab_size(pimpl_->model);
    
    for (int i = 0; i < config.max_tokens; i++) {
        if (stop_flag_.load()) {
            break;
        }
        
        // 获取logits
        float* logits = llama_get_logits_ith(pimpl_->ctx, -1);
        
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
            // 这里需要优化，通过维护滑动窗口来处理后续可能越来越慢的问题
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
        batch_add(batch, new_token, pimpl_->n_past, {0}, true);
        
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
    
    stats_.tokens_generated = static_cast<int>(generated_tokens.size());
    stats_.generation_time_ms = static_cast<float>(duration.count());
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
    // 解析JSON格式的函数调用
    std::vector<std::regex> patterns = {
        std::regex(R"(<function_call>\s*(\{[\s\S]*?\})\s*</function_call>)"),
        std::regex(R"(<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>)"),
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
        kv_cache_clear(pimpl_->ctx);
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
    
    int vocab_size = get_vocab_size(pimpl_->model);
    
    std::stringstream ss;
    ss << "Model: " << pimpl_->config.model_path << "\n";
    ss << "Context size: " << pimpl_->config.n_ctx << "\n";
    ss << "Vocab size: " << vocab_size << "\n";
    ss << "Embedding size: " << llama_n_embd(pimpl_->model) << "\n";
    
    return ss.str();
}

int LLMEngine::get_context_usage() const {
    return pimpl_ ? pimpl_->n_past : 0;
}

int LLMEngine::get_max_context() const {
    return pimpl_ ? pimpl_->config.n_ctx : 0;
}

} 
