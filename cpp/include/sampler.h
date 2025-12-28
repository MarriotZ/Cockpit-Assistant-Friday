#pragma once

#include <vector>
#include <random>
#include <cstdint>

namespace cockpit {

/**
 * 采样配置
 */
struct SamplerConfig {
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int32_t seed = -1;  // -1表示随机种子
};

/**
 * Token采样器
 * 
 * 实现各种采样策略：
 * - 温度采样
 * - Top-K采样
 * - Top-P (nucleus)采样
 * - 重复惩罚
 * - 频率/存在惩罚
 */
class Sampler {
public:
    explicit Sampler(const SamplerConfig& config = SamplerConfig());
    ~Sampler();
    
    /**
     * 从logits中采样一个token
     * @param logits 模型输出的logits
     * @param vocab_size 词表大小
     * @param last_tokens 最近生成的tokens（用于重复惩罚）
     * @return 采样得到的token ID
     */
    int32_t sample(
        float* logits,
        int vocab_size,
        const std::vector<int32_t>& last_tokens = {}
    );
    
    /**
     * 采样并返回概率
     * @param logits 模型输出的logits
     * @param vocab_size 词表大小
     * @param last_tokens 最近生成的tokens
     * @param out_prob 输出采样概率
     * @return 采样得到的token ID
     */
    int32_t sample_with_prob(
        float* logits,
        int vocab_size,
        const std::vector<int32_t>& last_tokens,
        float& out_prob
    );
    
    /**
     * 获取Top-K候选tokens
     * @param logits 模型输出的logits
     * @param vocab_size 词表大小
     * @param k 返回的候选数量
     * @return 候选tokens及其概率
     */
    std::vector<std::pair<int32_t, float>> get_top_k_tokens(
        float* logits,
        int vocab_size,
        int k
    );
    
    /**
     * 更新配置
     */
    void update_config(const SamplerConfig& config);
    
    /**
     * 获取当前配置
     */
    const SamplerConfig& get_config() const { return config_; }
    
    /**
     * 重置随机数生成器
     */
    void reset_rng(int32_t seed = -1);

private:
    SamplerConfig config_;
    std::mt19937 rng_;
    
    // 应用温度
    void apply_temperature(float* logits, int vocab_size);
    
    // 应用重复惩罚
    void apply_repetition_penalty(
        float* logits,
        int vocab_size,
        const std::vector<int32_t>& last_tokens
    );
    
    // Top-K过滤
    void apply_top_k(float* logits, int vocab_size);
    
    // Top-P过滤
    void apply_top_p(float* logits, int vocab_size);
    
    // Softmax
    void softmax(float* logits, int vocab_size);
};

/**
 * 贪婪采样器（总是选择概率最高的token）
 */
class GreedySampler {
public:
    int32_t sample(float* logits, int vocab_size);
};

/**
 * Mirostat采样器
 * 用于保持输出的"惊讶度"恒定
 */
class MirostatSampler {
public:
    MirostatSampler(float tau = 5.0f, float eta = 0.1f);
    
    int32_t sample(float* logits, int vocab_size);
    
    void reset();

private:
    float tau_;
    float eta_;
    float mu_;  // 当前mu值
    std::mt19937 rng_;
};

} // namespace cockpit
