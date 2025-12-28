#include "sampler.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace cockpit {

// ============================================================================
// Sampler 实现
// ============================================================================

Sampler::Sampler(const SamplerConfig& config) : config_(config) {
    reset_rng(config.seed);
}

Sampler::~Sampler() = default;

void Sampler::reset_rng(int32_t seed) {
    if (seed < 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(static_cast<unsigned int>(seed));
    }
}

void Sampler::update_config(const SamplerConfig& config) {
    config_ = config;
    if (config.seed >= 0) {
        reset_rng(config.seed);
    }
}

void Sampler::apply_temperature(float* logits, int vocab_size) {
    if (config_.temperature <= 0.0f) {
        // 温度为0时，使用贪婪解码
        return;
    }
    
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= config_.temperature;
    }
}

void Sampler::apply_repetition_penalty(
    float* logits,
    int vocab_size,
    const std::vector<int32_t>& last_tokens
) {
    if (config_.repeat_penalty == 1.0f || last_tokens.empty()) {
        return;
    }
    
    // 只考虑最近的N个tokens
    int start = std::max(0, (int)last_tokens.size() - config_.repeat_last_n);
    
    for (int i = start; i < (int)last_tokens.size(); i++) {
        int32_t token = last_tokens[i];
        if (token < 0 || token >= vocab_size) continue;
        
        // 应用重复惩罚
        if (logits[token] > 0) {
            logits[token] /= config_.repeat_penalty;
        } else {
            logits[token] *= config_.repeat_penalty;
        }
    }
    
    // 应用频率和存在惩罚
    if (config_.frequency_penalty != 0.0f || config_.presence_penalty != 0.0f) {
        std::unordered_map<int32_t, int> token_counts;
        for (int i = start; i < (int)last_tokens.size(); i++) {
            token_counts[last_tokens[i]]++;
        }
        
        for (const auto& [token, count] : token_counts) {
            if (token < 0 || token >= vocab_size) continue;
            
            logits[token] -= config_.frequency_penalty * count;
            logits[token] -= config_.presence_penalty;
        }
    }
}

void Sampler::apply_top_k(float* logits, int vocab_size) {
    if (config_.top_k <= 0 || config_.top_k >= vocab_size) {
        return;
    }
    
    // 找到第k大的值
    std::vector<float> sorted_logits(logits, logits + vocab_size);
    std::partial_sort(sorted_logits.begin(), 
                      sorted_logits.begin() + config_.top_k,
                      sorted_logits.end(),
                      std::greater<float>());
    
    float threshold = sorted_logits[config_.top_k - 1];
    
    // 将低于阈值的logits设为负无穷
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

void Sampler::apply_top_p(float* logits, int vocab_size) {
    if (config_.top_p >= 1.0f) {
        return;
    }
    
    // 创建(logit, index)对并排序
    std::vector<std::pair<float, int>> logit_idx(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        logit_idx[i] = {logits[i], i};
    }
    
    std::sort(logit_idx.begin(), logit_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 计算softmax并累积概率
    float max_logit = logit_idx[0].first;
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(logit_idx[i].first - max_logit);
        sum += probs[i];
    }
    
    // 归一化并累积
    float cumsum = 0.0f;
    int cutoff_idx = vocab_size;
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
        cumsum += probs[i];
        
        if (cumsum > config_.top_p) {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    // 将截断位置之后的logits设为负无穷
    for (int i = cutoff_idx; i < vocab_size; i++) {
        logits[logit_idx[i].second] = -std::numeric_limits<float>::infinity();
    }
}

void Sampler::softmax(float* logits, int vocab_size) {
    // 数值稳定的softmax
    float max_val = *std::max_element(logits, logits + vocab_size);
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }
}

int32_t Sampler::sample(
    float* logits,
    int vocab_size,
    const std::vector<int32_t>& last_tokens
) {
    // 应用各种采样策略
    apply_repetition_penalty(logits, vocab_size, last_tokens);
    apply_temperature(logits, vocab_size);
    apply_top_k(logits, vocab_size);
    apply_top_p(logits, vocab_size);
    
    // 温度为0时使用贪婪解码
    if (config_.temperature <= 0.0f) {
        return std::distance(logits, std::max_element(logits, logits + vocab_size));
    }
    
    // 转换为概率分布
    softmax(logits, vocab_size);
    
    // 多项式采样
    std::discrete_distribution<int32_t> dist(logits, logits + vocab_size);
    return dist(rng_);
}

int32_t Sampler::sample_with_prob(
    float* logits,
    int vocab_size,
    const std::vector<int32_t>& last_tokens,
    float& out_prob
) {
    // 复制logits以保留原始值
    std::vector<float> logits_copy(logits, logits + vocab_size);
    
    int32_t token = sample(logits_copy.data(), vocab_size, last_tokens);
    out_prob = logits_copy[token];
    
    return token;
}

std::vector<std::pair<int32_t, float>> Sampler::get_top_k_tokens(
    float* logits,
    int vocab_size,
    int k
) {
    // 创建(logit, index)对
    std::vector<std::pair<float, int32_t>> logit_idx(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        logit_idx[i] = {logits[i], static_cast<int32_t>(i)};
    }
    
    // 部分排序找到top-k
    k = std::min(k, vocab_size);
    std::partial_sort(logit_idx.begin(), 
                      logit_idx.begin() + k,
                      logit_idx.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 计算softmax概率
    float max_logit = logit_idx[0].first;
    float sum = 0.0f;
    
    std::vector<std::pair<int32_t, float>> result(k);
    for (int i = 0; i < k; i++) {
        float prob = std::exp(logit_idx[i].first - max_logit);
        sum += prob;
        result[i] = {logit_idx[i].second, prob};
    }
    
    // 归一化
    for (auto& [token, prob] : result) {
        prob /= sum;
    }
    
    return result;
}

// ============================================================================
// GreedySampler 实现
// ============================================================================

int32_t GreedySampler::sample(float* logits, int vocab_size) {
    return std::distance(logits, std::max_element(logits, logits + vocab_size));
}

// ============================================================================
// MirostatSampler 实现
// ============================================================================

MirostatSampler::MirostatSampler(float tau, float eta) 
    : tau_(tau), eta_(eta), mu_(2.0f * tau) {
    std::random_device rd;
    rng_.seed(rd());
}

void MirostatSampler::reset() {
    mu_ = 2.0f * tau_;
}

int32_t MirostatSampler::sample(float* logits, int vocab_size) {
    // Mirostat 2采样算法
    
    // 排序logits
    std::vector<std::pair<float, int32_t>> sorted(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        sorted[i] = {logits[i], static_cast<int32_t>(i)};
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 计算softmax
    float max_logit = sorted[0].first;
    float sum = 0.0f;
    std::vector<float> probs(vocab_size);
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(sorted[i].first - max_logit);
        sum += probs[i];
    }
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // 找到满足surprise约束的截断点
    int k = 0;
    float cumsum = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        float s = -std::log2(probs[i]);
        if (s > mu_) {
            k = std::max(1, i);
            break;
        }
        cumsum += probs[i];
        k = i + 1;
    }
    
    // 重新归一化前k个
    std::vector<float> truncated_probs(probs.begin(), probs.begin() + k);
    float truncated_sum = std::accumulate(truncated_probs.begin(), truncated_probs.end(), 0.0f);
    for (auto& p : truncated_probs) {
        p /= truncated_sum;
    }
    
    // 采样
    std::discrete_distribution<int> dist(truncated_probs.begin(), truncated_probs.end());
    int sampled_idx = dist(rng_);
    int32_t sampled_token = sorted[sampled_idx].second;
    
    // 更新mu
    float surprise = -std::log2(probs[sampled_idx]);
    float error = surprise - tau_;
    mu_ -= eta_ * error;
    
    return sampled_token;
}

} // namespace cockpit
