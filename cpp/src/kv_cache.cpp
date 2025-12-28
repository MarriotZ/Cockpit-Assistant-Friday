#include "kv_cache.h"

#include <algorithm>
#include <cstring>
#include <chrono>

namespace cockpit {

// ============================================================================
// KVCacheManager 实现
// ============================================================================

KVCacheManager::KVCacheManager(const CacheConfig& config) : config_(config) {
    token_history_.reserve(config.n_ctx);
}

KVCacheManager::~KVCacheManager() {
    // cache_data_由llama.cpp管理，不需要手动释放
}

int KVCacheManager::check_reusable(const std::vector<int32_t>& new_tokens) const {
    int reusable = 0;
    size_t min_len = std::min(new_tokens.size(), token_history_.size());
    
    for (size_t i = 0; i < min_len; i++) {
        if (new_tokens[i] == token_history_[i]) {
            reusable++;
        } else {
            break;
        }
    }
    
    return std::min(reusable, cached_tokens_);
}

void KVCacheManager::update(const std::vector<int32_t>& tokens) {
    token_history_ = tokens;
    cached_tokens_ = tokens.size();
}

void KVCacheManager::clear() {
    token_history_.clear();
    cached_tokens_ = 0;
}

void KVCacheManager::truncate(int length) {
    if (length < 0) {
        length = 0;
    }
    
    if (length < cached_tokens_) {
        token_history_.resize(length);
        cached_tokens_ = length;
    }
}

std::vector<uint8_t> KVCacheManager::serialize() const {
    std::vector<uint8_t> data;
    
    // 序列化token历史
    size_t n_tokens = token_history_.size();
    data.resize(sizeof(n_tokens) + n_tokens * sizeof(int32_t));
    
    std::memcpy(data.data(), &n_tokens, sizeof(n_tokens));
    std::memcpy(data.data() + sizeof(n_tokens), 
                token_history_.data(), 
                n_tokens * sizeof(int32_t));
    
    // TODO: 序列化实际的KV缓存数据
    
    return data;
}

bool KVCacheManager::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < sizeof(size_t)) {
        return false;
    }
    
    size_t n_tokens;
    std::memcpy(&n_tokens, data.data(), sizeof(n_tokens));
    
    if (data.size() < sizeof(n_tokens) + n_tokens * sizeof(int32_t)) {
        return false;
    }
    
    token_history_.resize(n_tokens);
    std::memcpy(token_history_.data(), 
                data.data() + sizeof(n_tokens), 
                n_tokens * sizeof(int32_t));
    
    cached_tokens_ = n_tokens;
    
    return true;
}

size_t KVCacheManager::get_memory_usage() const {
    // 估算内存使用量
    // KV缓存: 2 * n_layer * n_ctx * n_head * head_dim * sizeof(float)
    size_t kv_size = 2 * config_.n_layer * cached_tokens_ * 
                     config_.n_head * config_.head_dim;
    
    if (config_.use_fp16) {
        kv_size *= sizeof(uint16_t);
    } else {
        kv_size *= sizeof(float);
    }
    
    return kv_size + token_history_.capacity() * sizeof(int32_t);
}

// ============================================================================
// PrefixCacheManager 实现
// ============================================================================

PrefixCacheManager::PrefixCacheManager(size_t max_entries) 
    : max_entries_(max_entries) {
    entries_.reserve(max_entries);
}

PrefixCacheManager::~PrefixCacheManager() = default;

int PrefixCacheManager::find_prefix(const std::vector<int32_t>& tokens) const {
    int best_match = -1;
    size_t best_match_len = 0;
    
    for (size_t i = 0; i < entries_.size(); i++) {
        const auto& entry = entries_[i];
        
        // 计算匹配长度
        size_t match_len = 0;
        size_t min_len = std::min(tokens.size(), entry.tokens.size());
        
        for (size_t j = 0; j < min_len; j++) {
            if (tokens[j] == entry.tokens[j]) {
                match_len++;
            } else {
                break;
            }
        }
        
        // 只有当entry完全是tokens的前缀时才算匹配
        if (match_len == entry.tokens.size() && match_len > best_match_len) {
            best_match = i;
            best_match_len = match_len;
        }
    }
    
    return best_match;
}

void PrefixCacheManager::add_prefix(
    const std::vector<int32_t>& tokens, 
    const std::vector<uint8_t>& cache_data
) {
    // 检查是否已存在
    int existing = find_prefix(tokens);
    if (existing >= 0 && entries_[existing].tokens.size() == tokens.size()) {
        // 更新现有条目
        entries_[existing].cache_data = cache_data;
        entries_[existing].last_access_time = 
            std::chrono::system_clock::now().time_since_epoch().count();
        entries_[existing].access_count++;
        return;
    }
    
    // 如果缓存已满，驱逐LRU条目
    if (entries_.size() >= max_entries_) {
        evict_lru();
    }
    
    // 添加新条目
    PrefixEntry entry;
    entry.tokens = tokens;
    entry.cache_data = cache_data;
    entry.last_access_time = 
        std::chrono::system_clock::now().time_since_epoch().count();
    entry.access_count = 1;
    
    entries_.push_back(std::move(entry));
}

const PrefixCacheManager::PrefixEntry* PrefixCacheManager::get_entry(int index) const {
    if (index < 0 || index >= (int)entries_.size()) {
        return nullptr;
    }
    
    // 更新访问信息（const_cast因为这是一个逻辑上的const操作）
    auto& entry = const_cast<PrefixEntry&>(entries_[index]);
    entry.last_access_time = 
        std::chrono::system_clock::now().time_since_epoch().count();
    entry.access_count++;
    
    return &entries_[index];
}

void PrefixCacheManager::clear() {
    entries_.clear();
}

void PrefixCacheManager::evict_lru() {
    if (entries_.empty()) {
        return;
    }
    
    // 找到最久未访问的条目
    auto lru_it = std::min_element(entries_.begin(), entries_.end(),
        [](const PrefixEntry& a, const PrefixEntry& b) {
            return a.last_access_time < b.last_access_time;
        });
    
    entries_.erase(lru_it);
}

} // namespace cockpit
