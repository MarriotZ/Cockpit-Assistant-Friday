#pragma once

#include <vector>
#include <memory>
#include <cstdint>

namespace cockpit {

/**
 * KV缓存管理器
 * 
 * 管理LLM推理过程中的Key-Value缓存，支持：
 * - 增量更新
 * - 缓存复用
 * - 内存优化
 */
class KVCacheManager {
public:
    struct CacheConfig {
        int n_ctx;          // 最大上下文长度
        int n_layer;        // 模型层数
        int n_head;         // 注意力头数
        int head_dim;       // 每个头的维度
        bool use_fp16;      // 是否使用FP16
    };
    
    explicit KVCacheManager(const CacheConfig& config);
    ~KVCacheManager();
    
    // 禁止拷贝
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    
    /**
     * 获取当前缓存的token数量
     */
    int get_cached_tokens() const { return cached_tokens_; }
    
    /**
     * 获取最大容量
     */
    int get_capacity() const { return config_.n_ctx; }
    
    /**
     * 检查是否可以复用缓存
     * @param new_tokens 新的token序列
     * @return 可复用的token数量
     */
    int check_reusable(const std::vector<int32_t>& new_tokens) const;
    
    /**
     * 更新缓存（追加新token）
     * @param tokens 新的token
     */
    void update(const std::vector<int32_t>& tokens);
    
    /**
     * 清除缓存
     */
    void clear();
    
    /**
     * 截断缓存到指定长度
     * @param length 目标长度
     */
    void truncate(int length);
    
    /**
     * 序列化缓存
     * @return 序列化后的数据
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * 反序列化缓存
     * @param data 序列化数据
     * @return 是否成功
     */
    bool deserialize(const std::vector<uint8_t>& data);
    
    /**
     * 获取内存使用量（字节）
     */
    size_t get_memory_usage() const;

private:
    CacheConfig config_;
    std::vector<int32_t> token_history_;
    int cached_tokens_ = 0;
    
    // 内部缓存数据（由llama.cpp管理）
    void* cache_data_ = nullptr;
};

/**
 * 前缀缓存管理器
 * 
 * 用于管理多个对话的共享前缀缓存
 */
class PrefixCacheManager {
public:
    struct PrefixEntry {
        std::vector<int32_t> tokens;
        std::vector<uint8_t> cache_data;
        int64_t last_access_time;
        int access_count;
    };
    
    explicit PrefixCacheManager(size_t max_entries = 10);
    ~PrefixCacheManager();
    
    /**
     * 查找匹配的前缀
     * @param tokens 查询的token序列
     * @return 匹配的条目索引，如果没有匹配则返回-1
     */
    int find_prefix(const std::vector<int32_t>& tokens) const;
    
    /**
     * 添加前缀缓存
     * @param tokens token序列
     * @param cache_data 缓存数据
     */
    void add_prefix(const std::vector<int32_t>& tokens, const std::vector<uint8_t>& cache_data);
    
    /**
     * 获取前缀缓存
     * @param index 条目索引
     * @return 缓存数据
     */
    const PrefixEntry* get_entry(int index) const;
    
    /**
     * 清除所有缓存
     */
    void clear();
    
    /**
     * 获取缓存数量
     */
    size_t size() const { return entries_.size(); }

private:
    size_t max_entries_;
    std::vector<PrefixEntry> entries_;
    
    void evict_lru();
};

} // namespace cockpit
