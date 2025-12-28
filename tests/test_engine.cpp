/**
 * 测试套件 - C++ 推理引擎
 * 
 * 运行: cd build && ctest --output-on-failure
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <optional>

#include "inference_engine.h"
#include "sampler.h"
#include "kv_cache.h"
#include "tokenizer.h"

using namespace cockpit;

// ==============================================================================
// Sampler 测试
// ==============================================================================

class SamplerTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.temperature = 1.0f;
        config.top_p = 0.9f;
        config.top_k = 40;
        config.repeat_penalty = 1.1f;
        config.seed = 42;  // 固定种子以保证可重复性
    }
    
    SamplerConfig config;
};

TEST_F(SamplerTest, GreedySampling) {
    // 温度为0时应该使用贪婪采样
    config.temperature = 0.0f;
    Sampler sampler(config);
    
    std::vector<float> logits = {0.1f, 0.5f, 0.2f, 0.9f, 0.3f};
    int32_t result = sampler.sample(logits.data(), logits.size(), {});
    
    // 应该选择最大值的索引 (0.9f 在索引 3)
    EXPECT_EQ(result, 3);
}

TEST_F(SamplerTest, TopKFiltering) {
    config.top_k = 2;
    config.temperature = 1.0f;
    Sampler sampler(config);
    
    std::vector<float> logits = {1.0f, 5.0f, 2.0f, 4.0f, 3.0f};
    
    // 多次采样，结果应该只在top-2中
    for (int i = 0; i < 100; i++) {
        int32_t result = sampler.sample(logits.data(), logits.size(), {});
        EXPECT_TRUE(result == 1 || result == 3);  // 5.0 和 4.0 的索引
    }
}

TEST_F(SamplerTest, RepetitionPenalty) {
    config.repeat_penalty = 2.0f;
    config.repeat_last_n = 10;
    Sampler sampler(config);
    
    std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<int32_t> last_tokens = {0, 1};  // 惩罚索引0和1
    
    // 采样多次，被惩罚的token应该更少被选中
    int count_penalized = 0;
    for (int i = 0; i < 1000; i++) {
        std::vector<float> logits_copy = logits;
        int32_t result = sampler.sample(logits_copy.data(), logits_copy.size(), last_tokens);
        if (result == 0 || result == 1) {
            count_penalized++;
        }
    }
    
    // 被惩罚的token应该被选中更少 (大约小于40%)
    EXPECT_LT(count_penalized, 500);
}

TEST_F(SamplerTest, GetTopKTokens) {
    Sampler sampler(config);
    
    std::vector<float> logits = {0.1f, 0.5f, 0.2f, 0.9f, 0.3f};
    auto top_tokens = sampler.get_top_k_tokens(logits.data(), logits.size(), 3);
    
    EXPECT_EQ(top_tokens.size(), 3);
    // 第一个应该是最大值
    EXPECT_EQ(top_tokens[0].first, 3);  // 0.9f 的索引
}

// ==============================================================================
// GreedySampler 测试
// ==============================================================================

TEST(GreedySamplerTest, AlwaysSelectsMax) {
    GreedySampler sampler;
    
    std::vector<float> logits = {0.1f, 0.5f, 0.2f, 0.9f, 0.3f};
    int32_t result = sampler.sample(logits.data(), logits.size());
    
    EXPECT_EQ(result, 3);
}

// ==============================================================================
// KVCacheManager 测试
// ==============================================================================

class KVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.n_ctx = 1024;
        config.n_layer = 32;
        config.n_head = 32;
        config.head_dim = 128;
        config.use_fp16 = true;
    }
    
    KVCacheManager::CacheConfig config;
};

TEST_F(KVCacheTest, InitialState) {
    KVCacheManager cache(config);
    
    EXPECT_EQ(cache.get_cached_tokens(), 0);
    EXPECT_EQ(cache.get_capacity(), 1024);
}

TEST_F(KVCacheTest, UpdateCache) {
    KVCacheManager cache(config);
    
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
    cache.update(tokens);
    
    EXPECT_EQ(cache.get_cached_tokens(), 5);
}

TEST_F(KVCacheTest, CheckReusable) {
    KVCacheManager cache(config);
    
    std::vector<int32_t> tokens1 = {1, 2, 3, 4, 5};
    cache.update(tokens1);
    
    // 完全相同的序列
    std::vector<int32_t> tokens2 = {1, 2, 3, 4, 5};
    EXPECT_EQ(cache.check_reusable(tokens2), 5);
    
    // 部分匹配
    std::vector<int32_t> tokens3 = {1, 2, 3, 6, 7};
    EXPECT_EQ(cache.check_reusable(tokens3), 3);
    
    // 完全不匹配
    std::vector<int32_t> tokens4 = {6, 7, 8};
    EXPECT_EQ(cache.check_reusable(tokens4), 0);
}

TEST_F(KVCacheTest, ClearCache) {
    KVCacheManager cache(config);
    
    std::vector<int32_t> tokens = {1, 2, 3};
    cache.update(tokens);
    EXPECT_EQ(cache.get_cached_tokens(), 3);
    
    cache.clear();
    EXPECT_EQ(cache.get_cached_tokens(), 0);
}

TEST_F(KVCacheTest, TruncateCache) {
    KVCacheManager cache(config);
    
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
    cache.update(tokens);
    
    cache.truncate(3);
    EXPECT_EQ(cache.get_cached_tokens(), 3);
}

TEST_F(KVCacheTest, Serialization) {
    KVCacheManager cache1(config);
    
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
    cache1.update(tokens);
    
    // 序列化
    auto data = cache1.serialize();
    EXPECT_GT(data.size(), 0);
    
    // 反序列化
    KVCacheManager cache2(config);
    EXPECT_TRUE(cache2.deserialize(data));
    EXPECT_EQ(cache2.get_cached_tokens(), 5);
}

// ==============================================================================
// PrefixCacheManager 测试
// ==============================================================================

TEST(PrefixCacheTest, AddAndFind) {
    PrefixCacheManager cache(5);
    
    std::vector<int32_t> tokens1 = {1, 2, 3};
    std::vector<uint8_t> data1 = {0x01, 0x02, 0x03};
    cache.add_prefix(tokens1, data1);
    
    // 查找精确匹配
    int idx = cache.find_prefix({1, 2, 3, 4, 5});
    EXPECT_GE(idx, 0);
    
    // 查找不匹配
    idx = cache.find_prefix({4, 5, 6});
    EXPECT_EQ(idx, -1);
}

TEST(PrefixCacheTest, LRUEviction) {
    PrefixCacheManager cache(2);  // 最多2个条目
    
    // 添加3个条目，应该驱逐最旧的
    cache.add_prefix({1}, {0x01});
    cache.add_prefix({2}, {0x02});
    cache.add_prefix({3}, {0x03});
    
    EXPECT_EQ(cache.size(), 2);
    
    // 第一个应该被驱逐
    EXPECT_EQ(cache.find_prefix({1, 4, 5}), -1);
}

// ==============================================================================
// Message 结构体测试
// ==============================================================================

TEST(MessageTest, Construction) {
    Message msg1;
    EXPECT_TRUE(msg1.role.empty());
    EXPECT_TRUE(msg1.content.empty());
    
    Message msg2("user", "Hello");
    EXPECT_EQ(msg2.role, "user");
    EXPECT_EQ(msg2.content, "Hello");
}

// ==============================================================================
// FunctionCall 结构体测试
// ==============================================================================

TEST(FunctionCallTest, Construction) {
    FunctionCall fc1;
    EXPECT_TRUE(fc1.name.empty());
    EXPECT_TRUE(fc1.arguments.empty());
    
    FunctionCall fc2("test_func", R"({"key": "value"})");
    EXPECT_EQ(fc2.name, "test_func");
    EXPECT_EQ(fc2.arguments, R"({"key": "value"})");
}

// ==============================================================================
// GenerationConfig 测试
// ==============================================================================

TEST(GenerationConfigTest, DefaultValues) {
    GenerationConfig config;
    
    EXPECT_FLOAT_EQ(config.temperature, 0.7f);
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_EQ(config.top_k, 40);
    EXPECT_EQ(config.max_tokens, 512);
    EXPECT_GT(config.stop_sequences.size(), 0);
}

// ==============================================================================
// EngineConfig 测试
// ==============================================================================

TEST(EngineConfigTest, DefaultValues) {
    EngineConfig config;
    
    EXPECT_EQ(config.n_ctx, 4096);
    EXPECT_EQ(config.n_batch, 512);
    EXPECT_EQ(config.n_gpu_layers, 35);
    EXPECT_TRUE(config.use_mmap);
    EXPECT_FALSE(config.use_mlock);
}

// ==============================================================================
// 主函数
// ==============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
