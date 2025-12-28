#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace cockpit {

/**
 * 聊天模板类型
 */
enum class ChatTemplateType {
    AUTO,       // 自动检测
    CHATML,     // ChatML格式 (<|im_start|>...<|im_end|>)
    LLAMA2,     // Llama 2格式 ([INST]...[/INST])
    LLAMA3,     // Llama 3格式
    QWEN,       // Qwen格式
    CUSTOM      // 自定义模板
};

/**
 * 特殊token
 */
struct SpecialTokens {
    int32_t bos_token = 1;      // 序列开始
    int32_t eos_token = 2;      // 序列结束
    int32_t pad_token = 0;      // 填充
    int32_t unk_token = 0;      // 未知
    
    // ChatML特殊token
    int32_t im_start = -1;      // <|im_start|>
    int32_t im_end = -1;        // <|im_end|>
};

/**
 * 分词器
 * 
 * 处理文本到token的转换，支持多种聊天模板
 */
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    
    // 禁止拷贝
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    
    /**
     * 从模型文件加载分词器
     * @param model_path 模型文件路径
     * @return 是否成功
     */
    bool load_from_model(const std::string& model_path);
    
    /**
     * 从llama.cpp模型指针初始化
     * @param model llama_model指针
     */
    void init_from_llama_model(void* model);
    
    /**
     * 编码文本为tokens
     * @param text 输入文本
     * @param add_bos 是否添加BOS token
     * @param special 是否处理特殊token
     * @return token序列
     */
    std::vector<int32_t> encode(
        const std::string& text,
        bool add_bos = false,
        bool special = true
    );
    
    /**
     * 解码tokens为文本
     * @param tokens token序列
     * @param skip_special 是否跳过特殊token
     * @return 解码后的文本
     */
    std::string decode(
        const std::vector<int32_t>& tokens,
        bool skip_special = true
    );
    
    /**
     * 解码单个token
     * @param token token ID
     * @return 解码后的文本
     */
    std::string decode_token(int32_t token);
    
    /**
     * 应用聊天模板
     * @param messages 消息列表 (vector of {role, content})
     * @param add_generation_prompt 是否添加生成提示
     * @return 格式化后的文本
     */
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true
    );
    
    /**
     * 设置聊天模板类型
     */
    void set_chat_template(ChatTemplateType type);
    
    /**
     * 设置自定义聊天模板
     * @param template_str Jinja2风格的模板字符串
     */
    void set_custom_template(const std::string& template_str);
    
    /**
     * 获取词表大小
     */
    int32_t vocab_size() const { return vocab_size_; }
    
    /**
     * 获取特殊tokens
     */
    const SpecialTokens& special_tokens() const { return special_tokens_; }
    
    /**
     * 检查token是否是特殊token
     */
    bool is_special_token(int32_t token) const;
    
    /**
     * 检查token是否是EOS token
     */
    bool is_eos_token(int32_t token) const;
    
    /**
     * 获取token文本
     */
    std::string get_token_text(int32_t token) const;

private:
    void* llama_model_ = nullptr;
    int32_t vocab_size_ = 0;
    SpecialTokens special_tokens_;
    ChatTemplateType template_type_ = ChatTemplateType::AUTO;
    std::string custom_template_;
    
    // 检测模型的聊天模板类型
    ChatTemplateType detect_template_type();
    
    // 各种模板的实现
    std::string apply_chatml_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt
    );
    
    std::string apply_llama2_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt
    );
    
    std::string apply_llama3_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt
    );
    
    std::string apply_qwen_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt
    );
};

} // namespace cockpit
