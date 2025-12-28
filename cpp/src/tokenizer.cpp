#include "tokenizer.h"

#include <llama.h>

#include <sstream>
#include <algorithm>
#include <unordered_set>

namespace cockpit {

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

bool Tokenizer::load_from_model(const std::string& model_path) {
    // 这个方法主要用于独立加载分词器
    // 通常我们通过init_from_llama_model来初始化
    return false;  // TODO: 实现独立加载
}

void Tokenizer::init_from_llama_model(void* model) {
    llama_model_ = model;
    auto* llama_model = static_cast<llama_model*>(model);
    
    vocab_size_ = llama_n_vocab(llama_model);
    
    // 获取特殊tokens
    special_tokens_.bos_token = llama_token_bos(llama_model);
    special_tokens_.eos_token = llama_token_eos(llama_model);
    special_tokens_.pad_token = llama_token_pad(llama_model);
    
    // 尝试找到ChatML特殊tokens
    // 这些token的ID因模型而异，需要通过文本查找
    for (int32_t i = 0; i < vocab_size_; i++) {
        std::string token_text = get_token_text(i);
        if (token_text == "<|im_start|>") {
            special_tokens_.im_start = i;
        } else if (token_text == "<|im_end|>") {
            special_tokens_.im_end = i;
        }
    }
    
    // 自动检测聊天模板
    template_type_ = detect_template_type();
}

ChatTemplateType Tokenizer::detect_template_type() {
    // 根据特殊token判断模板类型
    if (special_tokens_.im_start >= 0 && special_tokens_.im_end >= 0) {
        return ChatTemplateType::CHATML;
    }
    
    // 检查是否有Llama风格的tokens
    for (int32_t i = 0; i < std::min(vocab_size_, (int32_t)10000); i++) {
        std::string token_text = get_token_text(i);
        if (token_text.find("[INST]") != std::string::npos) {
            return ChatTemplateType::LLAMA2;
        }
        if (token_text.find("<|start_header_id|>") != std::string::npos) {
            return ChatTemplateType::LLAMA3;
        }
    }
    
    // 默认使用ChatML
    return ChatTemplateType::CHATML;
}

std::vector<int32_t> Tokenizer::encode(
    const std::string& text,
    bool add_bos,
    bool special
) {
    if (!llama_model_) {
        return {};
    }
    
    auto* model = static_cast<llama_model*>(llama_model_);
    
    // 预分配足够的空间
    std::vector<int32_t> tokens(text.length() + 16);
    
    int n_tokens = llama_tokenize(
        model,
        text.c_str(),
        text.length(),
        tokens.data(),
        tokens.size(),
        add_bos,
        special
    );
    
    if (n_tokens < 0) {
        // 需要更多空间
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            model,
            text.c_str(),
            text.length(),
            tokens.data(),
            tokens.size(),
            add_bos,
            special
        );
    }
    
    tokens.resize(n_tokens);
    return tokens;
}

std::string Tokenizer::decode(
    const std::vector<int32_t>& tokens,
    bool skip_special
) {
    std::string result;
    for (int32_t token : tokens) {
        if (skip_special && is_special_token(token)) {
            continue;
        }
        result += decode_token(token);
    }
    return result;
}

std::string Tokenizer::decode_token(int32_t token) {
    if (!llama_model_) {
        return "";
    }
    
    auto* model = static_cast<llama_model*>(llama_model_);
    
    char buf[256];
    int n = llama_token_to_piece(model, token, buf, sizeof(buf), 0, true);
    
    if (n < 0) {
        return "";
    }
    
    return std::string(buf, n);
}

std::string Tokenizer::get_token_text(int32_t token) const {
    if (!llama_model_) {
        return "";
    }
    
    auto* model = static_cast<llama_model*>(llama_model_);
    
    char buf[256];
    int n = llama_token_to_piece(model, token, buf, sizeof(buf), 0, false);
    
    if (n < 0) {
        return "";
    }
    
    return std::string(buf, n);
}

bool Tokenizer::is_special_token(int32_t token) const {
    return token == special_tokens_.bos_token ||
           token == special_tokens_.eos_token ||
           token == special_tokens_.pad_token ||
           token == special_tokens_.im_start ||
           token == special_tokens_.im_end;
}

bool Tokenizer::is_eos_token(int32_t token) const {
    return token == special_tokens_.eos_token ||
           token == special_tokens_.im_end;
}

void Tokenizer::set_chat_template(ChatTemplateType type) {
    template_type_ = type;
}

void Tokenizer::set_custom_template(const std::string& template_str) {
    custom_template_ = template_str;
    template_type_ = ChatTemplateType::CUSTOM;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt
) {
    switch (template_type_) {
        case ChatTemplateType::CHATML:
        case ChatTemplateType::QWEN:
            return apply_chatml_template(messages, add_generation_prompt);
        case ChatTemplateType::LLAMA2:
            return apply_llama2_template(messages, add_generation_prompt);
        case ChatTemplateType::LLAMA3:
            return apply_llama3_template(messages, add_generation_prompt);
        case ChatTemplateType::CUSTOM:
            // TODO: 实现自定义模板解析
            return apply_chatml_template(messages, add_generation_prompt);
        default:
            return apply_chatml_template(messages, add_generation_prompt);
    }
}

std::string Tokenizer::apply_chatml_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt
) {
    std::ostringstream ss;
    
    for (const auto& [role, content] : messages) {
        ss << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
    }
    
    if (add_generation_prompt) {
        ss << "<|im_start|>assistant\n";
    }
    
    return ss.str();
}

std::string Tokenizer::apply_llama2_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt
) {
    std::ostringstream ss;
    bool first_user = true;
    std::string system_msg;
    
    for (const auto& [role, content] : messages) {
        if (role == "system") {
            system_msg = content;
        } else if (role == "user") {
            ss << "<s>[INST] ";
            if (first_user && !system_msg.empty()) {
                ss << "<<SYS>>\n" << system_msg << "\n<</SYS>>\n\n";
            }
            ss << content << " [/INST]";
            first_user = false;
        } else if (role == "assistant") {
            ss << " " << content << " </s>";
        }
    }
    
    return ss.str();
}

std::string Tokenizer::apply_llama3_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt
) {
    std::ostringstream ss;
    
    ss << "<|begin_of_text|>";
    
    for (const auto& [role, content] : messages) {
        ss << "<|start_header_id|>" << role << "<|end_header_id|>\n\n";
        ss << content << "<|eot_id|>";
    }
    
    if (add_generation_prompt) {
        ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    
    return ss.str();
}

std::string Tokenizer::apply_qwen_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt
) {
    // Qwen使用ChatML格式
    return apply_chatml_template(messages, add_generation_prompt);
}

} // namespace cockpit
