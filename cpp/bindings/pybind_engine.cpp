#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "inference_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(cockpit_engine_py, m) {
    m.doc() = "Cockpit Assistant LLM Engine - Python bindings";
    
    // Message结构体
    py::class_<cockpit::Message>(m, "Message")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("role"), py::arg("content"))
        .def_readwrite("role", &cockpit::Message::role)
        .def_readwrite("content", &cockpit::Message::content)
        .def("__repr__", [](const cockpit::Message& msg) {
            return "<Message role='" + msg.role + "' content='" + 
                   msg.content.substr(0, 50) + (msg.content.length() > 50 ? "..." : "") + "'>";
        });
    
    // FunctionCall结构体
    py::class_<cockpit::FunctionCall>(m, "FunctionCall")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("name"), py::arg("arguments"))
        .def_readwrite("name", &cockpit::FunctionCall::name)
        .def_readwrite("arguments", &cockpit::FunctionCall::arguments)
        .def("__repr__", [](const cockpit::FunctionCall& fc) {
            return "<FunctionCall name='" + fc.name + "' arguments='" + fc.arguments + "'>";
        });
    
    // GenerationConfig结构体
    py::class_<cockpit::GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("temperature", &cockpit::GenerationConfig::temperature)
        .def_readwrite("top_p", &cockpit::GenerationConfig::top_p)
        .def_readwrite("top_k", &cockpit::GenerationConfig::top_k)
        .def_readwrite("max_tokens", &cockpit::GenerationConfig::max_tokens)
        .def_readwrite("repeat_penalty", &cockpit::GenerationConfig::repeat_penalty)
        .def_readwrite("stop_sequences", &cockpit::GenerationConfig::stop_sequences);
    
    // EngineConfig结构体
    py::class_<cockpit::EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("model_path", &cockpit::EngineConfig::model_path)
        .def_readwrite("n_ctx", &cockpit::EngineConfig::n_ctx)
        .def_readwrite("n_batch", &cockpit::EngineConfig::n_batch)
        .def_readwrite("n_gpu_layers", &cockpit::EngineConfig::n_gpu_layers)
        .def_readwrite("n_threads", &cockpit::EngineConfig::n_threads)
        .def_readwrite("use_mmap", &cockpit::EngineConfig::use_mmap)
        .def_readwrite("use_mlock", &cockpit::EngineConfig::use_mlock)
        .def_readwrite("chat_template", &cockpit::EngineConfig::chat_template);
    
    // EngineStats结构体
    py::class_<cockpit::EngineStats>(m, "EngineStats")
        .def(py::init<>())
        .def_readonly("tokens_generated", &cockpit::EngineStats::tokens_generated)
        .def_readonly("generation_time_ms", &cockpit::EngineStats::generation_time_ms)
        .def_readonly("tokens_per_second", &cockpit::EngineStats::tokens_per_second)
        .def_readonly("prompt_tokens", &cockpit::EngineStats::prompt_tokens)
        .def_readonly("context_tokens", &cockpit::EngineStats::context_tokens)
        .def("__repr__", [](const cockpit::EngineStats& stats) {
            return "<EngineStats tokens=" + std::to_string(stats.tokens_generated) +
                   " speed=" + std::to_string(stats.tokens_per_second) + " tok/s>";
        });
    
    // LLMEngine类
    py::class_<cockpit::LLMEngine>(m, "LLMEngine")
        .def(py::init<const cockpit::EngineConfig&>(),
             py::arg("config"),
             "Create engine with full configuration")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"),
             py::arg("n_ctx") = 4096,
             py::arg("n_gpu_layers") = 35,
             "Create engine with model path")
        
        .def("is_initialized", &cockpit::LLMEngine::is_initialized,
             "Check if engine is initialized")
        
        .def("generate", &cockpit::LLMEngine::generate,
             py::arg("messages"),
             py::arg("config") = cockpit::GenerationConfig(),
             "Generate response (non-streaming)")
        
        .def("generate_stream", 
             [](cockpit::LLMEngine& self,
                const std::vector<cockpit::Message>& messages,
                py::function callback,
                const cockpit::GenerationConfig& config) {
                 // 包装Python回调
                 cockpit::StreamCallback cpp_callback = 
                     [callback](const std::string& token, bool is_end) {
                         py::gil_scoped_acquire acquire;
                         callback(token, is_end);
                     };
                 
                 // 释放GIL进行推理
                 py::gil_scoped_release release;
                 return self.generate_stream(messages, cpp_callback, config);
             },
             py::arg("messages"),
             py::arg("callback"),
             py::arg("config") = cockpit::GenerationConfig(),
             "Generate response with streaming callback")
        
        .def("parse_function_call", &cockpit::LLMEngine::parse_function_call,
             py::arg("response"),
             "Parse function call from response")
        
        .def("set_function_schema", &cockpit::LLMEngine::set_function_schema,
             py::arg("function_schema"),
             "Set function definitions for function calling")
        
        .def("clear_cache", &cockpit::LLMEngine::clear_cache,
             "Clear KV cache")
        
        .def("save_session", &cockpit::LLMEngine::save_session,
             py::arg("path"),
             "Save session state to file")
        
        .def("load_session", &cockpit::LLMEngine::load_session,
             py::arg("path"),
             "Load session state from file")
        
        .def("get_stats", &cockpit::LLMEngine::get_stats,
             "Get generation statistics")
        
        .def("reset_stats", &cockpit::LLMEngine::reset_stats,
             "Reset statistics")
        
        .def("stop_generation", &cockpit::LLMEngine::stop_generation,
             "Stop current generation")
        
        .def("get_model_info", &cockpit::LLMEngine::get_model_info,
             "Get model information")
        
        .def("get_context_usage", &cockpit::LLMEngine::get_context_usage,
             "Get current context usage")
        
        .def("get_max_context", &cockpit::LLMEngine::get_max_context,
             "Get maximum context size")
        
        .def_property_readonly("context_usage", &cockpit::LLMEngine::get_context_usage)
        .def_property_readonly("max_context", &cockpit::LLMEngine::get_max_context);
    
    // 便捷函数
    m.def("create_message", [](const std::string& role, const std::string& content) {
        return cockpit::Message(role, content);
    }, py::arg("role"), py::arg("content"), "Create a message");
    
    // 版本信息
    m.attr("__version__") = "1.0.0";
}
