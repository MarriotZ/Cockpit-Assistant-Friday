cmake_minimum_required(VERSION 3.18)
project(cockpit_assistant VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 选项
option(GGML_CUDA "Enable CUDA support" OFF)
option(GGML_METAL "Enable Metal support (macOS)" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" ON)

# 编译优化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# 查找依赖
find_package(Threads REQUIRED)

# llama.cpp路径
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp")

if(NOT EXISTS "${LLAMA_CPP_DIR}")
    message(STATUS "llama.cpp not found, will be downloaded during build")
endif()

# 添加llama.cpp子目录（如果存在）
if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    # 设置llama.cpp选项
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    
    if(GGML_CUDA)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
    endif()
    
    if(GGML_METAL)
        set(GGML_METAL ON CACHE BOOL "" FORCE)
    endif()
    
    add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)
endif()

# JSON库
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# 头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
)

# 源文件
set(ENGINE_SOURCES
    cpp/src/inference_engine.cpp
    cpp/src/kv_cache.cpp
    cpp/src/sampler.cpp
    cpp/src/tokenizer.cpp
)

# 推理引擎库
add_library(cockpit_engine ${ENGINE_SOURCES})
target_link_libraries(cockpit_engine 
    PUBLIC 
        Threads::Threads
        nlohmann_json::nlohmann_json
)

if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    target_link_libraries(cockpit_engine PUBLIC llama ggml)
endif()

# 可执行文件 - 命令行测试
add_executable(cockpit_cli cpp/src/main.cpp)
target_link_libraries(cockpit_cli PRIVATE cockpit_engine)

# Python绑定
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    
    pybind11_add_module(cockpit_engine_py cpp/bindings/pybind_engine.cpp)
    target_link_libraries(cockpit_engine_py PRIVATE cockpit_engine)
    
    # 设置输出路径
    set_target_properties(cockpit_engine_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
    )
endif()

# 测试
if(BUILD_TESTS)
    enable_testing()
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    
    add_executable(test_engine tests/test_engine.cpp)
    target_link_libraries(test_engine PRIVATE cockpit_engine GTest::gtest_main)
    
    include(GoogleTest)
    gtest_discover_tests(test_engine)
endif()

# 安装
install(TARGETS cockpit_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY cpp/include/ DESTINATION include/cockpit)

# 打印配置信息
message(STATUS "====================================")
message(STATUS "Cockpit Assistant Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA support: ${GGML_CUDA}")
message(STATUS "  Metal support: ${GGML_METAL}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "====================================")
