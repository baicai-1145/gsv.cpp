cmake_minimum_required(VERSION 3.20)
project(gsv_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GSV_BUILD_TESTS "Build the service unit tests" ON)
enable_testing()

# ---------------------------------------------------------------- llama.cpp/ggml
# Two modes:
#   1) prebuilt libs: point GSV_GGML_LIB_DIR at a directory containing ggml*.lib/.so/.a
#   2) source tree: set GSV_GGML_LIB_DIR to "" and the llama.cpp tree is added as a subproject
set(GSV_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "llama.cpp source tree")
set(GSV_GGML_LIB_DIR "${GSV_LLAMA_DIR}/build-vk-rel" CACHE PATH "directory with prebuilt ggml libraries (empty = build llama.cpp as a subproject)")

if(NOT EXISTS "${GSV_LLAMA_DIR}/ggml/include/ggml.h")
    message(FATAL_ERROR "llama.cpp not found at '${GSV_LLAMA_DIR}'. Clone it and pass -DGSV_LLAMA_DIR=...")
endif()

set(GSV_GGML_INCLUDES
    "${GSV_LLAMA_DIR}/ggml/include"
    "${GSV_LLAMA_DIR}/ggml/src"
    "${GSV_LLAMA_DIR}/include"
    "${GSV_LLAMA_DIR}/src"
)

set(GSV_GGML_LIBS "")
if(GSV_GGML_LIB_DIR)
    file(GLOB_RECURSE _ggml_candidates
         "${GSV_GGML_LIB_DIR}/*.lib"
         "${GSV_GGML_LIB_DIR}/*.so"
         "${GSV_GGML_LIB_DIR}/*.so.*"
         "${GSV_GGML_LIB_DIR}/*.a")
    foreach(_name ggml-base ggml-cpu ggml-vulkan ggml)
        foreach(_cand ${_ggml_candidates})
            get_filename_component(_stem "${_cand}" NAME_WE)
            string(REGEX REPLACE "^lib" "" _stem "${_stem}")
            if(_stem STREQUAL _name)
                list(APPEND GSV_GGML_LIBS "${_cand}")
                break()
            endif()
        endforeach()
    endforeach()
    if(NOT GSV_GGML_LIBS)
        message(FATAL_ERROR "no ggml libraries under '${GSV_GGML_LIB_DIR}' (set GSV_GGML_LIB_DIR or clear it to build llama.cpp as a subproject)")
    endif()
else()
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${GSV_LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)
    set(GSV_GGML_LIBS ggml ggml-base ggml-cpu)
    if(GGML_VULKAN)
        list(APPEND GSV_GGML_LIBS ggml-vulkan)
    endif()
endif()

# ---------------------------------------------------------------- engine
file(GLOB GSV_ENGINE_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_ar.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_bert.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_cond.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_dhubert.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_dit.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_encp.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_engine.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_hubert.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_kernels.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_mel.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_refcode.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_refenc.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_sampler.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_v2sovits.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_vocoder_ggml.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/gsv_wns1.cpp")

add_library(gsv_engine STATIC ${GSV_ENGINE_SOURCES})
target_include_directories(gsv_engine
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxhash"
    PUBLIC
        ${GSV_GGML_INCLUDES})
target_link_libraries(gsv_engine PUBLIC ${GSV_GGML_LIBS})
if(MSVC)
    target_compile_options(gsv_engine PUBLIC /utf-8 /W1)
    target_compile_options(gsv_engine PRIVATE /openmp)
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(gsv_engine PRIVATE -mavx2 -mfma)
    endif()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gsv_engine PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# ---------------------------------------------------------------- server
add_executable(gsv_server "${CMAKE_CURRENT_SOURCE_DIR}/tools/gsv_server.cpp")
target_link_libraries(gsv_server PRIVATE gsv_engine)
if(WIN32)
    target_link_libraries(gsv_server PRIVATE ws2_32)
    # ggml is usually built as a DLL: copy the DLLs next to the executable
    if(GSV_GGML_LIB_DIR)
        file(GLOB_RECURSE GSV_GGML_DLLS "${GSV_GGML_LIB_DIR}/*.dll")
        if(GSV_GGML_DLLS)
            add_custom_command(TARGET gsv_server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GSV_GGML_DLLS} "$<TARGET_FILE_DIR:gsv_server>")
        endif()
    endif()
endif()

# ---------------------------------------------------------------- tests
if(GSV_BUILD_TESTS)
    # distilled HuBERT numerical probe: dhubert_probe <gguf> <wav> [out.f32] [--ref ...]
    add_executable(dhubert_probe "${CMAKE_CURRENT_SOURCE_DIR}/tools/dhubert_probe.cpp")
    target_link_libraries(dhubert_probe PRIVATE gsv_engine)

    add_executable(test_svc "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_svc.cpp")
    target_include_directories(test_svc PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxhash"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann")
    target_link_libraries(test_svc PRIVATE gsv_engine)
    add_test(NAME test_svc COMMAND test_svc)

    # reference chain must be repeatable: test_prep_repeat <models_dir> <ref.wav> [other.wav]
    add_executable(test_prep_repeat "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_prep_repeat.cpp")
    target_link_libraries(test_prep_repeat PRIVATE gsv_engine)

    # v2 / v2Pro / v2ProPlus S2 解码器: 分段对拍 + 端到端 benchmark
    add_executable(probe_v2sovits "${CMAKE_CURRENT_SOURCE_DIR}/tests/probe_v2sovits.cpp")
    target_link_libraries(probe_v2sovits PRIVATE gsv_engine)

    # add_act kernel (CPU + Vulkan) incl. the mish activation used by the DiT trunk
    add_executable(test_add_act "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_add_act.cpp")
    target_include_directories(test_add_act PRIVATE ${GSV_GGML_INCLUDES})
    target_link_libraries(test_add_act PRIVATE ${GSV_GGML_LIBS})
    add_test(NAME test_add_act COMMAND test_add_act --cpu)

    add_executable(test_conv_direct_act "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conv_direct_act.cpp")
    target_include_directories(test_conv_direct_act PRIVATE ${GSV_GGML_INCLUDES})
    target_link_libraries(test_conv_direct_act PRIVATE ${GSV_GGML_LIBS})
    add_test(NAME test_conv_direct_act COMMAND test_conv_direct_act --cpu)

    # dec ResBlock1 手写 AVX2 内核的正确性自检（对参考实现 + 线程数不变性，不计时）
    add_executable(test_v2_kernels "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v2_kernels.cpp")
    target_link_libraries(test_v2_kernels PRIVATE gsv_engine)
    add_test(NAME test_v2_kernels COMMAND test_v2_kernels)

    # 直接卷积 vs im2col+ggml_mul_mat 的 A/B（dec 主算子选型依据）
    add_executable(bench_conv1d "${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_conv1d.cpp")
    target_include_directories(bench_conv1d PRIVATE ${GSV_GGML_INCLUDES})
    target_link_libraries(bench_conv1d PRIVATE ${GSV_GGML_LIBS})
endif()
