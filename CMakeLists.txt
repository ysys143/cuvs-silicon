cmake_minimum_required(VERSION 3.20)

project(cuvs_silicon LANGUAGES CXX OBJCXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# ── Metal framework ────────────────────────────────────────────────────────
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation REQUIRED)

# ── Metal shader compilation: .metal → .air → .metallib ───────────────────
# Seed constraint: build MUST fail if no .metal files exist.
set(METAL_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/cagra_kernels.metal)
set(METAL_AIR        ${CMAKE_CURRENT_BINARY_DIR}/cagra_kernels.air)
set(METAL_LIB        ${CMAKE_CURRENT_BINARY_DIR}/cagra_kernels.metallib)

add_custom_command(
    OUTPUT  ${METAL_AIR}
    COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SRC} -o ${METAL_AIR}
    DEPENDS ${METAL_SHADER_SRC}
    COMMENT "Compiling Metal shader: cagra_kernels.metal"
)
add_custom_command(
    OUTPUT  ${METAL_LIB}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
    DEPENDS ${METAL_AIR}
    COMMENT "Linking Metal library: cagra_kernels.metallib"
)
add_custom_target(cagra_metallib ALL DEPENDS ${METAL_LIB})

# ── Metal context library (ObjC++, links Metal.framework) ─────────────────
add_library(cuvs_silicon_context
    src/metal_context.mm
    src/metal_search.mm
)
target_include_directories(cuvs_silicon_context
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

target_link_libraries(cuvs_silicon_context
    PUBLIC
        ${METAL_FRAMEWORK}
        ${CORE_FOUNDATION_FRAMEWORK}
        ${MPS_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
)
target_compile_options(cuvs_silicon_context PRIVATE -fobjc-arc)
target_compile_definitions(cuvs_silicon_context
    PRIVATE CAGRA_METALLIB_PATH="${METAL_LIB}"
)
add_dependencies(cuvs_silicon_context cagra_metallib)

add_library(cuvs_silicon
    src/cagra.cpp
    src/distance.cpp
)
target_include_directories(cuvs_silicon
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(cuvs_silicon_sift_loader
    src/sift_loader.cpp
)
target_include_directories(cuvs_silicon_sift_loader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(cuvs_silicon_hardware
    src/hardware.cpp
)
target_include_directories(cuvs_silicon_hardware
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()

add_executable(test_sift_loader
    tests/test_sift_loader.cpp
)
target_link_libraries(test_sift_loader
    PRIVATE
        cuvs_silicon_sift_loader
)

add_test(NAME sift_loader COMMAND test_sift_loader)

add_executable(test_cpu_hnsw_baseline
    tests/test_cpu_hnsw_baseline.cpp
)
target_include_directories(test_cpu_hnsw_baseline
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_test(NAME cpu_hnsw_baseline COMMAND test_cpu_hnsw_baseline)

add_executable(test_l2_distance
    tests/test_l2_distance.cpp
)
target_link_libraries(test_l2_distance
    PRIVATE
        cuvs_silicon
)

add_test(NAME l2_distance COMMAND test_l2_distance)

add_executable(test_hardware_detection
    tests/test_hardware_detection.cpp
)
target_link_libraries(test_hardware_detection
    PRIVATE
        cuvs_silicon_hardware
)

add_test(NAME hardware_detection COMMAND test_hardware_detection)

add_test(
    NAME benchmark_batching
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p test_benchmark_batching.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
    NAME faiss_v1_8_cuvs_headers_compile
    COMMAND
        ${CMAKE_COMMAND}
        -DMETAL_CAGRA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DMETAL_CAGRA_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/run_faiss_v1_8_cuvs_header_compile.cmake
)

add_test(
    NAME faiss_v1_8_cuvs_link
    COMMAND
        ${CMAKE_COMMAND}
        -DMETAL_CAGRA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DMETAL_CAGRA_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -DMETAL_CAGRA_LIBRARY=$<TARGET_FILE:cuvs_silicon>
        -DMETAL_CAGRA_CONTEXT_LIBRARY=$<TARGET_FILE:cuvs_silicon_context>
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/run_faiss_v1_8_cuvs_link.cmake
)

add_test(
    NAME faiss_v1_8_cagra_runtime_smoke
    COMMAND
        ${CMAKE_COMMAND}
        -DMETAL_CAGRA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DMETAL_CAGRA_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -DMETAL_CAGRA_LIBRARY=$<TARGET_FILE:cuvs_silicon>
        -DMETAL_CAGRA_CONTEXT_LIBRARY=$<TARGET_FILE:cuvs_silicon_context>
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/run_faiss_v1_8_cagra_runtime_smoke.cmake
)

# Cohere Wikipedia validation binary (not a ctest — run manually with data files)
add_executable(cohere_wiki_validation
    tests/test_cohere_wiki_validation.cpp
)
target_link_libraries(cohere_wiki_validation
    PRIVATE
        cuvs_silicon
        cuvs_silicon_context
)

# TDD Red → Green: Metal GPU dispatch gate.
add_executable(test_metal_search
    tests/test_metal_search.cpp
)
target_link_libraries(test_metal_search
    PRIVATE
        cuvs_silicon
        cuvs_silicon_context
)
add_test(NAME metal_search_dispatches_gpu COMMAND test_metal_search)
