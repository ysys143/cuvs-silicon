cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED METAL_CAGRA_SOURCE_DIR)
  message(FATAL_ERROR "METAL_CAGRA_SOURCE_DIR is required")
endif()

if(NOT DEFINED METAL_CAGRA_BINARY_DIR)
  message(FATAL_ERROR "METAL_CAGRA_BINARY_DIR is required")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "CMAKE_CXX_COMPILER is required")
endif()

set(test_root "${METAL_CAGRA_BINARY_DIR}/faiss_v1_8_cuvs_header_compile")
set(archive_path "${test_root}/faiss-v1.8.0.tar.gz")
set(faiss_source_dir "${test_root}/faiss-1.8.0")
set(probe_source "${METAL_CAGRA_SOURCE_DIR}/tests/faiss_v1_8_cuvs_header_probe.cpp")
set(probe_object "${test_root}/faiss_v1_8_cuvs_header_probe.o")

file(MAKE_DIRECTORY "${test_root}")

if(NOT EXISTS "${archive_path}")
  file(
    DOWNLOAD
      "https://github.com/facebookresearch/faiss/archive/refs/tags/v1.8.0.tar.gz"
      "${archive_path}"
    EXPECTED_HASH
      SHA256=56ece0a419d62eaa11e39022fa27c8ed6d5a9b9eb7416cc5a0fdbeab07ec2f0c
    STATUS download_status
    SHOW_PROGRESS
  )
  list(GET download_status 0 download_code)
  list(GET download_status 1 download_message)
  if(NOT download_code EQUAL 0)
    message(FATAL_ERROR "Failed to download FAISS v1.8.0 headers: ${download_message}")
  endif()
endif()

if(NOT EXISTS "${faiss_source_dir}/faiss/Index.h")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar xzf "${archive_path}"
    WORKING_DIRECTORY "${test_root}"
    RESULT_VARIABLE extract_result
    OUTPUT_VARIABLE extract_stdout
    ERROR_VARIABLE extract_stderr
  )
  if(NOT extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract FAISS v1.8.0 archive: ${extract_stderr}")
  endif()
endif()

execute_process(
  COMMAND
    "${CMAKE_CXX_COMPILER}"
    -std=c++17
    -I
    "${METAL_CAGRA_SOURCE_DIR}/include"
    -I
    "${faiss_source_dir}"
    -c
    "${probe_source}"
    -o
    "${probe_object}"
  RESULT_VARIABLE compile_result
  OUTPUT_VARIABLE compile_stdout
  ERROR_VARIABLE compile_stderr
)

if(NOT compile_result EQUAL 0)
  message(FATAL_ERROR
    "FAISS v1.8.0 + metal-cagra cuVS CAGRA header compile probe failed:\n"
    "${compile_stdout}\n${compile_stderr}"
  )
endif()
