#pragma once

#include <cstdint>
#include <string>

namespace cuvs_silicon {

struct HostHardwareInfo {
  bool is_apple_platform = false;
  bool is_arm64 = false;
  bool has_metal_gpu = false;
  std::string cpu_brand;
};

struct HardwareSupport {
  bool supported = false;
  std::string platform_name;
  std::string reason;
};

struct HardwareMetadata {
  std::string chip_model;
  int gpu_core_count = 0;
  std::uint64_t unified_memory_bytes = 0;
};

[[nodiscard]] HardwareSupport classify_hardware(
    const HostHardwareInfo& hardware);

[[nodiscard]] HardwareMetadata collect_hardware_metadata(
    const std::string& sysctl_output,
    const std::string& system_profiler_output);

[[nodiscard]] std::string parse_apple_silicon_model(
    const std::string& sysctl_output,
    const std::string& system_profiler_output);

[[nodiscard]] int parse_apple_gpu_core_count(
    const std::string& system_profiler_output);

[[nodiscard]] std::uint64_t parse_unified_memory_bytes(
    const std::string& sysctl_output,
    const std::string& system_profiler_output);

[[nodiscard]] HardwareSupport detect_hardware_support();

}  // namespace cuvs_silicon
