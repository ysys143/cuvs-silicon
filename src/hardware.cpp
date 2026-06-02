#include "cuvs_silicon/hardware.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <sstream>
#include <regex>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace cuvs_silicon {
namespace {

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

bool contains_apple_m_series(const std::string& cpu_brand) {
  return !parse_apple_silicon_model(cpu_brand, {}).empty();
}

bool plausible_gpu_core_count(int core_count) {
  return core_count > 0 && core_count <= 256;
}

std::uint64_t parse_u64(const std::string& value) {
  try {
    std::size_t parsed = 0;
    const auto parsed_value = std::stoull(value, &parsed, 10);
    return parsed == value.size() ? parsed_value : 0;
  } catch (...) {
    return 0;
  }
}

std::uint64_t memory_unit_multiplier(const std::string& unit_text) {
  const auto unit = lower_copy(unit_text);
  if (unit == "b" || unit == "byte" || unit == "bytes") {
    return 1;
  }
  if (unit == "kb") {
    return 1024ULL;
  }
  if (unit == "mb") {
    return 1024ULL * 1024ULL;
  }
  if (unit == "gb") {
    return 1024ULL * 1024ULL * 1024ULL;
  }
  if (unit == "tb") {
    return 1024ULL * 1024ULL * 1024ULL * 1024ULL;
  }
  return 0;
}

std::uint64_t parse_memory_quantity(const std::string& value_text,
                                    const std::string& unit_text) {
  const auto unit_multiplier = memory_unit_multiplier(unit_text);
  if (unit_multiplier == 0) {
    return 0;
  }

  try {
    std::size_t parsed = 0;
    const auto quantity = std::stod(value_text, &parsed);
    if (parsed != value_text.size() || quantity <= 0.0) {
      return 0;
    }
    const auto bytes = quantity * static_cast<double>(unit_multiplier);
    if (bytes > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
      return 0;
    }
    return static_cast<std::uint64_t>(bytes);
  } catch (...) {
    return 0;
  }
}

bool has_gpu_context(const std::deque<std::string>& recent_lines) {
  for (const auto& line : recent_lines) {
    const auto lowered = lower_copy(line);
    if (lowered.find("type: gpu") != std::string::npos ||
        lowered.find("graphics/displays") != std::string::npos ||
        lowered.find("chipset model: apple m") != std::string::npos ||
        lowered.find("metal support:") != std::string::npos) {
      return true;
    }
  }
  return false;
}

#if defined(__APPLE__)
std::string sysctl_string(const char* name) {
  std::size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
    return {};
  }

  std::string value(size, '\0');
  if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0 || size == 0) {
    return {};
  }

  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}
#endif

}  // namespace

HardwareSupport classify_hardware(const HostHardwareInfo& hardware) {
  const auto chip_model = parse_apple_silicon_model(hardware.cpu_brand, {});
  if (!hardware.is_apple_platform) {
    return {false, hardware.cpu_brand, "host is not macOS on Apple hardware"};
  }
  if (!hardware.is_arm64) {
    return {false, hardware.cpu_brand, "host CPU is not Apple Silicon arm64"};
  }
  if (!hardware.has_metal_gpu) {
    return {false, hardware.cpu_brand, "host does not report a Metal-capable GPU"};
  }
  if (!contains_apple_m_series(hardware.cpu_brand)) {
    return {false, hardware.cpu_brand, "host CPU is not an Apple M-series M1 or later"};
  }

  return {true, chip_model, "Apple Silicon M-series with Metal GPU"};
}

HardwareMetadata collect_hardware_metadata(
    const std::string& sysctl_output,
    const std::string& system_profiler_output) {
  return {
      parse_apple_silicon_model(sysctl_output, system_profiler_output),
      parse_apple_gpu_core_count(system_profiler_output),
      parse_unified_memory_bytes(sysctl_output, system_profiler_output),
  };
}

std::string parse_apple_silicon_model(
    const std::string& sysctl_output,
    const std::string& system_profiler_output) {
  const std::regex chip_pattern(
      R"(Apple[[:space:]]+M[1-9][0-9]*([[:space:]]+(Pro|Max|Ultra))?)",
      std::regex::icase);

  for (const auto* output : {&sysctl_output, &system_profiler_output}) {
    std::smatch match;
    if (std::regex_search(*output, match, chip_pattern)) {
      auto model = match.str(0);
      std::replace_if(model.begin(), model.end(),
                      [](unsigned char ch) { return std::isspace(ch) != 0; },
                      ' ');
      return model;
    }
  }

  return {};
}

int parse_apple_gpu_core_count(const std::string& system_profiler_output) {
  const std::regex explicit_gpu_core_pattern(
      R"((GPU[[:space:]_-]*Cores|GPU[[:space:]_-]*Core[[:space:]_-]*Count|spdisplays_gpucorecount)[^0-9]*([0-9]+))",
      std::regex::icase);

  std::smatch match;
  if (std::regex_search(system_profiler_output, match,
                        explicit_gpu_core_pattern)) {
    const auto core_count = std::atoi(match.str(2).c_str());
    if (plausible_gpu_core_count(core_count)) {
      return core_count;
    }
  }

  const std::regex total_core_pattern(
      R"(Total[[:space:]]+Number[[:space:]]+of[[:space:]]+Cores[^0-9]*([0-9]+))",
      std::regex::icase);

  std::istringstream lines(system_profiler_output);
  std::string line;
  std::deque<std::string> recent_lines;
  while (std::getline(lines, line)) {
    if (std::regex_search(line, match, total_core_pattern) &&
        has_gpu_context(recent_lines)) {
      const auto core_count = std::atoi(match.str(1).c_str());
      if (plausible_gpu_core_count(core_count)) {
        return core_count;
      }
    }

    recent_lines.push_back(line);
    if (recent_lines.size() > 8) {
      recent_lines.pop_front();
    }
  }

  return 0;
}

std::uint64_t parse_unified_memory_bytes(
    const std::string& sysctl_output,
    const std::string& system_profiler_output) {
  const std::regex sysctl_bytes_pattern(
      R"((hw\.memsize|memsize)[^0-9]*([0-9]+))",
      std::regex::icase);

  std::smatch match;
  if (std::regex_search(sysctl_output, match, sysctl_bytes_pattern)) {
    const auto bytes = parse_u64(match.str(2));
    if (bytes > 0) {
      return bytes;
    }
  }

  const std::regex profiler_memory_pattern(
      R"((Unified[[:space:]]+Memory|Memory)[^0-9]*([0-9]+(?:\.[0-9]+)?)[[:space:]]*(bytes?|b|kb|mb|gb|tb))",
      std::regex::icase);
  if (std::regex_search(system_profiler_output, match,
                        profiler_memory_pattern)) {
    return parse_memory_quantity(match.str(2), match.str(3));
  }

  return 0;
}

HardwareSupport detect_hardware_support() {
  HostHardwareInfo hardware;

#if defined(__APPLE__)
  hardware.is_apple_platform = true;
  hardware.cpu_brand = sysctl_string("machdep.cpu.brand_string");
#if defined(__aarch64__) || defined(__arm64__)
  hardware.is_arm64 = true;
#endif
  hardware.has_metal_gpu = hardware.is_arm64;
#else
  hardware.cpu_brand = "unsupported non-Apple host";
#endif

  return classify_hardware(hardware);
}

}  // namespace cuvs_silicon
