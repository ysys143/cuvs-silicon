#include "cuvs_silicon/hardware.hpp"

#include <cassert>
#include <string>

namespace {

void parses_exact_chip_model_from_mocked_sysctl_brand_string() {
  const auto chip_model = cuvs_silicon::parse_apple_silicon_model(
      "machdep.cpu.brand_string: Apple M2 Pro\n", "");

  assert(chip_model == "Apple M2 Pro");
}

void parses_exact_chip_model_from_mocked_system_profiler_output() {
  const auto chip_model = cuvs_silicon::parse_apple_silicon_model(
      "",
      "Hardware:\n"
      "\n"
      "    Hardware Overview:\n"
      "\n"
      "      Model Name: MacBook Pro\n"
      "      Chip: Apple M3 Max\n"
      "      Total Number of Cores: 14\n");

  assert(chip_model == "Apple M3 Max");
}

void prefers_sysctl_chip_model_when_both_mocked_sources_exist() {
  const auto chip_model = cuvs_silicon::parse_apple_silicon_model(
      "Apple M4 Ultra", "Chip: Apple M3 Max\n");

  assert(chip_model == "Apple M4 Ultra");
}

void rejects_mocked_non_m_series_chip_text() {
  const auto chip_model = cuvs_silicon::parse_apple_silicon_model(
      "machdep.cpu.brand_string: Intel(R) Core(TM) i9\n",
      "Chip: Apple A14\n");

  assert(chip_model.empty());
}

void parses_gpu_core_count_from_spdisplays_text_block() {
  const auto core_count = cuvs_silicon::parse_apple_gpu_core_count(
      "Graphics/Displays:\n"
      "\n"
      "    Apple M3 Max:\n"
      "\n"
      "      Chipset Model: Apple M3 Max\n"
      "      Type: GPU\n"
      "      Bus: Built-In\n"
      "      Total Number of Cores: 40\n"
      "      Vendor: Apple (0x106b)\n"
      "      Metal Support: Metal 3\n");

  assert(core_count == 40);
}

void parses_gpu_core_count_from_explicit_gpu_cores_field() {
  const auto core_count = cuvs_silicon::parse_apple_gpu_core_count(
      "Hardware:\n"
      "\n"
      "    Hardware Overview:\n"
      "\n"
      "      Chip: Apple M4\n"
      "      Total Number of Cores: 10\n"
      "      GPU Cores: 10\n");

  assert(core_count == 10);
}

void parses_gpu_core_count_from_system_profiler_key_field() {
  const auto core_count = cuvs_silicon::parse_apple_gpu_core_count(
      "    spdisplays_gpucorecount: 76\n"
      "    spdisplays_metal: spdisplays_metal3\n");

  assert(core_count == 76);
}

void rejects_cpu_total_core_count_without_gpu_context() {
  const auto core_count = cuvs_silicon::parse_apple_gpu_core_count(
      "Hardware:\n"
      "\n"
      "    Hardware Overview:\n"
      "\n"
      "      Chip: Apple M3 Max\n"
      "      Total Number of Cores: 16\n"
      "      Memory: 128 GB\n");

  assert(core_count == 0);
}

void parses_unified_memory_bytes_from_mocked_sysctl_memsize() {
  const auto memory_bytes = cuvs_silicon::parse_unified_memory_bytes(
      "hw.memsize: 34359738368\n", "");

  assert(memory_bytes == 34359738368ULL);
}

void parses_unified_memory_gb_from_mocked_system_profiler() {
  const auto memory_bytes = cuvs_silicon::parse_unified_memory_bytes(
      "",
      "Hardware:\n"
      "\n"
      "    Hardware Overview:\n"
      "\n"
      "      Chip: Apple M3 Max\n"
      "      Memory: 128 GB\n");

  assert(memory_bytes == 128ULL * 1024ULL * 1024ULL * 1024ULL);
}

void parses_decimal_unified_memory_human_readable_format() {
  const auto memory_bytes = cuvs_silicon::parse_unified_memory_bytes(
      "",
      "Hardware:\n"
      "    Hardware Overview:\n"
      "      Unified Memory: 1.5 TB\n");

  assert(memory_bytes == 1536ULL * 1024ULL * 1024ULL * 1024ULL);
}

void prefers_sysctl_unified_memory_bytes_over_system_profiler_text() {
  const auto memory_bytes = cuvs_silicon::parse_unified_memory_bytes(
      "hw.memsize: 17179869184\n",
      "Hardware:\n"
      "    Hardware Overview:\n"
      "      Memory: 32 GB\n");

  assert(memory_bytes == 17179869184ULL);
}

void rejects_unparseable_unified_memory_inputs() {
  const auto memory_bytes = cuvs_silicon::parse_unified_memory_bytes(
      "hw.memsize: unknown\n",
      "Hardware:\n"
      "    Hardware Overview:\n"
      "      Memory: unknown\n");

  assert(memory_bytes == 0);
}

void aggregates_hardware_metadata_from_mocked_command_outputs() {
  const auto metadata = cuvs_silicon::collect_hardware_metadata(
      "machdep.cpu.brand_string: Apple M3 Max\n"
      "hw.memsize: 68719476736\n",
      "Hardware:\n"
      "\n"
      "    Hardware Overview:\n"
      "\n"
      "      Chip: Apple M3 Max\n"
      "      Memory: 128 GB\n"
      "\n"
      "Graphics/Displays:\n"
      "\n"
      "    Apple M3 Max:\n"
      "\n"
      "      Chipset Model: Apple M3 Max\n"
      "      Type: GPU\n"
      "      Total Number of Cores: 40\n"
      "      Metal Support: Metal 3\n");

  assert(metadata.chip_model == "Apple M3 Max");
  assert(metadata.gpu_core_count == 40);
  assert(metadata.unified_memory_bytes == 68719476736ULL);
}

void aggregates_partial_metadata_when_mocked_outputs_are_incomplete() {
  const auto metadata = cuvs_silicon::collect_hardware_metadata(
      "machdep.cpu.brand_string: Apple M2\n",
      "Hardware:\n"
      "    Hardware Overview:\n"
      "      Memory: unknown\n");

  assert(metadata.chip_model == "Apple M2");
  assert(metadata.gpu_core_count == 0);
  assert(metadata.unified_memory_bytes == 0);
}

void aggregates_populated_metadata_from_system_profiler_key_output() {
  const auto metadata = cuvs_silicon::collect_hardware_metadata(
      "",
      "    chip_type: Apple M3 Max\n"
      "    physical_memory: 36 GB\n"
      "    Chipset Model: Apple M3 Max\n"
      "    Type: GPU\n"
      "    GPU Core Count: 40\n");

  assert(metadata.chip_model == "Apple M3 Max");
  assert(metadata.gpu_core_count == 40);
  assert(metadata.unified_memory_bytes == 36ULL * 1024ULL * 1024ULL * 1024ULL);
}

void accepts_apple_m_series_arm64_with_metal() {
  const auto support = cuvs_silicon::classify_hardware({
      true,
      true,
      true,
      "Apple M1 Pro",
  });

  assert(support.supported);
  assert(support.platform_name == "Apple M1 Pro");
}

void accepts_apple_m1_identifier_as_minimum_supported_generation() {
  const auto support = cuvs_silicon::classify_hardware({
      true,
      true,
      true,
      "Apple M1",
  });

  assert(support.supported);
  assert(support.platform_name == "Apple M1");
}

void accepts_later_apple_m_series_generations() {
  const auto support = cuvs_silicon::classify_hardware({
      true,
      true,
      true,
      "Apple M3 Max",
  });

  assert(support.supported);
}

void rejects_pre_m1_apple_m_series_identifier() {
  const auto support = cuvs_silicon::classify_hardware({
      true,
      true,
      true,
      "Apple M0",
  });

  assert(!support.supported);
  assert(support.reason.find("M-series") != std::string::npos);
}

void rejects_intel_macos_hosts() {
  const auto support = cuvs_silicon::classify_hardware({
      true,
      false,
      true,
      "Intel(R) Core(TM) i9",
  });

  assert(!support.supported);
  assert(support.reason.find("arm64") != std::string::npos);
}

void rejects_non_apple_hosts() {
  const auto support = cuvs_silicon::classify_hardware({
      false,
      true,
      true,
      "Apple M2",
  });

  assert(!support.supported);
  assert(support.reason.find("macOS") != std::string::npos);
}

void rejects_hosts_without_metal_gpu() {
  const auto support = cuvs_silicon::classify_hardware({
      true,
      true,
      false,
      "Apple M2",
  });

  assert(!support.supported);
  assert(support.reason.find("Metal") != std::string::npos);
}

void rejects_unknown_apple_arm_cpus() {
  const auto support = cuvs_silicon::classify_hardware({
      true,
      true,
      true,
      "Apple A14",
  });

  assert(!support.supported);
  assert(support.reason.find("M-series") != std::string::npos);
}

}  // namespace

int main() {
  parses_exact_chip_model_from_mocked_sysctl_brand_string();
  parses_exact_chip_model_from_mocked_system_profiler_output();
  prefers_sysctl_chip_model_when_both_mocked_sources_exist();
  rejects_mocked_non_m_series_chip_text();
  parses_gpu_core_count_from_spdisplays_text_block();
  parses_gpu_core_count_from_explicit_gpu_cores_field();
  parses_gpu_core_count_from_system_profiler_key_field();
  rejects_cpu_total_core_count_without_gpu_context();
  parses_unified_memory_bytes_from_mocked_sysctl_memsize();
  parses_unified_memory_gb_from_mocked_system_profiler();
  parses_decimal_unified_memory_human_readable_format();
  prefers_sysctl_unified_memory_bytes_over_system_profiler_text();
  rejects_unparseable_unified_memory_inputs();
  aggregates_hardware_metadata_from_mocked_command_outputs();
  aggregates_partial_metadata_when_mocked_outputs_are_incomplete();
  aggregates_populated_metadata_from_system_profiler_key_output();
  accepts_apple_m_series_arm64_with_metal();
  accepts_apple_m1_identifier_as_minimum_supported_generation();
  accepts_later_apple_m_series_generations();
  rejects_pre_m1_apple_m_series_identifier();
  rejects_intel_macos_hosts();
  rejects_non_apple_hosts();
  rejects_hosts_without_metal_gpu();
  rejects_unknown_apple_arm_cpus();
  (void)cuvs_silicon::detect_hardware_support();
  return 0;
}
