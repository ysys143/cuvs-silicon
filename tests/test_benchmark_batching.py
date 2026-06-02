import json
import struct
import tempfile
from pathlib import Path
import unittest

import cuvs_silicon.benchmark as benchmark_module
from cuvs_silicon.benchmark import (
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_MEASURED_ITERATIONS,
    BENCHMARK_VECTOR_DIMENSION,
    BENCHMARK_WARMUP_ITERATIONS,
    BenchmarkSample,
    REQUIRED_BENCHMARK_IDENTITY_FIELDS,
    SIFT1M_DATASET_ID,
    UnsupportedHardwareError,
    benchmark_identity_metadata,
    cpu_hnsw_baseline_identity_metadata,
    detect_xcode_version,
    format_benchmark_report,
    generate_sift1m_benchmark_configs,
    load_sift1m_dataset_selection,
    load_sift_groundtruth_ivecs,
    require_supported_benchmark_hardware,
    report_benchmark_metrics,
    run_benchmark_iterations,
    run_benchmark_queries,
    validate_benchmark_metadata_identity,
    validate_benchmark_vectors,
)


class DtypedVectors(list[list[float]]):
    def __init__(self, rows: list[list[float]], dtype: str) -> None:
        super().__init__(rows)
        self.dtype = dtype


class DetectionResult:
    def __init__(self, supported: bool, platform_name: str, reason: str) -> None:
        self.supported = supported
        self.platform_name = platform_name
        self.reason = reason


class BenchmarkBatchingTests(unittest.TestCase):
    def test_parse_macos_version_returns_comparable_numeric_components(self) -> None:
        parsed = benchmark_module.parse_macos_version("14.5.1")

        self.assertEqual(parsed.major, 14)
        self.assertEqual(parsed.minor, 5)
        self.assertEqual(parsed.patch, 1)
        self.assertGreater(parsed, benchmark_module.parse_macos_version("14.4.9"))

    def test_detect_macos_version_uses_current_platform_version_provider(self) -> None:
        parsed = benchmark_module.detect_macos_version(
            lambda: ("15.0.2", ("", "", ""), "")
        )

        self.assertEqual((parsed.major, parsed.minor, parsed.patch), (15, 0, 2))

    def test_benchmark_metadata_identity_accepts_matching_result_artifacts(self) -> None:
        artifacts = _benchmark_artifacts_with_identity(
            os_release="23.5.0",
            machine="arm64",
            processor="arm",
        )

        validate_benchmark_metadata_identity(artifacts)

    def test_benchmark_metadata_identity_rejects_mismatching_result_artifacts(self) -> None:
        artifacts = _benchmark_artifacts_with_identity(
            os_release="23.5.0",
            machine="arm64",
            processor="arm",
        )
        artifacts["faiss_mlx"]["identity_metadata"]["machine"] = "x86_64"

        with self.assertRaisesRegex(
            ValueError,
            "identity mismatch between cpu_hnsw and faiss_mlx: machine",
        ):
            validate_benchmark_metadata_identity(artifacts)

    def test_cpu_hnsw_baseline_records_required_identity_metadata(self) -> None:
        metadata = cpu_hnsw_baseline_identity_metadata().as_dict()

        self.assertEqual(set(metadata), set(REQUIRED_BENCHMARK_IDENTITY_FIELDS))
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["run_name"], "cpu_hnsw_baseline")
        self.assertEqual(metadata["backend"], "cpu_hnsw")
        self.assertEqual(metadata["implementation"], "hnswlib-single-threaded")
        self.assertIsInstance(metadata["macos_version"], str)
        self.assertIsInstance(metadata["xcode_version"], str)
        for field in REQUIRED_BENCHMARK_IDENTITY_FIELDS:
            self.assertIsNotNone(metadata[field])

    def test_benchmark_identity_metadata_reports_xcode_version(self) -> None:
        metadata = benchmark_identity_metadata(
            run_name="metal",
            backend="metal",
            implementation="cuvs-silicon",
            xcode_version="15.4",
        ).as_dict()

        self.assertEqual(metadata["xcode_version"], "15.4")

    def test_benchmark_identity_metadata_serializes_apple_silicon_hardware(self) -> None:
        metadata = benchmark_identity_metadata(
            run_name="metal",
            backend="metal",
            implementation="cuvs-silicon",
            xcode_version="15.4",
            chip_model="Apple M3 Max",
            gpu_core_count=40,
            unified_memory_bytes=137438953472,
        ).as_dict()

        self.assertEqual(metadata["chip_model"], "Apple M3 Max")
        self.assertEqual(metadata["gpu_core_count"], 40)
        self.assertEqual(metadata["unified_memory_bytes"], 137438953472)

    def test_detect_xcode_version_parses_xcodebuild_output(self) -> None:
        def run_command(*args: object, **kwargs: object) -> object:
            return _CompletedCommand(
                returncode=0,
                stdout="Xcode 15.4\nBuild version 15F31d\n",
            )

        self.assertEqual(detect_xcode_version(run_command), "15.4")

    def test_detect_xcode_version_finds_xcode_line_after_leading_output(self) -> None:
        def run_command(*args: object, **kwargs: object) -> object:
            return _CompletedCommand(
                returncode=0,
                stdout="\nBuild version 15F31d\nXcode 15.4\n",
            )

        self.assertEqual(detect_xcode_version(run_command), "15.4")

    def test_preflight_rejects_unsupported_hardware_detection_result(self) -> None:
        detection = DetectionResult(
            supported=False,
            platform_name="Intel(R) Core(TM) i9",
            reason="host CPU is not Apple Silicon arm64",
        )

        with self.assertRaisesRegex(
            UnsupportedHardwareError,
            "unsupported benchmark hardware: Intel.*arm64",
        ):
            require_supported_benchmark_hardware(detection)

    def test_preflight_accepts_supported_hardware_detection_result(self) -> None:
        detection = DetectionResult(
            supported=True,
            platform_name="Apple M3 Max",
            reason="Apple Silicon M-series with Metal GPU",
        )

        require_supported_benchmark_hardware(detection, macos_version="14.5")

    def test_preflight_rejects_macos_major_version_earlier_than_14(self) -> None:
        detection = DetectionResult(
            supported=True,
            platform_name="Apple M3 Max",
            reason="Apple Silicon M-series with Metal GPU",
        )

        with self.assertRaisesRegex(
            UnsupportedHardwareError,
            r"macOS 14\.0\+ is required; detected 13.6.7",
        ):
            require_supported_benchmark_hardware(detection, macos_version="13.6.7")

    def test_benchmark_macos_validation_rejects_parsed_versions_before_14_0(
        self,
    ) -> None:
        parsed = benchmark_module.parse_macos_version("13.9.9")

        with self.assertRaisesRegex(
            UnsupportedHardwareError,
            r"macOS 14\.0\+ is required; detected 13.9.9",
        ):
            benchmark_module.validate_benchmark_macos_version(parsed)

    def test_benchmark_macos_validation_accepts_14_0_and_newer(self) -> None:
        for version in ("14", "14.0", "14.0.1", "15.0"):
            with self.subTest(version=version):
                parsed = benchmark_module.parse_macos_version(version)
                benchmark_module.validate_benchmark_macos_version(parsed)

    def test_sift1m_benchmark_configs_use_l2_distance(self) -> None:
        configs = generate_sift1m_benchmark_configs(
            ["cuvs-silicon", "cpu-hnsw", "faiss-mlx"]
        )

        self.assertEqual([config.distance_metric for config in configs], ["l2"] * 3)

    def test_loads_sift1m_dataset_selection_from_benchmark_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "sift_base.fvecs"
            query_path = root / "sift_query.fvecs"
            groundtruth_path = root / "sift_groundtruth.ivecs"
            for path in (base_path, query_path, groundtruth_path):
                path.touch()

            selection = load_sift1m_dataset_selection(root)

        self.assertEqual(selection.dataset_id, SIFT1M_DATASET_ID)
        self.assertEqual(selection.base_path, base_path)
        self.assertEqual(selection.query_path, query_path)
        self.assertEqual(selection.groundtruth_path, groundtruth_path)

    def test_sift1m_benchmark_configs_reject_non_l2_distance_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "distance_metric must be l2"):
            generate_sift1m_benchmark_configs(["cuvs-silicon"], distance_metric="cosine")

    def test_sift1m_benchmark_distance_evaluation_applies_l2(self) -> None:
        config = generate_sift1m_benchmark_configs(["cuvs-silicon"])[0]
        query = [0.0] * BENCHMARK_VECTOR_DIMENSION
        candidates = [
            [0.0] * BENCHMARK_VECTOR_DIMENSION,
            [0.0] * BENCHMARK_VECTOR_DIMENSION,
            [0.0] * BENCHMARK_VECTOR_DIMENSION,
        ]
        query[0] = 1.0
        candidates[0][0] = 1.0
        candidates[1][0] = 2.0
        candidates[2][1] = 1.0

        distances = benchmark_module.evaluate_sift1m_search_distances(
            query,
            candidates,
            config,
        )

        self.assertEqual(distances, [0.0, 1.0, 2.0])

    def test_loads_sift_groundtruth_ivecs_as_integer_neighbor_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groundtruth.ivecs"
            _write_ivecs(path, [[11, 12, 13], [21, 22, 23]])

            neighbors = load_sift_groundtruth_ivecs(path, expected_row_width=3)

        self.assertEqual(neighbors, [[11, 12, 13], [21, 22, 23]])
        self.assertTrue(all(isinstance(value, int) for row in neighbors for value in row))

    def test_load_sift_groundtruth_ivecs_rejects_unexpected_row_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groundtruth.ivecs"
            _write_ivecs(path, [[11, 12, 13]])

            with self.assertRaisesRegex(ValueError, "row width 100"):
                load_sift_groundtruth_ivecs(path, expected_row_width=100)

    def test_accepts_128_dimensional_benchmark_vectors(self) -> None:
        vectors = [[0.0] * BENCHMARK_VECTOR_DIMENSION for _ in range(2)]

        validate_benchmark_vectors(vectors)

    def test_rejects_benchmark_vectors_whose_dimensionality_is_not_128(self) -> None:
        vectors = [[0.0] * (BENCHMARK_VECTOR_DIMENSION - 1)]

        with self.assertRaisesRegex(ValueError, "dimensionality 128"):
            validate_benchmark_vectors(vectors)

    def test_rejects_benchmark_vectors_whose_dtype_is_not_float32(self) -> None:
        vectors = DtypedVectors(
            [[0.0] * BENCHMARK_VECTOR_DIMENSION for _ in range(2)],
            dtype="float64",
        )

        with self.assertRaisesRegex(ValueError, "dtype float32"):
            validate_benchmark_vectors(vectors)

    def test_rejects_benchmark_vectors_with_non_sift1m_count(self) -> None:
        vectors = [[0.0] * BENCHMARK_VECTOR_DIMENSION for _ in range(2)]

        with self.assertRaisesRegex(ValueError, "SIFT-1M.*1,000,000.*got 2"):
            validate_benchmark_vectors(vectors, expected_vector_count=1_000_000)

    def test_scheduler_runs_5_warmups_then_10_measured_iterations(self) -> None:
        calls: list[tuple[str, int]] = []

        def run_iteration(phase: str, iteration: int) -> tuple[str, int]:
            calls.append((phase, iteration))
            return (phase, iteration)

        results = run_benchmark_iterations(run_iteration)

        expected_warmups = [
            ("warmup", iteration) for iteration in range(BENCHMARK_WARMUP_ITERATIONS)
        ]
        expected_measured = [
            ("measured", iteration)
            for iteration in range(BENCHMARK_MEASURED_ITERATIONS)
        ]

        self.assertEqual(calls, expected_warmups + expected_measured)
        self.assertEqual(results, expected_measured)

    def test_each_measured_invocation_receives_100_queries(self) -> None:
        seen_batch_sizes: list[int] = []

        def search(batch: list[int]) -> int:
            seen_batch_sizes.append(len(batch))
            return len(batch)

        queries = list(range(BENCHMARK_BATCH_SIZE * 3))

        results = run_benchmark_queries(queries, search)

        self.assertEqual(results, [100, 100, 100])
        self.assertEqual(seen_batch_sizes, [100, 100, 100])

    def test_rejects_tail_batch_smaller_than_100_queries(self) -> None:
        seen_batch_sizes: list[int] = []

        def search(batch: list[int]) -> int:
            seen_batch_sizes.append(len(batch))
            return len(batch)

        queries = list(range(BENCHMARK_BATCH_SIZE + 1))

        with self.assertRaisesRegex(ValueError, "multiple of 100"):
            run_benchmark_queries(queries, search)

        self.assertEqual(seen_batch_sizes, [])

    def test_rejects_non_standard_benchmark_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be 100"):
            run_benchmark_queries(list(range(100)), lambda batch: len(batch), batch_size=50)

    def test_metric_reporting_excludes_warmup_samples(self) -> None:
        samples = [
            BenchmarkSample("warmup", latency_seconds=100.0, query_count=100)
            for _ in range(BENCHMARK_WARMUP_ITERATIONS)
        ]
        samples.extend(
            BenchmarkSample("measured", latency_seconds=0.01, query_count=100)
            for _ in range(BENCHMARK_MEASURED_ITERATIONS)
        )

        metrics = report_benchmark_metrics(samples)

        self.assertEqual(metrics.measured_samples, 10)
        self.assertEqual(metrics.total_queries, 1000)
        self.assertEqual(metrics.qps, 10000.0)
        self.assertEqual(metrics.p50_latency_ms, 10.0)
        self.assertEqual(metrics.p99_latency_ms, 10.0)

    def test_p99_latency_uses_10_measured_batch_samples(self) -> None:
        samples = [
            BenchmarkSample("warmup", latency_seconds=999.0, query_count=100),
            BenchmarkSample("warmup", latency_seconds=500.0, query_count=100),
        ]
        samples.extend(
            BenchmarkSample("measured", latency_seconds=latency_ms / 1000.0, query_count=100)
            for latency_ms in [7.0, 3.0, 11.0, 5.0, 13.0, 2.0, 17.0, 19.0, 23.0, 29.0]
        )

        metrics = report_benchmark_metrics(samples)

        self.assertEqual(metrics.measured_samples, 10)
        self.assertEqual(metrics.p99_latency_ms, 29.0)

    def test_report_format_labels_p99_latency_as_ms_per_batch(self) -> None:
        samples = [
            BenchmarkSample("measured", latency_seconds=latency_ms / 1000.0, query_count=100)
            for latency_ms in [7.0, 3.0, 11.0, 5.0, 13.0, 2.0, 17.0, 19.0, 23.0, 29.0]
        ]

        report = format_benchmark_report(report_benchmark_metrics(samples))

        self.assertIn("p99_latency_ms_per_batch: 29.00 ms/batch", report)

    def test_report_output_serializes_hardware_metadata_fields(self) -> None:
        samples = [
            BenchmarkSample("measured", latency_seconds=0.01, query_count=100)
            for _ in range(BENCHMARK_MEASURED_ITERATIONS)
        ]
        identity = benchmark_identity_metadata(
            run_name="metal",
            backend="metal",
            implementation="cuvs-silicon",
            xcode_version="15.4",
            chip_model="Apple M3 Max",
            gpu_core_count=40,
            unified_memory_bytes=137438953472,
        )

        report = format_benchmark_report(report_benchmark_metrics(samples), identity)

        json_line = next(
            line.removeprefix("benchmark_result_json: ")
            for line in report.splitlines()
            if line.startswith("benchmark_result_json: ")
        )
        payload = json.loads(json_line)
        self.assertEqual(payload["identity_metadata"]["chip_model"], "Apple M3 Max")
        self.assertEqual(payload["identity_metadata"]["gpu_core_count"], 40)
        self.assertEqual(
            payload["identity_metadata"]["unified_memory_bytes"], 137438953472
        )
        self.assertEqual(payload["metrics"]["measured_samples"], 10)

    def test_report_output_includes_hardware_metadata_text_fields(self) -> None:
        samples = [
            BenchmarkSample("measured", latency_seconds=0.01, query_count=100)
            for _ in range(BENCHMARK_MEASURED_ITERATIONS)
        ]
        identity = benchmark_identity_metadata(
            run_name="metal",
            backend="metal",
            implementation="cuvs-silicon",
            xcode_version="15.4",
            chip_model="Apple M3 Max",
            gpu_core_count=40,
            unified_memory_bytes=137438953472,
        )

        report = format_benchmark_report(report_benchmark_metrics(samples), identity)

        self.assertIn("chip_model: Apple M3 Max", report)
        self.assertIn("gpu_core_count: 40", report)
        self.assertIn("unified_memory_bytes: 137438953472", report)

    def test_report_output_includes_xcode_version_text_metadata(self) -> None:
        samples = [
            BenchmarkSample("measured", latency_seconds=0.01, query_count=100)
            for _ in range(BENCHMARK_MEASURED_ITERATIONS)
        ]
        identity = benchmark_identity_metadata(
            run_name="metal",
            backend="metal",
            implementation="cuvs-silicon",
            xcode_version="15.4",
        )

        report = format_benchmark_report(report_benchmark_metrics(samples), identity)

        self.assertIn("xcode_version: 15.4", report)

    def test_metric_reporting_requires_10_measured_samples(self) -> None:
        samples = [
            BenchmarkSample("measured", latency_seconds=0.01, query_count=100)
            for _ in range(BENCHMARK_MEASURED_ITERATIONS - 1)
        ]

        with self.assertRaisesRegex(ValueError, "exactly 10 measured samples"):
            report_benchmark_metrics(samples)


def _benchmark_artifacts_with_identity(
    *,
    os_release: str,
    machine: str,
    processor: str,
) -> dict[str, dict[str, object]]:
    shared_environment = {
        "schema_version": 1,
        "os_name": "Darwin",
        "macos_version": "14.5",
        "os_release": os_release,
        "machine": machine,
        "processor": processor,
        "python_version": "3.12.8",
        "xcode_version": "15.4",
        "chip_model": "Apple M3 Max",
        "gpu_core_count": 40,
        "unified_memory_bytes": 137438953472,
    }
    return {
        name: {
            "identity_metadata": {
                **shared_environment,
                "run_name": name,
                "backend": name,
                "implementation": implementation,
            },
            "metrics": {"recall_at_10": 0.91},
        }
        for name, implementation in {
            "cpu_hnsw": "hnswlib-single-threaded",
            "metal": "cuvs-silicon",
            "faiss_mlx": "faiss-mlx",
        }.items()
    }


class _CompletedCommand:
    def __init__(self, *, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def _write_ivecs(path: Path, rows: list[list[int]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(struct.pack("<i", len(row)))
            output.write(struct.pack(f"<{len(row)}i", *row))


if __name__ == "__main__":
    unittest.main()
