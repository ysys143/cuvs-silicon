"""Benchmark query batching helpers."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from math import ceil
from os import PathLike
from pathlib import Path
import platform
import struct
import subprocess
from time import perf_counter
from typing import Protocol, TypeVar

BENCHMARK_BATCH_SIZE = 100
BENCHMARK_VECTOR_DIMENSION = 128
SIFT1M_BASE_VECTOR_COUNT = 1_000_000
SIFT1M_QUERY_VECTOR_COUNT = 10_000
SIFT1M_GROUNDTRUTH_ROW_WIDTH = 100
BENCHMARK_WARMUP_ITERATIONS = 5
BENCHMARK_MEASURED_ITERATIONS = 10
MINIMUM_BENCHMARK_MACOS_MAJOR_VERSION = 14
MINIMUM_BENCHMARK_MACOS_MINOR_VERSION = 0
MINIMUM_BENCHMARK_MACOS_PATCH_VERSION = 0
SIFT1M_DATASET_ID = "sift1m"
SIFT1M_BASE_FILENAME = "sift_base.fvecs"
SIFT1M_QUERY_FILENAME = "sift_query.fvecs"
SIFT1M_GROUNDTRUTH_FILENAME = "sift_groundtruth.ivecs"
SIFT1M_DISTANCE_METRIC = "l2"
BENCHMARK_IDENTITY_SCHEMA_VERSION = 1
REQUIRED_BENCHMARK_IDENTITY_FIELDS = (
    "schema_version",
    "run_name",
    "backend",
    "implementation",
    "os_name",
    "macos_version",
    "os_release",
    "machine",
    "processor",
    "python_version",
    "xcode_version",
    "chip_model",
    "gpu_core_count",
    "unified_memory_bytes",
)
BENCHMARK_ENVIRONMENT_IDENTITY_FIELDS = (
    "schema_version",
    "os_name",
    "macos_version",
    "os_release",
    "machine",
    "processor",
    "python_version",
    "xcode_version",
    "chip_model",
    "gpu_core_count",
    "unified_memory_bytes",
)
REQUIRED_BENCHMARK_RESULT_ARTIFACTS = ("cpu_hnsw", "metal", "faiss_mlx")

Query = TypeVar("Query")
SearchResult = TypeVar("SearchResult")
IterationResult = TypeVar("IterationResult")


class UnsupportedHardwareError(RuntimeError):
    """Raised when benchmark preflight rejects the current host."""


class HardwareDetectionResult(Protocol):
    supported: bool
    platform_name: str
    reason: str


@dataclass(frozen=True)
class BenchmarkSample:
    phase: str
    latency_seconds: float
    query_count: int


@dataclass(frozen=True)
class BenchmarkMetrics:
    measured_samples: int
    total_queries: int
    qps: float
    p50_latency_ms: float
    p99_latency_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "measured_samples": self.measured_samples,
            "total_queries": self.total_queries,
            "qps": self.qps,
            "p50_latency_ms": self.p50_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }


@dataclass(frozen=True)
class BenchmarkIdentityMetadata:
    schema_version: int
    run_name: str
    backend: str
    implementation: str
    os_name: str
    macos_version: str
    os_release: str
    machine: str
    processor: str
    python_version: str
    xcode_version: str
    chip_model: str
    gpu_core_count: int
    unified_memory_bytes: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "backend": self.backend,
            "implementation": self.implementation,
            "os_name": self.os_name,
            "macos_version": self.macos_version,
            "os_release": self.os_release,
            "machine": self.machine,
            "processor": self.processor,
            "python_version": self.python_version,
            "xcode_version": self.xcode_version,
            "chip_model": self.chip_model,
            "gpu_core_count": self.gpu_core_count,
            "unified_memory_bytes": self.unified_memory_bytes,
        }


@dataclass(frozen=True, order=True)
class MacOSVersion:
    major: int
    minor: int
    patch: int
    raw: str


@dataclass(frozen=True)
class Sift1MBenchmarkConfig:
    name: str
    vector_dimension: int
    batch_size: int
    distance_metric: str


@dataclass(frozen=True)
class Sift1MDatasetSelection:
    dataset_id: str
    base_path: Path
    query_path: Path
    groundtruth_path: Path


def parse_macos_version(version_text: str) -> MacOSVersion:
    """Parse a macOS version string into comparable numeric components."""
    raw = version_text.strip()
    if not raw:
        raise ValueError("macOS version is empty")

    pieces = raw.split(".")
    if len(pieces) > 3:
        raise ValueError(f"macOS version has too many components: {version_text!r}")
    if any(not piece.isdecimal() for piece in pieces):
        raise ValueError(f"macOS version contains non-numeric components: {version_text!r}")

    components = [int(piece) for piece in pieces]
    components.extend([0] * (3 - len(components)))
    return MacOSVersion(
        major=components[0],
        minor=components[1],
        patch=components[2],
        raw=raw,
    )


def detect_macos_version(
    version_provider: Callable[[], tuple[str, object, str]] = platform.mac_ver,
) -> MacOSVersion:
    """Detect and parse the current host macOS version."""
    version_text = version_provider()[0]
    return parse_macos_version(version_text)


def validate_benchmark_macos_version(parsed_version: MacOSVersion) -> None:
    """Reject benchmark hosts older than macOS 14.0 Sonoma."""
    minimum_components = (
        MINIMUM_BENCHMARK_MACOS_MAJOR_VERSION,
        MINIMUM_BENCHMARK_MACOS_MINOR_VERSION,
        MINIMUM_BENCHMARK_MACOS_PATCH_VERSION,
    )
    parsed_components = (
        parsed_version.major,
        parsed_version.minor,
        parsed_version.patch,
    )
    if parsed_components < minimum_components:
        raise UnsupportedHardwareError(
            "unsupported benchmark environment: macOS "
            f"{MINIMUM_BENCHMARK_MACOS_MAJOR_VERSION}."
            f"{MINIMUM_BENCHMARK_MACOS_MINOR_VERSION}+ is required; "
            f"detected {parsed_version.raw}"
        )


def benchmark_identity_metadata(
    *,
    run_name: str,
    backend: str,
    implementation: str,
    xcode_version: str | None = None,
    chip_model: str = "unavailable",
    gpu_core_count: int = 0,
    unified_memory_bytes: int = 0,
) -> BenchmarkIdentityMetadata:
    """Record the shared identity schema for comparable benchmark runs."""
    return BenchmarkIdentityMetadata(
        schema_version=BENCHMARK_IDENTITY_SCHEMA_VERSION,
        run_name=run_name,
        backend=backend,
        implementation=implementation,
        os_name=platform.system(),
        macos_version=platform.mac_ver()[0],
        os_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        xcode_version=xcode_version or detect_xcode_version(),
        chip_model=chip_model,
        gpu_core_count=gpu_core_count,
        unified_memory_bytes=unified_memory_bytes,
    )


def detect_xcode_version(
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Return the installed Xcode version used for benchmark reporting."""
    try:
        result = command_runner(
            ["xcodebuild", "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return "unavailable"

    if result.returncode != 0:
        return "unavailable"

    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate.startswith("Xcode "):
            version = candidate.removeprefix("Xcode ").strip()
            return version or "unavailable"
    return "unavailable"


def cpu_hnsw_baseline_identity_metadata() -> BenchmarkIdentityMetadata:
    """Record CPU HNSW baseline identity using the shared benchmark schema."""
    return benchmark_identity_metadata(
        run_name="cpu_hnsw_baseline",
        backend="cpu_hnsw",
        implementation="hnswlib-single-threaded",
    )


def validate_benchmark_metadata_identity(
    artifacts: Mapping[str, Mapping[str, object]],
) -> None:
    """Verify comparable benchmark artifacts were recorded on the same host."""
    missing_artifacts = [
        name for name in REQUIRED_BENCHMARK_RESULT_ARTIFACTS if name not in artifacts
    ]
    if missing_artifacts:
        raise ValueError(
            "missing benchmark result artifacts: " + ", ".join(missing_artifacts)
        )

    reference_name = REQUIRED_BENCHMARK_RESULT_ARTIFACTS[0]
    reference_identity = _artifact_identity_metadata(
        reference_name, artifacts[reference_name]
    )
    reference_values = {
        field: reference_identity[field]
        for field in BENCHMARK_ENVIRONMENT_IDENTITY_FIELDS
    }

    for artifact_name in REQUIRED_BENCHMARK_RESULT_ARTIFACTS[1:]:
        identity = _artifact_identity_metadata(artifact_name, artifacts[artifact_name])
        mismatches = [
            field
            for field, expected in reference_values.items()
            if identity[field] != expected
        ]
        if mismatches:
            mismatch_text = ", ".join(mismatches)
            raise ValueError(
                "benchmark result artifact identity mismatch between "
                f"{reference_name} and {artifact_name}: {mismatch_text}"
            )


def require_supported_benchmark_hardware(
    detection: HardwareDetectionResult,
    *,
    macos_version: str | None = None,
) -> None:
    """Fail benchmark preflight when host hardware or OS is unsupported."""
    if detection.supported:
        version_text = macos_version
        if version_text is None:
            version_text = platform.mac_ver()[0]
        _require_supported_benchmark_macos_version(version_text)
        return

    platform_name = detection.platform_name or "unknown hardware"
    reason = detection.reason or "hardware detection reported unsupported"
    raise UnsupportedHardwareError(
        f"unsupported benchmark hardware: {platform_name}: {reason}"
    )


def generate_sift1m_benchmark_configs(
    names: Sequence[str],
    *,
    distance_metric: str = SIFT1M_DISTANCE_METRIC,
) -> list[Sift1MBenchmarkConfig]:
    """Generate fixed SIFT-1M benchmark configs."""
    if distance_metric != SIFT1M_DISTANCE_METRIC:
        raise ValueError(
            "SIFT-1M benchmark distance_metric must be "
            f"{SIFT1M_DISTANCE_METRIC}; got {distance_metric}"
        )

    return [
        Sift1MBenchmarkConfig(
            name=name,
            vector_dimension=BENCHMARK_VECTOR_DIMENSION,
            batch_size=BENCHMARK_BATCH_SIZE,
            distance_metric=distance_metric,
        )
        for name in names
    ]


def load_sift1m_dataset_selection(
    dataset_root: str | PathLike[str],
    *,
    dataset_id: str = SIFT1M_DATASET_ID,
) -> Sift1MDatasetSelection:
    """Resolve the canonical SIFT-1M benchmark files under a dataset root."""
    if dataset_id != SIFT1M_DATASET_ID:
        raise ValueError(f"unsupported benchmark dataset_id {dataset_id!r}")

    root = Path(dataset_root)
    selection = Sift1MDatasetSelection(
        dataset_id=dataset_id,
        base_path=root / SIFT1M_BASE_FILENAME,
        query_path=root / SIFT1M_QUERY_FILENAME,
        groundtruth_path=root / SIFT1M_GROUNDTRUTH_FILENAME,
    )
    missing_paths = [
        path
        for path in (
            selection.base_path,
            selection.query_path,
            selection.groundtruth_path,
        )
        if not path.is_file()
    ]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"missing SIFT-1M benchmark files: {missing_text}")

    return selection


def load_sift_groundtruth_ivecs(
    path: str | PathLike[str],
    *,
    expected_row_width: int = SIFT1M_GROUNDTRUTH_ROW_WIDTH,
) -> list[list[int]]:
    """Load SIFT ground-truth .ivecs rows as integer neighbor ids."""
    rows: list[list[int]] = []
    with open(path, "rb") as file:
        while True:
            width_bytes = file.read(4)
            if not width_bytes:
                return rows
            if len(width_bytes) != 4:
                raise ValueError("ivecs file ended before row width")

            (row_width,) = struct.unpack("<i", width_bytes)
            if row_width != expected_row_width:
                raise ValueError(
                    f"ivecs row width {row_width} does not match expected "
                    f"row width {expected_row_width}"
                )

            payload = file.read(row_width * 4)
            if len(payload) != row_width * 4:
                raise ValueError("ivecs file ended before row payload")
            rows.append(list(struct.unpack(f"<{row_width}i", payload)))


def _artifact_identity_metadata(
    artifact_name: str,
    artifact: Mapping[str, object],
) -> Mapping[str, object]:
    metadata = artifact.get("identity_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"{artifact_name} benchmark result artifact is missing identity_metadata"
        )

    missing_fields = [
        field for field in REQUIRED_BENCHMARK_IDENTITY_FIELDS if field not in metadata
    ]
    if missing_fields:
        raise ValueError(
            f"{artifact_name} identity_metadata is missing required fields: "
            + ", ".join(missing_fields)
        )
    return metadata


def _require_supported_benchmark_macos_version(macos_version: str) -> None:
    try:
        parsed_version = parse_macos_version(macos_version)
    except ValueError as error:
        raise UnsupportedHardwareError(
            f"unsupported benchmark environment: could not parse macOS version "
            f"{macos_version!r}"
        ) from error

    validate_benchmark_macos_version(parsed_version)


def format_benchmark_report(
    metrics: BenchmarkMetrics,
    identity_metadata: BenchmarkIdentityMetadata | None = None,
) -> str:
    """Format aggregate benchmark output with serialized comparable metadata."""
    lines = [
        f"measured_samples: {metrics.measured_samples}",
        f"total_queries: {metrics.total_queries}",
        f"qps: {metrics.qps:.2f}",
        f"p50_latency_ms_per_batch: {metrics.p50_latency_ms:.2f} ms/batch",
        f"p99_latency_ms_per_batch: {metrics.p99_latency_ms:.2f} ms/batch",
    ]
    if identity_metadata is not None:
        lines.extend(
            [
                f"xcode_version: {identity_metadata.xcode_version}",
                f"chip_model: {identity_metadata.chip_model}",
                f"gpu_core_count: {identity_metadata.gpu_core_count}",
                f"unified_memory_bytes: {identity_metadata.unified_memory_bytes}",
            ]
        )
        lines.append(
            "benchmark_result_json: "
            + json.dumps(
                {
                    "identity_metadata": identity_metadata.as_dict(),
                    "metrics": metrics.as_dict(),
                },
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def validate_benchmark_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_vector_count: int | None = None,
) -> None:
    """Validate that benchmark vectors use the SIFT-1M 128d contract."""
    dtype = getattr(vectors, "dtype", None)
    if dtype is not None and str(dtype) != "float32":
        raise ValueError(f"benchmark vectors must have dtype float32; got {dtype}")

    if expected_vector_count is not None and len(vectors) != expected_vector_count:
        raise ValueError(
            "SIFT-1M benchmark vectors must contain "
            f"{expected_vector_count:,} vectors; got {len(vectors):,}"
        )

    for index, vector in enumerate(vectors):
        if len(vector) != BENCHMARK_VECTOR_DIMENSION:
            raise ValueError(
                "benchmark vectors must have dimensionality "
                f"{BENCHMARK_VECTOR_DIMENSION}; vector {index} has {len(vector)}"
            )


def evaluate_sift1m_search_distances(
    query: Sequence[float],
    candidates: Sequence[Sequence[float]],
    config: Sift1MBenchmarkConfig,
) -> list[float]:
    """Evaluate benchmark candidate distances using SIFT-1M squared L2."""
    if config.distance_metric != SIFT1M_DISTANCE_METRIC:
        raise ValueError(
            "SIFT-1M benchmark distance_metric must be "
            f"{SIFT1M_DISTANCE_METRIC}; got {config.distance_metric}"
        )
    if len(query) != BENCHMARK_VECTOR_DIMENSION:
        raise ValueError(
            "benchmark query must have dimensionality "
            f"{BENCHMARK_VECTOR_DIMENSION}; got {len(query)}"
        )
    validate_benchmark_vectors(candidates)

    distances: list[float] = []
    for candidate in candidates:
        distance = 0.0
        for query_value, candidate_value in zip(query, candidate):
            delta = query_value - candidate_value
            distance += delta * delta
        distances.append(distance)
    return distances


def load_sift_groundtruth_ivecs(
    path: str | PathLike[str],
    *,
    expected_row_width: int = SIFT1M_GROUNDTRUTH_ROW_WIDTH,
) -> list[list[int]]:
    """Load SIFT ground-truth ``.ivecs`` rows as integer neighbor IDs."""
    if expected_row_width <= 0:
        raise ValueError("expected ivecs row width must be positive")

    rows: list[list[int]] = []
    with open(path, "rb") as input_file:
        row_index = 0
        while True:
            header = input_file.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError("ivecs file ended inside row width header")

            (row_width,) = struct.unpack("<i", header)
            if row_width != expected_row_width:
                raise ValueError(
                    f"expected ivecs row width {expected_row_width}; "
                    f"row {row_index} has {row_width}"
                )

            payload = input_file.read(row_width * 4)
            if len(payload) != row_width * 4:
                raise ValueError("ivecs file ended inside neighbor payload")

            row = list(struct.unpack(f"<{row_width}i", payload))
            if any(neighbor < 0 for neighbor in row):
                raise ValueError("ivecs neighbor IDs must be non-negative")
            rows.append(row)
            row_index += 1

    return rows


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot report metrics without measured samples")
    sorted_values = sorted(values)
    index = ceil((percentile / 100.0) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def report_benchmark_metrics(samples: Sequence[BenchmarkSample]) -> BenchmarkMetrics:
    """Report benchmark metrics from measured samples only.

    The benchmark protocol records warmup iterations for execution stability,
    but reported QPS and latency must be computed from exactly the fixed 10
    measured iterations.
    """
    measured = [sample for sample in samples if sample.phase == "measured"]
    if len(measured) != BENCHMARK_MEASURED_ITERATIONS:
        raise ValueError(
            "benchmark metrics require exactly "
            f"{BENCHMARK_MEASURED_ITERATIONS} measured samples; got {len(measured)}"
        )

    total_queries = sum(sample.query_count for sample in measured)
    total_latency_seconds = sum(sample.latency_seconds for sample in measured)
    if total_latency_seconds <= 0:
        raise ValueError("total measured latency must be positive")

    latencies_ms = [sample.latency_seconds * 1000.0 for sample in measured]
    return BenchmarkMetrics(
        measured_samples=len(measured),
        total_queries=total_queries,
        qps=total_queries / total_latency_seconds,
        p50_latency_ms=_nearest_rank_percentile(latencies_ms, 50.0),
        p99_latency_ms=_nearest_rank_percentile(latencies_ms, 99.0),
    )


def run_benchmark_iterations(
    run_iteration: Callable[[str, int], IterationResult],
) -> list[IterationResult]:
    """Execute the fixed benchmark iteration schedule.

    The benchmark protocol runs warmup iterations before measured iterations so
    setup effects do not pollute reported latency and QPS.
    """
    for iteration in range(BENCHMARK_WARMUP_ITERATIONS):
        run_iteration("warmup", iteration)

    measured_results: list[IterationResult] = []
    for iteration in range(BENCHMARK_MEASURED_ITERATIONS):
        measured_start = perf_counter()
        result = run_iteration("measured", iteration)
        _elapsed_seconds = perf_counter() - measured_start
        measured_results.append(result)
    return measured_results


def run_benchmark_queries(
    queries: Sequence[Query],
    search: Callable[[Sequence[Query]], SearchResult],
    *,
    batch_size: int = BENCHMARK_BATCH_SIZE,
) -> list[SearchResult]:
    """Run measured benchmark search calls using exact-size query batches.

    Benchmark comparisons require batch_size=100. This helper rejects a query
    set that would create a smaller tail invocation instead of measuring it.
    """
    if batch_size != BENCHMARK_BATCH_SIZE:
        raise ValueError(f"benchmark batch_size must be {BENCHMARK_BATCH_SIZE}")
    if len(queries) % batch_size != 0:
        raise ValueError(
            f"benchmark query count must be a multiple of {batch_size}; "
            f"got {len(queries)}"
        )

    results: list[SearchResult] = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        measured_start = perf_counter()
        result = search(batch)
        _elapsed_seconds = perf_counter() - measured_start
        results.append(result)
    return results
