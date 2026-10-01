"""BASELINE_UNTUNED camera registration for Task 009.

The first model is intentionally translation-only. OpenCV is imported lazily so
the diagnostics core remains usable without the optional perception extra.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from .perception_models import RegistrationResult


class RegistrationError(ValueError):
    """Raised for invalid registration inputs or configuration."""


@dataclass(frozen=True)
class TranslationRegistrationConfig:
    """Frozen first-pass parameters; these are not tuned from benchmark output."""

    max_corners: int = 250
    quality_level: float = 0.01
    min_distance: float = 7.0
    block_size: int = 7
    min_inliers: int = 8
    min_inlier_ratio: float = 0.50
    residual_cutoff_px: float = 3.5

    def __post_init__(self) -> None:
        if self.max_corners < 1 or self.min_inliers < 1 or self.block_size < 3:
            raise RegistrationError("corner and inlier counts must be positive")
        if not 0 < self.quality_level < 1 or self.min_distance <= 0:
            raise RegistrationError("feature parameters are invalid")
        if self.min_inliers > self.max_corners:
            raise RegistrationError("min_inliers cannot exceed max_corners")
        if not 0 < self.min_inlier_ratio <= 1 or self.residual_cutoff_px <= 0:
            raise RegistrationError("robust registration parameters are invalid")


BASELINE_UNTUNED = TranslationRegistrationConfig()


def _opencv() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RegistrationError(
            "camera registration requires the optional [perception] extra "
            "(numpy + opencv-python-headless)"
        ) from exc
    return cv2, numpy


def _gray(frame: Any, cv2: Any) -> Any:
    if getattr(frame, "ndim", None) == 2:
        return frame
    if getattr(frame, "ndim", None) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise RegistrationError("frame must be grayscale or a 3-channel image")


def _failure(start: float, reason: str) -> RegistrationResult:
    return RegistrationResult(
        success=False,
        failure_reason=reason,
        processing_ms=(time.perf_counter() - start) * 1000.0,
        model="translation",
    )


def _robust_translation(previous_points: Any, current_points: Any, config: TranslationRegistrationConfig = BASELINE_UNTUNED) -> RegistrationResult:
    """Estimate displacement from point pairs using median + residual rejection."""

    _, numpy = _opencv()
    previous = numpy.asarray(previous_points, dtype=numpy.float64).reshape(-1, 2)
    current = numpy.asarray(current_points, dtype=numpy.float64).reshape(-1, 2)
    if previous.shape != current.shape or previous.shape[0] == 0:
        return RegistrationResult(False, failure_reason="invalid_point_pairs", model="translation")
    if not numpy.isfinite(previous).all() or not numpy.isfinite(current).all():
        return RegistrationResult(False, failure_reason="nonfinite_point_pairs", model="translation")
    displacement = current - previous
    median = numpy.median(displacement, axis=0)
    residuals = numpy.linalg.norm(displacement - median, axis=1)
    inlier_mask = residuals <= config.residual_cutoff_px
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / float(len(displacement))
    if inlier_count < config.min_inliers:
        return RegistrationResult(False, failure_reason="insufficient_inliers", inliers=inlier_count, inlier_ratio=inlier_ratio, model="translation")
    if inlier_ratio < config.min_inlier_ratio:
        return RegistrationResult(False, failure_reason="low_inlier_ratio", inliers=inlier_count, inlier_ratio=inlier_ratio, model="translation")
    robust = numpy.median(displacement[inlier_mask], axis=0)
    robust_residuals = numpy.linalg.norm(displacement[inlier_mask] - robust, axis=1)
    residual_px = float(numpy.sqrt(numpy.mean(robust_residuals * robust_residuals)))
    return RegistrationResult(True, dx=float(robust[0]), dy=float(robust[1]), inliers=inlier_count, residual_px=residual_px, model="translation", inlier_ratio=inlier_ratio)


def register_translation(previous_frame: Any, current_frame: Any, *, mask: Any | None = None, config: TranslationRegistrationConfig = BASELINE_UNTUNED) -> RegistrationResult:
    """Register two frames with translation-only sparse optical flow.

    The result is invalid on shape mismatch, unavailable optional dependencies,
    insufficient features/inliers, or weak robust consensus.
    """

    start = time.perf_counter()
    try:
        cv2, numpy = _opencv()
        previous_gray = _gray(previous_frame, cv2)
        current_gray = _gray(current_frame, cv2)
        if previous_gray.shape != current_gray.shape or previous_gray.ndim != 2:
            return _failure(start, "invalid_dimensions")
        if previous_gray.shape[0] < 16 or previous_gray.shape[1] < 16:
            return _failure(start, "frames_too_small")
        if mask is not None and (getattr(mask, "shape", None) != previous_gray.shape or getattr(mask, "dtype", None) != numpy.uint8):
            return _failure(start, "invalid_mask")
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=config.max_corners,
            qualityLevel=config.quality_level,
            minDistance=config.min_distance,
            mask=mask,
            blockSize=config.block_size,
        )
        if previous_points is None or len(previous_points) < config.min_inliers:
            return _failure(start, "insufficient_features")
        current_points, status, _error = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_points, None)
        if current_points is None or status is None:
            return _failure(start, "optical_flow_failed")
        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < config.min_inliers:
            return _failure(start, "insufficient_tracked_features")
        result = _robust_translation(previous_points.reshape(-1, 2)[valid], current_points.reshape(-1, 2)[valid], config)
        return RegistrationResult(
            success=result.success,
            dx=result.dx,
            dy=result.dy,
            affine=result.affine,
            inliers=result.inliers,
            residual_px=result.residual_px,
            failure_reason=result.failure_reason,
            processing_ms=(time.perf_counter() - start) * 1000.0,
            model="translation",
            inlier_ratio=result.inlier_ratio,
        )
    except RegistrationError:
        raise
    except Exception as exc:  # OpenCV can throw on malformed image buffers.
        return _failure(start, f"opencv_error:{type(exc).__name__}")


def registration_dict(result: RegistrationResult) -> dict[str, Any]:
    """Return the stable JSON-facing transition contract."""

    return {
        "success": result.success,
        "model": result.model,
        "dx": result.dx,
        "dy": result.dy,
        "inlier_count": result.inliers,
        "inliers": result.inliers,
        "inlier_ratio": result.inlier_ratio,
        "residual_px": result.residual_px,
        "processing_ms": result.processing_ms,
        "failure_reason": result.failure_reason,
    }
