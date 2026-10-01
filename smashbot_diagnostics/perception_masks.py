"""Conservative Task 009 color and motion masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .perception_registration import RegistrationResult


class MaskError(ValueError):
    """Raised for invalid frame/mask inputs."""


@dataclass(frozen=True)
class MaskConfig:
    hud_rows: int = 260
    white_saturation_max: int = 90
    white_value_min: int = 170
    yellow_hue_low: int = 12
    yellow_hue_high: int = 45
    yellow_saturation_min: int = 70
    yellow_value_min: int = 100
    cyan_hue_low: int = 78
    cyan_hue_high: int = 108
    cyan_saturation_min: int = 80
    cyan_value_min: int = 80
    motion_delta_threshold: int = 18


BASELINE_MASKS = MaskConfig()


def _opencv() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MaskError("Task 009 masks require the optional [perception] extra") from exc
    return cv2, numpy


@dataclass(frozen=True)
class MaskBundle:
    body: Any
    trail: Any
    motion: Any
    eligible: Any
    # Keep the already-computed component masks available to explicit V1
    # paths.  Existing BASELINE_UNTUNED consumers continue to use body/trail/
    # motion unchanged; these fields avoid rebuilding HSV masks for V1.
    yellow: Any | None = None
    white: Any | None = None


def _gray(frame: Any, cv2: Any) -> Any:
    if getattr(frame, "ndim", None) == 2:
        return frame
    if getattr(frame, "ndim", None) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise MaskError("frame must be grayscale or BGR with three channels")


def build_masks(
    frame: Any,
    *,
    previous_frame: Any | None = None,
    registration: RegistrationResult | None = None,
    config: MaskConfig = BASELINE_MASKS,
) -> MaskBundle:
    """Build separate body, trail, motion, and non-HUD masks.

    The previous frame is used only for motion evidence. Ground-truth labels
    are intentionally not accepted by this API.
    """

    cv2, numpy = _opencv()
    if getattr(frame, "ndim", None) != 3 or frame.shape[2] != 3:
        raise MaskError("mask input must be a three-channel BGR frame")
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hsv,
        numpy.array([0, 0, config.white_value_min], dtype=numpy.uint8),
        numpy.array([180, config.white_saturation_max, 255], dtype=numpy.uint8),
    )
    yellow = cv2.inRange(
        hsv,
        numpy.array([config.yellow_hue_low, config.yellow_saturation_min, config.yellow_value_min], dtype=numpy.uint8),
        numpy.array([config.yellow_hue_high, 255, 255], dtype=numpy.uint8),
    )
    cyan = cv2.inRange(
        hsv,
        numpy.array([config.cyan_hue_low, config.cyan_saturation_min, config.cyan_value_min], dtype=numpy.uint8),
        numpy.array([config.cyan_hue_high, 255, 255], dtype=numpy.uint8),
    )
    eligible = numpy.ones((height, width), dtype=numpy.uint8) * 255
    eligible[: min(config.hud_rows, height), :] = 0
    yellow = cv2.bitwise_and(yellow, eligible)
    white = cv2.bitwise_and(white, eligible)
    # Preserve the historical BASELINE_UNTUNED operation exactly: the body
    # morphology is applied after the union, while V1 receives separate
    # component masks below.
    body = cv2.bitwise_or(white, yellow)
    trail = cv2.bitwise_and(cyan, eligible)
    kernel = numpy.ones((3, 3), dtype=numpy.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    trail = cv2.morphologyEx(trail, cv2.MORPH_OPEN, kernel)
    motion = numpy.zeros((height, width), dtype=numpy.uint8)
    if previous_frame is not None:
        previous_gray = _gray(previous_frame, cv2)
        current_gray = _gray(frame, cv2)
        if previous_gray.shape != current_gray.shape:
            raise MaskError("previous and current frame dimensions differ")
        aligned = previous_gray
        if registration is not None and registration.success:
            matrix = numpy.float32([[1, 0, -registration.dx], [0, 1, -registration.dy]])
            aligned = cv2.warpAffine(current_gray, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
            difference = cv2.absdiff(previous_gray, aligned)
        else:
            difference = cv2.absdiff(previous_gray, current_gray)
        motion = cv2.threshold(difference, config.motion_delta_threshold, 255, cv2.THRESH_BINARY)[1]
        motion = cv2.bitwise_and(motion, eligible)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
    return MaskBundle(body=body, trail=trail, motion=motion, eligible=eligible, yellow=yellow, white=white)
