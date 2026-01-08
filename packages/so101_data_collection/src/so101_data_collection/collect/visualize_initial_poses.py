#!/usr/bin/env python
"""
Visualize initial poses from dataset using classical CV.

Extracts the 5th frame from each episode and uses classical computer vision
to detect:
- Mat (black rectangle) - position/bounds
- Cube (green cube) - position and orientation
- Coaster (circular patterned object) - position

Outputs a scatter plot showing initial positions similar to training data
distribution plots.

Usage:
    python -m so101_data_collection.collect.visualize_initial_poses \
        --repo-id giacomoran/so101_data_collection_cube_hand_guided \
        --dataset-root data
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from so101_data_collection.collect.collect import DEFAULT_DATASET_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of object detection for a single frame."""

    episode_idx: int
    # Mat detection
    mat_detected: bool = False
    mat_bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    mat_corners: np.ndarray | None = None  # 4 corner points for orientation
    mat_angle: float | None = None  # Mat rotation angle in degrees
    # Cube detection
    cube_detected: bool = False
    cube_center: tuple[float, float] | None = None  # (cx, cy)
    cube_angle: float | None = None  # degrees
    cube_bbox: tuple[tuple[float, float], tuple[float, float], float] | None = (
        None  # minAreaRect result
    )
    # Coaster detection
    coaster_detected: bool = False
    coaster_center: tuple[float, float] | None = None  # (cx, cy)
    coaster_radius: float | None = None
    coaster_angle: float | None = None  # Orientation from pattern analysis


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is in uint8 format for OpenCV."""
    if image.dtype == np.uint8:
        return image

    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        else:
            return image.astype(np.uint8)

    return image.astype(np.uint8)


def _to_numpy_hwc(img: Any) -> np.ndarray:
    """Convert image to numpy HWC format."""
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    # Handle torch.Tensor
    try:
        import torch

        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            return img_np
    except ImportError:
        pass

    # Try direct conversion
    return np.array(img)


def detect_mat(
    image: np.ndarray,
) -> tuple[bool, tuple[int, int, int, int] | None, np.ndarray | None, float | None]:
    """
    Detect the black mat using threshold and contour detection.

    The mat is a black rectangular surface on a white desk, typically in the
    center-right portion of the image.

    Args:
        image: BGR or RGB image

    Returns:
        (detected, bbox, corners, angle) where:
        - bbox is (x, y, w, h) or None
        - corners is array of 4 corner points or None
        - angle is rotation angle in degrees or None
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    img_h, img_w = gray.shape[:2]

    # Threshold to find dark regions (mat is black)
    # Use a moderate threshold to avoid picking up shadows
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, None, None, None

    # Filter contours to find the mat
    # Mat should be: rectangular, reasonably large, and roughly in center-right
    best_contour = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        img_area = img_h * img_w

        # Mat should be 5-40% of image area
        if area < img_area * 0.05 or area > img_area * 0.5:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Check aspect ratio (mat is roughly rectangular, wider than tall or square)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.5:
            continue

        # Mat should be in the right portion of the image (not on the left edge)
        center_x = x + w / 2
        if center_x < img_w * 0.25:  # Reject if center is in leftmost 25%
            continue

        # Check rectangularity (how well the contour fills its bounding rect)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < 0.6:  # Mat should be fairly rectangular
            continue

        # Score based on area and position (prefer larger, more central)
        score = area * rectangularity
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return False, None, None, None

    # Get bounding box
    x, y, w, h = cv2.boundingRect(best_contour)

    # Get minimum area rectangle for precise corners and angle
    min_rect = cv2.minAreaRect(best_contour)
    corners = cv2.boxPoints(min_rect)
    corners = np.int32(corners)
    angle = min_rect[2]

    # Normalize angle (-90 to 0 range from minAreaRect)
    rect_w, rect_h = min_rect[1]
    if rect_w < rect_h:
        angle = angle + 90

    return True, (x, y, w, h), corners, angle


def detect_cube(
    image: np.ndarray,
    mat_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[bool, tuple[float, float] | None, float | None, Any]:
    """
    Detect the green cube using HSV color filtering.

    The cube is a small, bright green/teal square object (like a keycap).
    It should be on or near the mat, not part of the robot arm.

    Args:
        image: RGB image
        mat_bbox: Optional mat bounding box to help filter candidates

    Returns:
        (detected, center, angle, minAreaRect_result)
    """
    img_h, img_w = image.shape[:2]

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Try multiple HSV ranges to handle lighting variations
    # The cube is a bright green/teal color
    hsv_ranges = [
        # Primary range: bright green/teal
        (np.array([35, 60, 60]), np.array([85, 255, 255])),
        # Slightly different lighting
        (np.array([40, 40, 80]), np.array([90, 255, 255])),
    ]

    all_contours = []
    for lower, upper in hsv_ranges:
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

    if not all_contours:
        return False, None, None, None

    # Find the best cube candidate
    best_contour = None
    best_score = 0

    for contour in all_contours:
        area = cv2.contourArea(contour)

        # Cube area range (relaxed)
        if area < 200 or area > 20000:
            continue

        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        center = rect[0]
        (w, h) = rect[1]

        if w == 0 or h == 0:
            continue

        # Check aspect ratio (cube should be roughly square)
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.5:  # Too elongated
            continue

        # Reject if in the bottom 30% of image (likely robot arm)
        if center[1] > img_h * 0.70:
            continue

        # Must be on mat if mat detected
        on_mat = False
        if mat_bbox is not None:
            mx, my, mw, mh = mat_bbox
            if mx <= center[0] <= mx + mw and my <= center[1] <= my + mh:
                on_mat = True
            else:
                # Skip candidates not on mat (cube should always be on mat)
                continue

        # Score: prefer larger, more square contours
        squareness = 1.0 / aspect_ratio
        score = area * squareness * (2.0 if on_mat else 1.0)

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return False, None, None, None

    # Get minimum area rectangle for orientation
    rect = cv2.minAreaRect(best_contour)
    center = rect[0]
    angle = rect[2]

    # Normalize angle to be consistent
    width, height = rect[1]
    if width < height:
        angle = angle + 90

    return True, center, angle, rect


def detect_coaster(
    image: np.ndarray, mat_bbox: tuple[int, int, int, int] | None = None
) -> tuple[bool, tuple[float, float] | None, float | None, float | None]:
    """
    Detect the circular coaster using pattern/texture analysis.

    The coaster is a circular patterned object (like a decorative coaster)
    with a distinctive mandala-like pattern. Uses texture variance to identify
    the patterned region and computes orientation from pattern gradients.

    Args:
        image: RGB image
        mat_bbox: Optional mat bounding box to focus search area

    Returns:
        (detected, center, radius, angle) where angle is orientation in degrees
    """
    # Coaster must be on the mat - if no mat detected, can't reliably find coaster
    if mat_bbox is None:
        return False, None, None, None

    mx, my, mw, mh = mat_bbox

    # Extract mat region
    mat_region = image[my : my + mh, mx : mx + mw]
    gray_mat = cv2.cvtColor(mat_region, cv2.COLOR_RGB2GRAY)

    # The coaster has a distinctive pattern with high local variance
    # Use Laplacian variance to detect textured regions
    laplacian = cv2.Laplacian(gray_mat, cv2.CV_64F)

    # Also look for the brownish/reddish color of the coaster
    hsv_mat = cv2.cvtColor(mat_region, cv2.COLOR_RGB2HSV)

    # Coaster color: brownish/reddish tones (H: 0-20 or 160-180, moderate S, moderate V)
    # Create mask for coaster-like colors
    lower_brown1 = np.array([0, 30, 50])
    upper_brown1 = np.array([25, 200, 200])
    lower_brown2 = np.array([160, 30, 50])
    upper_brown2 = np.array([180, 200, 200])

    mask_brown1 = cv2.inRange(hsv_mat, lower_brown1, upper_brown1)
    mask_brown2 = cv2.inRange(hsv_mat, lower_brown2, upper_brown2)
    color_mask = cv2.bitwise_or(mask_brown1, mask_brown2)

    # Clean up the color mask
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the color mask
    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_result = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        # Coaster area should be roughly pi * r^2 where r is 40-70
        if area < 3000 or area > 20000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Must be reasonably circular
        if circularity < 0.5:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # Check texture variance within the detected region
        # Create a mask for this region
        region_mask = np.zeros(gray_mat.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)

        # Compute texture score (Laplacian variance)
        masked_laplacian = np.abs(laplacian) * (region_mask / 255.0)
        texture_score = np.sum(masked_laplacian) / max(area, 1)

        # The coaster has high texture due to its pattern
        if texture_score < 5:  # Low texture, probably not the coaster
            continue

        # Score based on circularity, texture, and position
        score = circularity * 50 + texture_score * 10

        # Bonus for right side of mat (coaster typically there)
        relative_x = cx / mw
        if relative_x > 0.4:
            score += 50

        # Bonus for upper portion of mat
        relative_y = cy / mh
        if relative_y < 0.6:
            score += 30

        if score > best_score:
            best_score = score
            best_result = (cx, cy, radius, contour)

    if best_result is not None and best_score > 80:
        cx, cy, radius, contour = best_result

        # Compute orientation from the pattern using gradient analysis
        angle = compute_coaster_orientation(gray_mat, int(cx), int(cy), int(radius))

        # Convert to full image coordinates
        full_cx = cx + mx
        full_cy = cy + my

        return True, (float(full_cx), float(full_cy)), float(radius), angle

    # Fallback to Hough circles if color-based detection fails
    blurred = cv2.GaussianBlur(gray_mat, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=50,
        param2=30,
        minRadius=35,
        maxRadius=75,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = None
        best_score = -1

        for circle in circles[0, :]:
            cx, cy, r = circle

            # Check texture at this location
            y1, y2 = max(0, int(cy - r)), min(gray_mat.shape[0], int(cy + r))
            x1, x2 = max(0, int(cx - r)), min(gray_mat.shape[1], int(cx + r))
            region = gray_mat[y1:y2, x1:x2]
            if region.size == 0:
                continue
            texture_var = np.var(region)

            # Score based on texture and position
            score = texture_var / 10

            relative_x = cx / mw
            if relative_x > 0.4:
                score += 80

            relative_y = cy / mh
            if relative_y < 0.6:
                score += 40

            if score > best_score:
                best_score = score
                best_circle = (cx, cy, r)

        if best_circle is not None and best_score > 100:
            cx, cy, r = best_circle
            angle = compute_coaster_orientation(gray_mat, int(cx), int(cy), int(r))
            full_cx = cx + mx
            full_cy = cy + my
            return True, (float(full_cx), float(full_cy)), float(r), angle

    return False, None, None, None


def compute_coaster_orientation(
    gray: np.ndarray, cx: int, cy: int, radius: int
) -> float | None:
    """
    Compute coaster orientation from its pattern using gradient analysis.

    The coaster has a radially symmetric pattern, so we use gradient direction
    histogram to find the dominant orientation.

    Args:
        gray: Grayscale image of mat region
        cx, cy: Center of coaster in mat coordinates
        radius: Radius of coaster

    Returns:
        Orientation angle in degrees, or None if cannot be determined
    """
    h, w = gray.shape

    # Extract coaster region with some padding
    pad = 5
    y1, y2 = max(0, cy - radius - pad), min(h, cy + radius + pad)
    x1, x2 = max(0, cx - radius - pad), min(w, cx + radius + pad)

    if y2 - y1 < 20 or x2 - x1 < 20:
        return None

    coaster_region = gray[y1:y2, x1:x2].astype(np.float32)

    # Compute gradients
    gx = cv2.Sobel(coaster_region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(coaster_region, cv2.CV_32F, 0, 1, ksize=3)

    # Compute gradient magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)  # in radians

    # Create circular mask
    local_cy = cy - y1
    local_cx = cx - x1
    Y, X = np.ogrid[: coaster_region.shape[0], : coaster_region.shape[1]]
    dist = np.sqrt((X - local_cx) ** 2 + (Y - local_cy) ** 2)
    circular_mask = (dist < radius * 0.9) & (dist > radius * 0.3)

    if np.sum(circular_mask) < 100:
        return None

    # Weight directions by magnitude and create histogram
    masked_directions = direction[circular_mask]
    masked_magnitudes = magnitude[circular_mask]

    # Create histogram of gradient directions (weighted by magnitude)
    num_bins = 36
    hist, bin_edges = np.histogram(
        masked_directions,
        bins=num_bins,
        range=(-np.pi, np.pi),
        weights=masked_magnitudes,
    )

    # Find dominant direction
    dominant_bin = np.argmax(hist)
    dominant_angle_rad = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
    dominant_angle_deg = np.degrees(dominant_angle_rad)

    return dominant_angle_deg


def process_frame(
    image: np.ndarray, episode_idx: int, debug: bool = False
) -> DetectionResult:
    """
    Process a single frame to detect mat, cube, and coaster.

    Args:
        image: RGB image (HWC format)
        episode_idx: Episode index for tracking
        debug: If True, show debug visualizations

    Returns:
        DetectionResult with all detection information
    """
    result = DetectionResult(episode_idx=episode_idx)

    # Ensure proper format
    image = _ensure_uint8(image)

    # Detect mat first (helps with cube and coaster detection)
    mat_detected, mat_bbox, mat_corners, mat_angle = detect_mat(image)
    result.mat_detected = mat_detected
    result.mat_bbox = mat_bbox
    result.mat_corners = mat_corners
    result.mat_angle = mat_angle

    # Detect cube (pass mat_bbox to help filter candidates)
    cube_detected, cube_center, cube_angle, cube_rect = detect_cube(image, mat_bbox)
    result.cube_detected = cube_detected
    result.cube_center = cube_center
    result.cube_angle = cube_angle
    result.cube_bbox = cube_rect

    # Detect coaster (pass mat_bbox to focus search area)
    coaster_detected, coaster_center, coaster_radius, coaster_angle = detect_coaster(
        image, mat_bbox
    )
    result.coaster_detected = coaster_detected
    result.coaster_center = coaster_center
    result.coaster_radius = coaster_radius
    result.coaster_angle = coaster_angle

    if debug:
        debug_image = image.copy()

        # Draw mat bbox
        if mat_bbox:
            x, y, w, h = mat_bbox
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw cube
        if cube_rect:
            box = cv2.boxPoints(cube_rect)
            box = np.int32(box)
            cv2.drawContours(debug_image, [box], 0, (0, 255, 0), 2)
            if cube_center:
                cv2.circle(
                    debug_image,
                    (int(cube_center[0]), int(cube_center[1])),
                    5,
                    (0, 255, 0),
                    -1,
                )

        # Draw coaster
        if coaster_center and coaster_radius:
            cv2.circle(
                debug_image,
                (int(coaster_center[0]), int(coaster_center[1])),
                int(coaster_radius),
                (0, 0, 255),
                2,
            )

        plt.figure(figsize=(10, 8))
        plt.imshow(debug_image)
        plt.title(f"Episode {episode_idx} Detection")
        plt.show()

    return result


def load_frame_from_episode(
    dataset: LeRobotDataset,
    ds_meta: LeRobotDatasetMetadata,
    episode_idx: int,
    frame_offset: int = 4,  # 5th frame (0-indexed)
    camera_name: str = "top",
) -> np.ndarray | None:
    """
    Load a specific frame from an episode.

    Args:
        dataset: LeRobotDataset instance
        ds_meta: Dataset metadata
        episode_idx: Episode index
        frame_offset: Frame offset within episode (default 4 = 5th frame)
        camera_name: Camera to load from

    Returns:
        Image as numpy array (HWC, RGB) or None if failed
    """
    # Get episode bounds
    from_idx = ds_meta.episodes["dataset_from_index"][episode_idx]
    to_idx = ds_meta.episodes["dataset_to_index"][episode_idx]
    from_idx = int(from_idx.item() if hasattr(from_idx, "item") else from_idx)
    to_idx = int(to_idx.item() if hasattr(to_idx, "item") else to_idx)

    num_frames = to_idx - from_idx
    if frame_offset >= num_frames:
        logger.warning(
            f"Episode {episode_idx} has only {num_frames} frames, "
            f"requested frame {frame_offset}"
        )
        frame_offset = min(frame_offset, num_frames - 1)

    global_idx = from_idx + frame_offset

    try:
        sample = dataset[global_idx]
        image_key = f"observation.images.{camera_name}"

        if image_key not in sample:
            logger.warning(f"Camera {camera_name} not found in episode {episode_idx}")
            return None

        img = sample[image_key]
        img_np = _to_numpy_hwc(img)
        return _ensure_uint8(img_np)

    except Exception as e:
        logger.warning(f"Failed to load frame from episode {episode_idx}: {e}")
        return None


def visualize_initial_poses(
    repo_id: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    frame_offset: int = 4,
    camera_name: str = "top",
    output_path: Path | None = None,
    debug_episodes: list[int] | None = None,
    exclude_ranges: list[tuple[int, int]] | None = None,
) -> None:
    """
    Visualize initial poses from all episodes.

    Args:
        repo_id: HuggingFace repo_id
        dataset_root: Root directory containing datasets
        frame_offset: Frame offset to extract (default 4 = 5th frame)
        camera_name: Camera to analyze
        output_path: Optional output path for plot
        debug_episodes: Optional list of episode indices to show debug visualizations
        exclude_ranges: List of (start, end) tuples for episode ranges to exclude
    """
    # Load dataset
    dataset_path = dataset_root / repo_id
    if dataset_path.exists():
        logger.info(f"Found local dataset at {dataset_path}")
        ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_path)
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path, episodes=None)
    else:
        logger.info("Will stream from HuggingFace Hub")
        ds_meta = LeRobotDatasetMetadata(repo_id)
        dataset = LeRobotDataset(repo_id=repo_id, episodes=None)

    num_episodes = len(ds_meta.episodes["dataset_from_index"])
    logger.info(f"Processing {num_episodes} episodes...")

    # Build set of excluded episodes
    excluded_episodes: set[int] = set()
    if exclude_ranges:
        for start, end in exclude_ranges:
            excluded_episodes.update(range(start, end + 1))
        logger.info(f"Excluding {len(excluded_episodes)} episodes: {exclude_ranges}")

    # Process all episodes
    results: list[DetectionResult] = []
    failed_episodes: list[int] = []
    processed_count = 0

    for episode_idx in range(num_episodes):
        # Skip excluded episodes
        if episode_idx in excluded_episodes:
            continue

        processed_count += 1
        if processed_count % 20 == 0 or processed_count == 1:
            print(
                f"\rProcessing episodes... {processed_count}/{num_episodes - len(excluded_episodes)}",
                end="",
                flush=True,
            )

        # Load frame
        image = load_frame_from_episode(
            dataset, ds_meta, episode_idx, frame_offset, camera_name
        )

        if image is None:
            failed_episodes.append(episode_idx)
            results.append(DetectionResult(episode_idx=episode_idx))
            continue

        # Process frame
        debug = debug_episodes is not None and episode_idx in debug_episodes
        result = process_frame(image, episode_idx, debug=debug)
        results.append(result)

        # Track failures
        if not result.cube_detected:
            failed_episodes.append(episode_idx)

    print(
        f"\rProcessing episodes... {processed_count}/{num_episodes - len(excluded_episodes)}"
    )

    # Report failures
    cube_failures = [r.episode_idx for r in results if not r.cube_detected]
    coaster_failures = [r.episode_idx for r in results if not r.coaster_detected]
    mat_failures = [r.episode_idx for r in results if not r.mat_detected]

    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total episodes processed: {len(results)}")
    print(f"Episodes excluded: {len(excluded_episodes)}")
    print(
        f"Cube detection failures ({len(cube_failures)}): {cube_failures[:20]}{'...' if len(cube_failures) > 20 else ''}"
    )
    print(
        f"Coaster detection failures ({len(coaster_failures)}): {coaster_failures[:20]}{'...' if len(coaster_failures) > 20 else ''}"
    )
    print(
        f"Mat detection failures ({len(mat_failures)}): {mat_failures[:20]}{'...' if len(mat_failures) > 20 else ''}"
    )
    print("=" * 60)

    # Create visualization with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # === Left plot: Object positions ===
    ax1 = axes[0]

    # Get reference mat bbox (from first successful detection)
    ref_mat_bbox = None
    for r in results:
        if r.mat_bbox:
            ref_mat_bbox = r.mat_bbox
            break

    # Plot mat boundary
    if ref_mat_bbox:
        x, y, w, h = ref_mat_bbox
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=2,
            label="Mat boundary (ref)",
        )
        ax1.add_patch(rect)

    # Collect cube positions and angles
    cube_positions = []
    cube_angles = []

    for r in results:
        if r.cube_detected and r.cube_center:
            cube_positions.append(r.cube_center)
            cube_angles.append(r.cube_angle if r.cube_angle else 0)

    # Plot cube positions with orientation arrows
    if cube_positions:
        positions = np.array(cube_positions)
        angles = np.array(cube_angles)

        # Convert angles to radians and compute arrow directions
        angles_rad = np.deg2rad(angles)
        arrow_length = 15  # pixels

        # Plot scatter points
        ax1.scatter(
            positions[:, 0],
            positions[:, 1],
            c="green",
            s=50,
            alpha=0.7,
            label=f"Cube positions (n={len(positions)})",
            zorder=3,
        )

        # Plot orientation arrows
        for pos, angle_rad in zip(positions, angles_rad):
            dx = arrow_length * np.cos(angle_rad)
            dy = arrow_length * np.sin(angle_rad)
            ax1.arrow(
                pos[0],
                pos[1],
                dx,
                dy,
                head_width=5,
                head_length=3,
                fc="darkgreen",
                ec="darkgreen",
                alpha=0.6,
                zorder=2,
            )

    # Collect and plot coaster positions with orientation
    coaster_positions = []
    coaster_angles = []
    for r in results:
        if r.coaster_detected and r.coaster_center:
            coaster_positions.append(r.coaster_center)
            coaster_angles.append(r.coaster_angle if r.coaster_angle else 0)

    if coaster_positions:
        positions = np.array(coaster_positions)
        angles = np.array(coaster_angles)
        angles_rad = np.deg2rad(angles)

        ax1.scatter(
            positions[:, 0],
            positions[:, 1],
            c="brown",
            s=100,
            marker="o",
            alpha=0.7,
            label=f"Coaster positions (n={len(positions)})",
            zorder=3,
        )

        # Plot coaster orientation arrows
        arrow_length = 20
        for pos, angle_rad in zip(positions, angles_rad):
            dx = arrow_length * np.cos(angle_rad)
            dy = arrow_length * np.sin(angle_rad)
            ax1.arrow(
                pos[0],
                pos[1],
                dx,
                dy,
                head_width=6,
                head_length=4,
                fc="darkred",
                ec="darkred",
                alpha=0.5,
                zorder=2,
            )

    # Formatting for left plot
    ax1.set_xlabel("X Position (pixels)", fontsize=12)
    ax1.set_ylabel("Y Position (pixels)", fontsize=12)
    ax1.set_title(
        f"Initial Object Poses\nFrame {frame_offset + 1}, {len(results)} episodes",
        fontsize=14,
    )
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()

    # === Right plot: Mat positions and orientations over time ===
    ax2 = axes[1]

    # Collect mat corners and plot each mat's edges
    mat_count = 0
    mat_angles = []

    # Use colormap for episode progression
    cmap = plt.cm.viridis

    for i, r in enumerate(results):
        if r.mat_corners is not None:
            mat_count += 1
            # Normalize color by episode index for progression visualization
            color = cmap(i / len(results))

            # Plot mat edges (4 corners connected)
            corners = r.mat_corners
            # Close the polygon by adding first corner at the end
            corners_closed = np.vstack([corners, corners[0]])

            ax2.plot(
                corners_closed[:, 0],
                corners_closed[:, 1],
                color=color,
                alpha=0.3,
                linewidth=1,
            )

            # Store angle for statistics
            if r.mat_angle is not None:
                mat_angles.append(r.mat_angle)

    # Add colorbar to show episode progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(results)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="Episode Index", shrink=0.8)

    # Compute and display mat statistics
    if mat_angles:
        mean_angle = np.mean(mat_angles)
        std_angle = np.std(mat_angles)
        angle_range = np.max(mat_angles) - np.min(mat_angles)

        stats_text = (
            f"Mat Statistics:\n"
            f"Detected: {mat_count}/{len(results)}\n"
            f"Angle mean: {mean_angle:.2f}°\n"
            f"Angle std: {std_angle:.2f}°\n"
            f"Angle range: {angle_range:.2f}°"
        )
        ax2.text(
            0.02,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Formatting for right plot
    ax2.set_xlabel("X Position (pixels)", fontsize=12)
    ax2.set_ylabel("Y Position (pixels)", fontsize=12)
    ax2.set_title(
        "Mat Positions Over Time\n(edges show mat boundaries per episode)",
        fontsize=14,
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_aspect("equal")
    ax2.invert_yaxis()

    plt.suptitle(
        f"Dataset: {repo_id}\nCamera: {camera_name}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")
    else:
        default_path = Path("initial_poses_visualization.png")
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {default_path}")

    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize initial poses from dataset using classical CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="giacomoran/so101_data_collection_cube_hand_guided",
        help="HuggingFace repo_id",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=4,
        help="Frame offset within episode (0-indexed, default 4 = 5th frame)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="top",
        help="Camera to analyze (top, wrist)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for visualization plot",
    )
    parser.add_argument(
        "--debug-episodes",
        type=int,
        nargs="+",
        help="Episode indices to show debug visualizations",
    )
    parser.add_argument(
        "--exclude-ranges",
        type=str,
        nargs="+",
        default=["80-100", "180-200"],
        help="Episode ranges to exclude (format: start-end). Default: 80-100 180-200",
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="Process all episodes without exclusions",
    )

    return parser.parse_args()


def parse_exclude_ranges(range_strs: list[str]) -> list[tuple[int, int]]:
    """Parse exclude range strings like '80-100' into tuples."""
    ranges = []
    for s in range_strs:
        if "-" in s:
            parts = s.split("-")
            ranges.append((int(parts[0]), int(parts[1])))
        else:
            # Single episode
            ep = int(s)
            ranges.append((ep, ep))
    return ranges


def main() -> None:
    args = parse_args()

    # Parse exclude ranges
    exclude_ranges = None
    if not args.no_exclude and args.exclude_ranges:
        exclude_ranges = parse_exclude_ranges(args.exclude_ranges)

    visualize_initial_poses(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        frame_offset=args.frame_offset,
        camera_name=args.camera,
        output_path=args.output,
        debug_episodes=args.debug_episodes,
        exclude_ranges=exclude_ranges,
    )


if __name__ == "__main__":
    main()
