from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .preprocessing import read_image


def compute_orb_keypoints(image_path: str | Path, n_features: int = 1000):
    """Compute ORB keypoints and descriptors for one image."""
    image_bgr = read_image(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    return orb.detectAndCompute(image_gray, None)


def count_good_orb_matches(query_descriptors, candidate_descriptors, ratio: float = 0.75) -> int:
    """Count good ORB matches using Lowe's ratio test."""
    if query_descriptors is None or candidate_descriptors is None:
        return 0

    if len(query_descriptors) < 2 or len(candidate_descriptors) < 2:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)

    good_matches = 0
    for pair in matches:
        if len(pair) != 2:
            continue

        best_match, second_best_match = pair
        if best_match.distance < ratio * second_best_match.distance:
            good_matches += 1

    return good_matches


def count_homography_inliers(
    query_keypoints,
    query_descriptors,
    candidate_keypoints,
    candidate_descriptors,
    ratio: float = 0.75,
) -> int:
    """Count geometrically consistent matches using homography RANSAC."""
    if query_descriptors is None or candidate_descriptors is None:
        return 0

    if len(query_descriptors) < 4 or len(candidate_descriptors) < 4:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) != 2:
            continue

        best_match, second_best_match = pair
        if best_match.distance < ratio * second_best_match.distance:
            good_matches.append(best_match)

    if len(good_matches) < 4:
        return len(good_matches)

    query_points = np.float32(
        [query_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    candidate_points = np.float32(
        [candidate_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)

    _, inlier_mask = cv2.findHomography(
        query_points,
        candidate_points,
        cv2.RANSAC,
        5.0,
    )
    if inlier_mask is None:
        return 0

    return int(inlier_mask.sum())
