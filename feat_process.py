import sys
import logging

import numpy as np
import cv2 as cv

from typing import List, Optional

from feat_detectors import FEAT_DETECTORS
from feat_descriptors import FEAT_DESCRIPTORS_LSH, FEAT_DESCRIPTORS_KDTREE
from feat_matchers import match_features, match_features_in_scene
from feat_storage import FeatureStorage

log: logging.Logger = logging.getLogger(__name__)


def read_image(filename: str, scale: float = None) -> np.ndarray:
    # Use cv.imread(filename, cv.IMREAD_GRAYSCALE) to read images as grayscale
    # Note that LUCID descriptor does not work with grayscale images
    image = cv.imread(filename)

    if image is None or image.size == 0:
        log.error(f"Error reading image filename {filename}")
        sys.exit(-1)

    if scale is not None:
        scaled_width = int(image.shape[1] * scale)
        scaled_height = int(image.shape[0] * scale)
        scaled_dims = (scaled_width, scaled_height)

        # cv.INTER_LINEAR is faster and less accurate than cv.INTER_CUBIC
        interpolation_flag = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
        resized = cv.resize(image, scaled_dims, interpolation=interpolation_flag)
        return resized

    return image


def read_mask(filename: str):
    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    if image is None or image.size == 0:
        log.error(f"Error reading image filename {filename} - set mask as empty")
        return None

    return image


def filter_keypoints(
    key_points: List[cv.KeyPoint], image: np.ndarray
) -> List[cv.KeyPoint]:
    filtered_keypoints: List[cv.KeyPoint] = []
    for key_point in key_points:
        if (
            key_point.pt[0] < 0
            or key_point.pt[1] < 0
            or key_point.pt[0] > image.shape[1] - 1
            or key_point.pt[1] > image.shape[0] - 1
        ):
            continue
        filtered_keypoints.append(key_point)

    return filtered_keypoints


def compute_keypoints(
    image: np.ndarray,
    feat_detector: cv.Feature2D,
    image_type: str,
    detector_name: str,
    mask_image: np.ndarray = None,
) -> List[cv.KeyPoint]:

    keypoint_storage = FeatureStorage(
        image_type=image_type, detector_name=detector_name
    )

    key_points: Optional[List[cv.KeyPoint]] = None

    # Uncomment to enable reading from storage file
    # key_points = keypoint_storage.read_keypoints()

    if key_points is None:
        log.debug(f"Computing keypoints for {image_type} image using {detector_name}")
        key_points = feat_detector.detect(image, mask_image)

    # Draw keypoints
    if log.getEffectiveLevel() <= logging.DEBUG:
        keypoints_image = image.copy()
        cv.drawKeypoints(
            image,
            key_points,
            keypoints_image,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        features_window_name = f"Features {image_type} {detector_name}"
        cv.namedWindow(features_window_name, cv.WINDOW_NORMAL)
        cv.imshow(features_window_name, keypoints_image)

        while True:
            print(
                "Press ESC to continue, 's' to save image , 'k' to export keypoints data"
            )
            ch = cv.waitKey(0)
            # Press ESC to continue
            if ch == 27:  # ESC key
                break
            # Press 's' to save keypoints image
            if ch == ord("s"):
                keypoints_image_name = f"Keypoints_{image_type}_{detector_name}.tiff"
                cv.imwrite(keypoints_image_name, keypoints_image)
                log.info(f"Image saved: {keypoints_image_name}")
            # Save keypoints info
            if ch == ord("k"):
                keypoint_storage.write_keypoints(key_points)

        cv.destroyWindow(features_window_name)

    return filter_keypoints(key_points, image)


def match_query_in_scene(
    query_image: np.ndarray,
    scene_image: np.ndarray,
    scene_image_filename: str,
    query_mask_image: np.ndarray = None,
    query_contour: np.ndarray = None,
):
    if query_contour is None:
        if query_mask_image is not None:
            contours, _ = cv.findContours(
                image=query_mask_image,
                mode=cv.RETR_EXTERNAL,
                method=cv.CHAIN_APPROX_NONE,
            )
            if contours is not None:
                query_contour = np.squeeze(contours[0]).astype(np.float32)

        if query_contour is None:
            query_contour = np.array(
                [
                    [0, 0],
                    [query_image.shape[1], 0],
                    [query_image.shape[1], query_image.shape[0]],
                    [0, query_image.shape[0]],
                ],
                dtype="float32",
            )
    # Iterate through available detectors
    for detector_name, feat_detector in FEAT_DETECTORS.items():
        key_points_query = compute_keypoints(
            query_image.copy(), feat_detector, "Query", detector_name, query_mask_image
        )

        key_points_scene = compute_keypoints(
            scene_image.copy(), feat_detector, "Scene", detector_name
        )
        # Iterate through available descriptors
        for descriptor_name, feat_descriptor in FEAT_DESCRIPTORS_LSH.items():
            match_features(
                feat_descriptor,
                query_image.copy(),
                key_points_query,
                scene_image.copy(),
                key_points_scene,
                detector_name,
                descriptor_name,
                scene_image_filename,
                query_mask_image,
                query_contour,
            )

        for descriptor_name, feat_descriptor in FEAT_DESCRIPTORS_KDTREE.items():
            match_features(
                feat_descriptor,
                query_image.copy(),
                key_points_query,
                scene_image.copy(),
                key_points_scene,
                detector_name,
                descriptor_name,
                scene_image_filename,
                query_mask_image,
                query_contour,
            )

        # Special case for KAZE descriptor
        if detector_name == "KAZE":
            match_features(
                cv.KAZE_create(),
                query_image.copy(),
                key_points_query,
                scene_image.copy(),
                key_points_scene,
                detector_name,
                detector_name,
                scene_image_filename,
                query_mask_image,
                query_contour,
            )

        # Special case for AKAZE descriptor
        if detector_name == "AKAZE":
            match_features(
                cv.AKAZE_create(),
                query_image.copy(),
                key_points_query,
                scene_image.copy(),
                key_points_scene,
                detector_name,
                detector_name,
                scene_image_filename,
                query_mask_image,
                query_contour,
            )


def match_query_features_in_scene(
    feature_storage: FeatureStorage, scene_image: np.ndarray, scene_image_filename: str
):
    detector_name = feature_storage.get_detector_name()
    log.debug(f"Feature detector: {detector_name}")

    feat_detector = FEAT_DETECTORS[detector_name]

    # TODO: Check if data read is valid
    key_points_query = feature_storage.read_keypoints()
    descriptors_query = feature_storage.read_descriptors()

    key_points_scene = compute_keypoints(
        scene_image.copy(), feat_detector, "Scene", detector_name
    )

    descriptor_name = feature_storage.get_descriptor_name()
    log.debug(f"Feature descriptor: {descriptor_name}")

    if descriptor_name in FEAT_DESCRIPTORS_LSH:
        feat_descriptor = FEAT_DESCRIPTORS_LSH[descriptor_name]
    elif descriptor_name in FEAT_DESCRIPTORS_KDTREE:
        feat_descriptor = FEAT_DESCRIPTORS_KDTREE[descriptor_name]
    else:
        log.error(f"Unsupported descriptor: {descriptor_name}")
        return

    match_features_in_scene(
        feat_descriptor,
        key_points_query,
        descriptors_query,
        scene_image.copy(),
        key_points_scene,
        detector_name,
        descriptor_name,
        scene_image_filename,
        None,
    )
