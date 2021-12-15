import cv2 as cv
import numpy as np

import logging

from typing import List, Callable, Optional, Tuple

from feat_descriptors import FEAT_DESCRIPTORS_LSH, FEAT_DESCRIPTORS_KDTREE
from feat_storage import FeatureStorage

# Minimal distance between transformed points
MIN_DIST_TRANSFORMED_POINTS = 20.0

# Minimal area size for transformed points
MIN_SIZE_TRANSFORMED_AREA = 20.0

# Invalid result value for matching transformation error
INVALID_RESULT = 1000.0

log: logging.Logger = logging.getLogger(__name__)


def flann_kdtree_matcher(
    descriptors_query: np.ndarray, descriptors_scene: np.ndarray
) -> List[List[cv.DMatch]]:
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)

    # Set number of searches. Higher is better, but takes longer
    search_params = dict(checks=100)

    # Initialize matches
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Find matches
    matches = flann.knnMatch(descriptors_query, descriptors_scene, k=2)

    return matches


def flann_lsh_matcher(
    descriptors_query: np.ndarray, descriptors_scene: np.ndarray
) -> List[List[cv.DMatch]]:
    flann_index_lsh = 6
    index_params = dict(
        algorithm=flann_index_lsh, table_number=6, key_size=12, multi_probe_level=1
    )

    # Set number of searches. Higher is better, but takes longer
    search_params = dict(checks=100)

    # Initialize matches
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Find matches
    matches = flann.knnMatch(descriptors_query, descriptors_scene, k=2)

    return matches


def draw_matches(
    query_image: np.ndarray,
    key_points_query: List[cv.KeyPoint],
    scene_image: np.ndarray,
    key_points_scene: List[cv.KeyPoint],
    matches: List[List[cv.DMatch]],
    detector_name: str,
    descriptor_name: str,
    storage_callback: Callable,
):
    if query_image is None:
        return

    # Disable drawing for usual work
    if log.getEffectiveLevel() > logging.DEBUG:
        return

    matches_image = cv.drawMatchesKnn(
        query_image,
        key_points_query,
        scene_image,
        key_points_scene,
        matches,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )

    # Draw matches
    matches_window_name = f"Matches {detector_name}({descriptor_name})"
    cv.namedWindow(matches_window_name, cv.WINDOW_NORMAL)
    cv.imshow(matches_window_name, matches_image)

    while True:
        print("Press ESC to continue, 's' to save image , 'k' to export features data")
        ch = cv.waitKey(0)
        if ch == 27:  # ESC key
            break
        # Press 's' to save matched keypoints image
        if ch == ord("s"):
            matches_image_name = f"Matches_{detector_name}_{descriptor_name}.tiff"
            cv.imwrite(matches_image_name, matches_image)
            log.info(f"Image saved: {matches_image_name}")
        # Save keypoints info
        if ch == ord("k"):
            storage_callback()

    cv.destroyWindow(matches_window_name)


def check_transformation(
    persp_mat: np.ndarray,
    query_contour: np.array,
    scene_image_size: List[int],
    detector_name: str,
    descriptor_name: str,
):
    # Check for negative or zero scaling
    if persp_mat[0][0] < 0 or persp_mat[1][1] < 0:
        log.warning(
            f"Non positive scale: {persp_mat[0][0]} {persp_mat[1][1]} in {detector_name}({descriptor_name})"
        )
        return False

    # TODO: Consider checking for maximal and minimal scale
    # This could be related to the descriptor capabilities

    # Check if the transformed query corner points are outside image borders
    warped_to_scene_contour = cv.perspectiveTransform(
        np.array([query_contour]), persp_mat
    )

    # Points are pixels in image so they should be integer
    warped_to_scene_contour = warped_to_scene_contour[0].astype(np.int32)

    lower_left_scene_corner = np.array([0, 0])
    upper_right_scene_corner = np.array([scene_image_size[1], scene_image_size[0]])

    in_scene_indices = np.all(
        np.logical_and(
            lower_left_scene_corner <= warped_to_scene_contour,
            warped_to_scene_contour <= upper_right_scene_corner,
        ),
        axis=1,
    )
    in_scene_query_points = warped_to_scene_contour[in_scene_indices]

    if in_scene_query_points.size < warped_to_scene_contour.size:
        log.warning(
            f"Transformed point(s) outside scene in {detector_name}({descriptor_name})"
        )
        return False

    # Check for points too close
    for i in range(in_scene_query_points.shape[0]):
        elem_lhs = in_scene_query_points[i, :]
        for j in range(i + 1, in_scene_query_points.shape[0]):
            elem_rhs = in_scene_query_points[j, :]
            # TODO: Consider checking for max distance
            # This could be related to the descriptor capabilities
            if np.linalg.norm(elem_lhs - elem_rhs) < MIN_DIST_TRANSFORMED_POINTS:
                log.warning(
                    f"Transformed point(s) too close in {detector_name}({descriptor_name})"
                )
                return False

    # TODO: Consider checking for max area size
    # Check for points projected on a line or on a single point
    _, area_size, _ = cv.minAreaRect(in_scene_query_points)
    if (
        area_size[0] < MIN_SIZE_TRANSFORMED_AREA
        or area_size[1] < MIN_SIZE_TRANSFORMED_AREA
    ):
        log.warning(f"Transformed area to line in {detector_name}({descriptor_name})")
        return False

    return True


def calculate_transformation(
    matches: List[List[cv.DMatch]],
    key_points_query: List[cv.KeyPoint],
    key_points_scene: List[cv.KeyPoint],
    query_image_size: List[int],
    scene_image_size: List[int],
    detector_name: str,
    descriptor_name: str,
    query_contour: np.array,
) -> (np.ndarray, np.array):
    # Select points to calculate homography
    query_points = []
    scene_points = []
    for match in matches:
        if len(match) == 1:
            query_points.append(key_points_query[match[0].queryIdx].pt)
            scene_points.append(key_points_scene[match[0].trainIdx].pt)
            continue
        if len(match) < 2:
            log.warning(f"Not enough knnMatches in {detector_name}({descriptor_name})")
            continue
        # Ratio test as per Lowe's SIFT paper
        if match[0].distance >= 0.7 * match[1].distance:
            continue
        query_points.append(key_points_query[match[0].queryIdx].pt)
        scene_points.append(key_points_scene[match[0].trainIdx].pt)

    if len(query_points) < 4:
        log.warning(
            f"Not enough points to calculate homography in {detector_name}({descriptor_name})"
        )
        return None, None

    # Calculate homography
    # The inlier mask of findHomography is ignored
    # Available OpenCV algorithms: RANSAC, LMEDS, RHO
    persp_transform_matrix, _ = cv.findHomography(
        np.array(query_points), np.array(scene_points), cv.RANSAC
    )

    if persp_transform_matrix is None:
        log.warning(f"Empty perspective matrix in {detector_name}({descriptor_name})")
        return None, None

    if query_contour is None:
        # TODO: Fix case for empty query_image_size
        # The whole query image will be considered
        query_contour = np.array(
            [
                [0, 0],
                [0, query_image_size[0]],
                [query_image_size[1], query_image_size[0]],
                [query_image_size[1], 0],
            ],
            dtype="float32",
        )

    # Sanity check for the perspective transformation
    bound_rect = cv.minAreaRect(query_contour)
    bound_contour = cv.boxPoints(bound_rect)
    if not check_transformation(
        persp_transform_matrix,
        bound_contour,
        scene_image_size,
        detector_name,
        descriptor_name,
    ):
        log.warning(f"Invalid perspective matrix in {detector_name}({descriptor_name})")
        return None, None

    query_contour = np.array([query_contour])
    warped_to_scene_contour = cv.perspectiveTransform(
        query_contour, persp_transform_matrix
    )

    return persp_transform_matrix, warped_to_scene_contour


def calculate_matching_accuracy(
    persp_transform_matrix: Optional[np.ndarray],
    warped_to_scene_contour: Optional[np.array],
    query_image: np.ndarray,
    scene_image: np.ndarray,
    mask_image: np.ndarray,
    detector_name: str,
    descriptor_name: str,
    query_contour: np.array,
) -> float:
    if persp_transform_matrix is None:
        log.error("Empty perspective transformation matrix")
        return INVALID_RESULT

    if query_image is None:
        log.error("Empty query image")
        return INVALID_RESULT

    # Warp scene to query
    warp_back_image = cv.warpPerspective(
        scene_image,
        persp_transform_matrix,
        (query_image.shape[1], query_image.shape[0]),
        flags=cv.WARP_INVERSE_MAP,
    )
    query_rect = cv.boundingRect(query_contour)

    # Calculate difference between query and back warped to query
    result = cv.norm(
        query_image, warp_back_image, cv.NORM_RELATIVE + cv.NORM_L2, mask=mask_image
    )

    # Visualize warping
    if log.getEffectiveLevel() <= logging.INFO:
        if warped_to_scene_contour is None:
            # TODO: Investigate this case
            log.warning("No query contour to visualize")
            return INVALID_RESULT

        warped_window_name = f"Warped To Scene {detector_name}({descriptor_name})"
        warp_image = scene_image.copy()
        cv.drawContours(
            warp_image,
            np.array([warped_to_scene_contour]).astype(np.int32),
            0,
            (0, 0, 255),
            6,
        )
        cv.namedWindow(warped_window_name, cv.WINDOW_NORMAL)
        cv.imshow(warped_window_name, warp_image)

        back_warped_window_name = (
            f"Warped From Scene {detector_name}({descriptor_name})"
        )
        cv.namedWindow(back_warped_window_name, cv.WINDOW_NORMAL)
        warp_back_image_roi = warp_back_image[
            query_rect[1] : query_rect[1] + query_rect[3],
            query_rect[0] : query_rect[0] + query_rect[2],
            ...,
        ]
        cv.imshow(back_warped_window_name, warp_back_image_roi)

        log.info(f"{detector_name}({descriptor_name}) Matching difference {result}")

        while True:
            print("Press ESC to continue, 's' to save images")
            ch = cv.waitKey(0)
            if ch == 27:  # ESC key
                break
            # Press 's' to save warped and back warped images
            if ch == ord("s"):
                warp_image_name = (
                    f"WarpedToScene_{detector_name}_{descriptor_name}.tiff"
                )
                cv.imwrite(warp_image_name, warp_image)
                log.info(f"Image saved: {warp_image_name}")
                back_warp_image_name = (
                    f"WarpedFromScene_{detector_name}_{descriptor_name}.tiff"
                )
                cv.imwrite(back_warp_image_name, warp_back_image_roi)
                log.info(f"Image saved: {back_warp_image_name}")

        cv.destroyWindow(warped_window_name)
        cv.destroyWindow(back_warped_window_name)

    return result


def match_descriptors(
    feat_descriptor: cv.Feature2D,
    descriptors_query: np.ndarray,
    descriptors_scene: np.ndarray,
    detector_name: str,
) -> Optional[List[List[cv.DMatch]]]:
    if feat_descriptor in FEAT_DESCRIPTORS_LSH.values():
        matches = flann_lsh_matcher(descriptors_query, descriptors_scene)
    elif feat_descriptor in FEAT_DESCRIPTORS_KDTREE.values():
        matches = flann_kdtree_matcher(descriptors_query, descriptors_scene)
    elif detector_name == "KAZE":
        matches = flann_kdtree_matcher(descriptors_query, descriptors_scene)
    elif detector_name == "AKAZE":
        matches = flann_lsh_matcher(descriptors_query, descriptors_scene)
    else:
        return None

    return matches


def prepare_descriptors(
    key_points: List[cv.KeyPoint],
    feat_descriptor: cv.Feature2D,
    image: np.ndarray,
    descriptor_storage: FeatureStorage,
) -> Tuple[Optional[List[cv.KeyPoint]], Optional[np.ndarray]]:
    descriptors: Optional[np.ndarray] = None
    stored_key_points: Optional[List[cv.KeyPoint]] = None

    # Uncomment to enable reading from storage file
    # descriptors = descriptor_storage.read_descriptors()
    # stored_key_points = descriptor_storage.read_keypoints()

    if descriptors is not None and stored_key_points is not None:
        return stored_key_points, descriptors

    return feat_descriptor.compute(image, key_points)


def match_features(
    feat_descriptor: cv.Feature2D,
    query_image: np.ndarray,
    key_points_query: List[cv.KeyPoint],
    scene_image: np.ndarray,
    key_points_scene: List[cv.KeyPoint],
    detector_name: str,
    descriptor_name: str,
    scene_image_filename: str,
    mask_image: np.ndarray,
    query_contour: np.array,
):
    if descriptor_name == "LUCID" and len(scene_image.shape) < 3:
        log.error(
            f"LUCID does not work with grayscale images {detector_name}({descriptor_name})"
        )
        return

    # TODO: Expose "Query" and "Scene" literals
    query_descriptor_storage = FeatureStorage(
        image_type="Query", detector_name=detector_name, descriptor_name=descriptor_name
    )

    scene_descriptor_storage = FeatureStorage(
        image_type="Scene", detector_name=detector_name, descriptor_name=descriptor_name
    )

    key_points_query, descriptors_query = prepare_descriptors(
        key_points_query, feat_descriptor, query_image, query_descriptor_storage
    )

    key_points_scene, descriptors_scene = prepare_descriptors(
        key_points_scene, feat_descriptor, scene_image, scene_descriptor_storage
    )

    matches = match_descriptors(
        feat_descriptor, descriptors_query, descriptors_scene, detector_name
    )
    if not matches:
        log.error(f"Matcher not set for descriptor {detector_name}({descriptor_name})")
        return

    def storage_callback():
        query_descriptor_storage.write_keypoints(key_points_query, descriptors_query)
        scene_descriptor_storage.write_keypoints(key_points_scene, descriptors_scene)

    draw_matches(
        query_image,
        key_points_query,
        scene_image,
        key_points_scene,
        matches,
        detector_name,
        descriptor_name,
        storage_callback,
    )

    persp_transform_matrix, warped_to_scene_contour = calculate_transformation(
        matches,
        key_points_query,
        key_points_scene,
        [query_image.shape[0], query_image.shape[1]],
        [scene_image.shape[0], scene_image.shape[1]],
        detector_name,
        descriptor_name,
        query_contour,
    )

    matching_accuracy = calculate_matching_accuracy(
        persp_transform_matrix,
        warped_to_scene_contour,
        query_image,
        scene_image,
        mask_image,
        detector_name,
        descriptor_name,
        query_contour,
    )

    # Display information for matching accuracy
    print(scene_image_filename, detector_name, descriptor_name, matching_accuracy)


def match_features_in_scene(
    feat_descriptor: cv.Feature2D,
    key_points_query: List[cv.KeyPoint],
    descriptors_query: np.ndarray,
    scene_image: np.ndarray,
    key_points_scene: List[cv.KeyPoint],
    detector_name: str,
    descriptor_name: str,
    scene_image_filename: str,
    mask_image: Optional[np.ndarray],
):
    if descriptor_name == "LUCID" and len(scene_image.shape) < 3:
        log.error(
            f"LUCID does not work with grayscale images {detector_name}({descriptor_name})"
        )
        return

    # TODO: Expose "Query" and "Scene" literals
    scene_descriptor_storage = FeatureStorage(
        image_type="Scene", detector_name=detector_name, descriptor_name=descriptor_name
    )

    key_points_scene, descriptors_scene = prepare_descriptors(
        key_points_scene, feat_descriptor, scene_image, scene_descriptor_storage
    )

    matches = match_descriptors(
        feat_descriptor, descriptors_query, descriptors_scene, detector_name
    )
    if not matches:
        log.error(f"Matcher not set for descriptor {detector_name}({descriptor_name})")
        return

    # Calculate the minimal area rectangle for the feature points
    bounding_contour_query = cv.convexHull(
        np.array([point.pt for point in key_points_query], dtype="float32")
    )
    min_area_rect = cv.minAreaRect(bounding_contour_query)
    box_points = cv.boxPoints(min_area_rect)
    query_contour = np.array(np.int0(box_points), dtype="float32")

    query_image = None
    query_image_shape = None
    if log.getEffectiveLevel() <= logging.DEBUG:
        # Generate query image to be used for visualization
        query_min_x = int(np.min(query_contour[:, 1]))
        query_min_y = int(np.min(query_contour[:, 0]))
        query_max_x = int(np.max(query_contour[:, 1]))
        query_max_y = int(np.max(query_contour[:, 0]))
        query_image = np.zeros((query_max_x, query_max_y, 3), np.uint8)
        query_image[query_min_x:query_max_x, query_min_y:query_max_y, :] = 255
        query_image_shape = [query_image.shape[0], query_image.shape[1]]

    def storage_callback():
        scene_descriptor_storage.write_keypoints(key_points_scene, descriptors_scene)

    draw_matches(
        query_image,
        key_points_query,
        scene_image,
        key_points_scene,
        matches,
        detector_name,
        descriptor_name,
        storage_callback,
    )

    persp_transform_matrix, warped_to_scene_contour = calculate_transformation(
        matches,
        key_points_query,
        key_points_scene,
        query_image_shape,
        [scene_image.shape[0], scene_image.shape[1]],
        detector_name,
        descriptor_name,
        query_contour,
    )

    if query_image is None:
        # TODO: Consider better printing
        print(
            scene_image_filename, detector_name, descriptor_name, persp_transform_matrix
        )
        return

    matching_accuracy = calculate_matching_accuracy(
        persp_transform_matrix,
        warped_to_scene_contour,
        query_image,
        scene_image,
        mask_image,
        detector_name,
        descriptor_name,
        query_contour,
    )

    # Display information for matching accuracy
    print(scene_image_filename, detector_name, descriptor_name, matching_accuracy)
