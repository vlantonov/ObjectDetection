import numpy as np
import logging
import cv2 as cv

from typing import List, Optional, Tuple

from feat_detectors import FEAT_DETECTORS
from feat_descriptors import FEAT_DESCRIPTORS_LSH, FEAT_DESCRIPTORS_KDTREE
from feat_matchers import match_features_in_scene
from feat_process import compute_keypoints
from feat_storage import FeatureStorage

log: logging.Logger = logging.getLogger(__name__)


class ToteDetector:
    def __init__(self, feat_data_file: str):
        self._feat_data_file = feat_data_file
        query_features_storage: FeatureStorage = FeatureStorage(
            data_file=feat_data_file
        )

        self._detector_name: str = query_features_storage.get_detector_name()
        self._feat_detector: cv.Feature2D = FEAT_DETECTORS[self._detector_name]
        log.debug(f"Feature detector: {self._detector_name}")

        # TODO: Check if data read is valid
        self._key_points_query: List[
            cv.KeyPoint
        ] = query_features_storage.read_keypoints()
        self._descriptors_query: np.ndarray = query_features_storage.read_descriptors()

        descriptor_name: str = query_features_storage.get_descriptor_name()
        log.debug(f"Feature descriptor: {descriptor_name}")

        if descriptor_name in FEAT_DESCRIPTORS_LSH:
            self._feat_descriptor = FEAT_DESCRIPTORS_LSH[descriptor_name]
        elif descriptor_name in FEAT_DESCRIPTORS_KDTREE:
            self._feat_descriptor = FEAT_DESCRIPTORS_KDTREE[descriptor_name]
        else:
            raise ValueError(f"Unsupported descriptor: {descriptor_name}")

        self._descriptor_name: str = descriptor_name

        self._tote_roi: Optional[Tuple[int, int, int, int]] = None

    def detect(self, img: np.ndarray) -> np.ndarray:
        # Copying is required as certain detectors seem to modify the image
        key_points_scene: List[cv.KeyPoint] = compute_keypoints(
            img.copy(), self._feat_detector, "Scene", self._detector_name
        )

        self._tote_roi: Optional[Tuple[int, int, int, int]] = match_features_in_scene(
            self._feat_descriptor,
            self._key_points_query,
            self._descriptors_query,
            img.copy(),
            key_points_scene,
            self._detector_name,
            self._descriptor_name,
            self._feat_data_file,
            None,
        )

        return img

    def get_tote_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._tote_roi
