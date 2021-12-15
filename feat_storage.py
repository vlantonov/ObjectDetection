from string import Template
import os.path

import cv2 as cv
import numpy as np

import logging
from typing import List, Optional

log: logging.Logger = logging.getLogger(__name__)


class FeatureStorage:
    _PREFIX: str = "kp"
    _DETECTOR_KEY: str = "detector"
    _IMAGE_TYPE_KEY: str = "image_type"
    _KEYPOINT_COUNT_KEY: str = _PREFIX + "_count"
    _ANGLE_KEY: Template = Template(_PREFIX + "_${i}_angle")
    _CLASS_ID_KEY: Template = Template(_PREFIX + "_${i}_class_id")
    _OCTAVE_KEY: Template = Template(_PREFIX + "_${i}_octave")
    _POINT_X_KEY: Template = Template(_PREFIX + "_${i}_point_x")
    _POINT_Y_KEY: Template = Template(_PREFIX + "_${i}_point_y")
    _RESPONSE_KEY: Template = Template(_PREFIX + "_${i}_response")
    _SIZE_KEY: Template = Template(_PREFIX + "_${i}_size")
    _DESCRIPTOR_KEY: str = "descriptor"
    _DESCRIPTORS_KEY: str = "descriptors"

    def __init__(self, **kwargs):

        if "data_file" in kwargs:
            self.__set_file_name(kwargs["data_file"])
            return

        image_type: str = kwargs["image_type"]
        detector_name: str = kwargs["detector_name"]

        self._image_type: str = image_type
        self._detector_name: str = detector_name
        self._descriptor_name: Optional[str] = None

        if "descriptor_name" in kwargs:
            descriptor_name: str = kwargs["descriptor_name"]
            self._descriptor_name = descriptor_name
            self._keypoints_data_file: str = (
                f"Descriptors_{image_type}_{detector_name}_{descriptor_name}.ocv"
            )
        else:
            self._keypoints_data_file: str = (
                f"Keypoints_{image_type}_{detector_name}.ocv"
            )

    def read_keypoints(self) -> Optional[List[cv.KeyPoint]]:
        if not os.path.isfile(self._keypoints_data_file):
            log.debug(f"Keypoints file {self._keypoints_data_file} not found")
            return None

        log.debug(f"Reading keypoints from file {self._keypoints_data_file}")

        # TODO: Common logic for reading
        keypoints_reader = cv.FileStorage(
            self._keypoints_data_file, cv.FileStorage_READ
        )
        if not keypoints_reader.isOpened():
            log.error(f"Error opening file {self._keypoints_data_file}")
            return None

        # Check file data for detector
        detector: str = keypoints_reader.getNode(self._DETECTOR_KEY).string()
        if self._detector_name != detector:
            log.debug(
                f"Detector type mismatch in {self._keypoints_data_file}: \
                  Expected: {self._detector_name}  Read: {detector}"
            )
            return None

        # Check file data for image type
        image_type: str = keypoints_reader.getNode(self._IMAGE_TYPE_KEY).string()
        if self._image_type != image_type:
            log.debug(
                f"Image type mismatch in {self._keypoints_data_file}: Expected: {self._image_type}  Read: {image_type}"
            )
            return None

        kp_count: int = int(keypoints_reader.getNode(self._KEYPOINT_COUNT_KEY).real())
        log.debug(f"Reading {kp_count} keypoints from {self._keypoints_data_file}")

        key_points: List[cv.KeyPoint] = []
        for i in range(kp_count):
            keypoint = cv.KeyPoint()
            keypoint.angle = keypoints_reader.getNode(
                self._ANGLE_KEY.substitute(i=str(i))
            ).real()
            keypoint.class_id = int(
                keypoints_reader.getNode(self._CLASS_ID_KEY.substitute(i=str(i))).real()
            )
            keypoint.octave = int(
                keypoints_reader.getNode(self._OCTAVE_KEY.substitute(i=str(i))).real()
            )
            keypoint.pt = (
                keypoints_reader.getNode(self._POINT_X_KEY.substitute(i=str(i))).real(),
                keypoints_reader.getNode(self._POINT_Y_KEY.substitute(i=str(i))).real(),
            )
            keypoint.response = keypoints_reader.getNode(
                self._RESPONSE_KEY.substitute(i=str(i))
            ).real()
            keypoint.size = keypoints_reader.getNode(
                self._SIZE_KEY.substitute(i=str(i))
            ).real()
            # TODO: Check for error reading keypoint data
            # print(keypoints_reader.getNode(self._SIZE_KEY.substitute(i=str(i))).empty())
            # print(keypoints_reader.getNode(self._SIZE_KEY.substitute(i=str(i))).type())
            key_points.append(keypoint)

        keypoints_reader.release()

        return key_points

    def read_descriptors(self) -> Optional[np.ndarray]:
        if not os.path.isfile(self._keypoints_data_file):
            log.debug(f"Descriptors file {self._keypoints_data_file} not found")
            return None

        log.debug(f"Reading descriptors from file {self._keypoints_data_file}")

        # TODO: Common logic for reading
        keypoints_reader = cv.FileStorage(
            self._keypoints_data_file, cv.FileStorage_READ
        )
        if not keypoints_reader.isOpened():
            log.error(f"Error opening file {self._keypoints_data_file}")
            return None

        # Check file data for detector
        detector = keypoints_reader.getNode(self._DETECTOR_KEY).string()
        if self._detector_name != detector:
            log.debug(
                f"Detector type mismatch in {self._keypoints_data_file}:  \
                  Expected: {self._detector_name}  Read: {detector}"
            )
            return None

        # Check file data for image type
        image_type = keypoints_reader.getNode(self._IMAGE_TYPE_KEY).string()
        if self._image_type != image_type:
            log.debug(
                f"Image type mismatch in {self._keypoints_data_file}: Expected: {self._image_type}  Read: {image_type}"
            )
            return None

        # TODO: Descriptor specific check
        descriptor_name = keypoints_reader.getNode(self._DESCRIPTOR_KEY).string()
        if self._descriptor_name != descriptor_name:
            log.debug(
                f"Descriptor type mismatch in {self._keypoints_data_file}: \
                  Expected: {self._descriptor_name}  Read: {descriptor_name}"
            )
            return None

        kp_count = int(keypoints_reader.getNode(self._KEYPOINT_COUNT_KEY).real())
        log.debug(f"Reading {kp_count} keypoints from {self._keypoints_data_file}")

        descriptors = keypoints_reader.getNode("descriptors").mat()
        # TODO: Check descriptor size vs keypoint size
        if kp_count != descriptors.shape[0]:
            log.debug(
                f"File {self._keypoints_data_file}: \
                Descriptor count {descriptors.shape[0]} not equal to keypoint count {kp_count}"
            )
            return None

        return descriptors

    def get_detector_name(self) -> str:
        return self._detector_name

    def get_descriptor_name(self) -> str:
        return self._descriptor_name

    def write_keypoints(
        self, key_points: List[cv.KeyPoint], descriptors: Optional[np.ndarray] = None
    ):
        # TODO: Prepare proper handling of these cases
        if self._descriptor_name is None:
            if descriptors is not None:
                log.warning(
                    "Missing descriptor name - descriptors argument data ignored!"
                )
                return
        else:
            if descriptors is None:
                log.warning("Missing descriptors argument data!")
                return

        # TODO: Check descriptor size vs keypoint size
        # len(key_points) != descriptors.shape[0]

        log.debug(f"Writing keypoints to file {self._keypoints_data_file}")
        keypoints_writer = cv.FileStorage(
            self._keypoints_data_file, cv.FileStorage_WRITE
        )

        if not keypoints_writer.isOpened():
            log.error(f"Error opening file {self._keypoints_data_file}")
            return

        keypoints_writer.write(self._DETECTOR_KEY, self._detector_name)  # str
        log.debug(
            f"File: {self._keypoints_data_file}  Detector type: {self._detector_name}"
        )

        keypoints_writer.write(self._IMAGE_TYPE_KEY, self._image_type)  # str
        log.debug(f"File: {self._keypoints_data_file}  Image type: {self._image_type}")

        keypoints_writer.write(self._KEYPOINT_COUNT_KEY, len(key_points))  # int
        log.debug(f"Saving {len(key_points)} keypoints to {self._keypoints_data_file}")

        for i, keypoint in enumerate(key_points):
            keypoints_writer.write(
                self._ANGLE_KEY.substitute(i=str(i)), keypoint.angle
            )  # float
            keypoints_writer.write(
                self._CLASS_ID_KEY.substitute(i=str(i)), keypoint.class_id
            )  # int
            keypoints_writer.write(
                self._OCTAVE_KEY.substitute(i=str(i)), keypoint.octave
            )  # int
            keypoints_writer.write(
                self._POINT_X_KEY.substitute(i=str(i)), keypoint.pt[0]
            )  # float
            keypoints_writer.write(
                self._POINT_Y_KEY.substitute(i=str(i)), keypoint.pt[1]
            )  # float
            keypoints_writer.write(
                self._RESPONSE_KEY.substitute(i=str(i)), keypoint.response
            )  # float
            keypoints_writer.write(
                self._SIZE_KEY.substitute(i=str(i)), keypoint.size
            )  # float

        keypoints_writer.write(self._DESCRIPTOR_KEY, self._descriptor_name)  # str
        log.debug(
            f"File: {self._keypoints_data_file}  Descriptor type: {self._descriptor_name}"
        )

        if descriptors is not None:
            log.debug(
                f"Saving {descriptors.shape[0]} descriptors to {self._keypoints_data_file}"
            )
            keypoints_writer.write(self._DESCRIPTORS_KEY, descriptors)  # cv::Mat

        keypoints_writer.release()

    def __set_file_name(self, keypoints_data_file: str):
        # Attempt to read file
        keypoints_reader = cv.FileStorage(keypoints_data_file, cv.FileStorage_READ)
        if not os.path.isfile(keypoints_data_file):
            log.debug(f"Keypoints file {keypoints_data_file} not found")
            raise ValueError(f"Keypoints file {keypoints_data_file} not found")

        if not keypoints_reader.isOpened():
            raise ValueError(f"Error opening file {self._keypoints_data_file}")

        self._detector_name = keypoints_reader.getNode(self._DETECTOR_KEY).string()
        if not self._detector_name:
            raise ValueError(f"Error reading detector name")

        # Check file data for image type
        self._image_type = keypoints_reader.getNode(self._IMAGE_TYPE_KEY).string()
        if not self._image_type:
            raise ValueError(f"Error reading image type")

        self._descriptor_name = keypoints_reader.getNode(self._DESCRIPTOR_KEY).string()
        if not self._descriptor_name:
            log.debug(f"Keypoints file name set to {keypoints_data_file}")
        else:
            log.debug(f"Descriptors file set to  {keypoints_data_file}")

        self._keypoints_data_file = keypoints_data_file
