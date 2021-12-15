import cv2 as cv

from typing import Dict

# SimpleBlobDetector parameters
# Need to be set as default ones lead to crash
__blob_detector_params = cv.SimpleBlobDetector_Params()
__blob_detector_params.minThreshold = 1
__blob_detector_params.maxThreshold = 255
__blob_detector_params.filterByArea = True
__blob_detector_params.minArea = 1
__blob_detector_params.filterByCircularity = False
__blob_detector_params.filterByConvexity = False
__blob_detector_params.filterByInertia = False

FEAT_DETECTORS: Dict[str, cv.Feature2D] = {
    "Agast": cv.AgastFeatureDetector_create(),
    "AKAZE": cv.AKAZE_create(),
    "BRISK": cv.BRISK_create(),
    "FAST": cv.FastFeatureDetector_create(),
    "GFTT": cv.GFTTDetector_create(),
    "KAZE": cv.KAZE_create(),
    "MSER": cv.MSER_create(),
    "ORB": cv.ORB_create(),
    "SimpleBlob": cv.SimpleBlobDetector_create(__blob_detector_params),
    "HarrisLaplace": cv.xfeatures2d.HarrisLaplaceFeatureDetector_create(),
    "STAR": cv.xfeatures2d.StarDetector_create(),
}
