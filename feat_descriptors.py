import cv2 as cv

from typing import Dict

FEAT_DESCRIPTORS_LSH: Dict[str, cv.Feature2D] = {
    "ORB": cv.ORB_create(),
    "BRISK": cv.BRISK_create(),
    "BoostDesc": cv.xfeatures2d.BoostDesc_create(),
    "BRIEF": cv.xfeatures2d.BriefDescriptorExtractor_create(),
    "FREAK": cv.xfeatures2d.FREAK_create(),
    "LATCH": cv.xfeatures2d.LATCH_create(),
    "LUCID": cv.xfeatures2d.LUCID_create(),
}

FEAT_DESCRIPTORS_KDTREE: Dict[str, cv.Feature2D] = {
    "DAISY": cv.xfeatures2d.DAISY_create(),
    "VGG": cv.xfeatures2d.VGG_create(),
}
