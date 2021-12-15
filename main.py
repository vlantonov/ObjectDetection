import sys
import logging
import argparse

from feat_process import (
    read_image,
    read_mask,
    match_query_in_scene,
    match_query_features_in_scene,
)
from feat_storage import FeatureStorage

# filename='log.txt'
# Set to DEBUG to print the feature detection and matching
# Set to INFO to print the transformation warp result
# Set to FATAL to print the final result only
logging.basicConfig(level=logging.FATAL)

log: logging.Logger = logging.getLogger(__name__)

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-q", "--query", help="Query object name", type=str, required=False
    )
    parser.add_argument(
        "-m", "--mask", help="Mask image name", type=str, required=False
    )
    parser.add_argument(
        "-d", "--desc", help="Query descriptors name", type=str, required=False
    )
    parser.add_argument(
        "-s", "--scene", help="Scene image name", type=str, required=True
    )
    parser.add_argument(
        "-r", "--rescale", help="Rescale ratio of scene", type=float, required=False
    )
    args = parser.parse_args()

    # Prepare scene image processing
    scene_image_filename = args.scene
    log.info(f"Scene image filename {scene_image_filename}")

    scene_rescale = args.rescale
    if scene_rescale is not None:
        scale_argument = scene_rescale

        if scene_rescale < 0:
            print(f"Scale {scale_argument} is not positive")
            sys.exit(-1)

        log.info(f"Scene image rescale {scene_rescale}")

    # Read scene image
    scene_image = read_image(scene_image_filename, scene_rescale)

    # If query descriptors are available then query image will not be used
    query_descriptors_filename = args.desc
    if query_descriptors_filename is not None:
        log.info(f"Query descriptors filename {query_descriptors_filename}")
        query_features = FeatureStorage(data_file=query_descriptors_filename)
        match_query_features_in_scene(query_features, scene_image, scene_image_filename)
        sys.exit(0)

    # Prepare query image processing
    query_image_filename = args.query
    if query_image_filename is None:
        log.info("Query image must be provided when no descriptor data is available")
        sys.exit(-1)

    log.info(f"Query image filename {query_image_filename}")

    # Read query image and mask
    query_image = read_image(query_image_filename)
    query_mask_image = read_mask(args.mask)
    if (
        query_mask_image is not None
        and query_image.shape[:2] != query_mask_image.shape[:2]
    ):
        print(
            f"Mask size {query_mask_image.shape[:2]} != query image size {query_image.shape[:2]}"
        )
        sys.exit(-1)

    # Frame of tote visualization
    query_contour = None

    match_query_in_scene(
        query_image, scene_image, scene_image_filename, query_mask_image, query_contour
    )
