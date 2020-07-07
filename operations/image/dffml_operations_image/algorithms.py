from typing import List
import cv2
import numpy as np
import skimage.feature

from dffml.df.base import op


@op
async def HOG(
    image: List[int],
    orientations: int = None,
    pixels_per_cell: List[int] = None,
    cells_per_block: List[int] = None,
    block_norm: str = None,
    visualize: bool = None,
    transform_sqrt: bool = None,
    feature_vector: bool = None,
    multichannel: bool = None,
) -> List[int]:
    """
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        pixels_per_cell = tuple(pixels_per_cell)
        cells_per_block = tuple(cells_per_block)
    except cv2.error:
        pass

    parameters = {k: v for k, v in locals().items() if v is not None}
    hog = skimage.feature.hog(**parameters)

    return hog


@op
async def KAZE(
    image: List[int],
    extended: bool = None,
    upright: bool = None,
    threshold: float = None,
    nOctaves: int = None,
    nOctaveLayers: int = None,
    diffusivity: int = None,
) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}
    parameters.pop("image", None)

    kaze_features = cv2.KAZE_create(**parameters)
    keypoints = kaze_features.detect(image)
    keypoints, descriptors = kaze_features.compute(image, keypoints)
    return descriptors.flatten()


@op
async def ORB(
    image: List[int],
    mask: List[int] = None,
    nfeatures: int = None,
    scaleFactor: float = None,
    nlevels: int = None,
    edgeThreshold: int = None,
    firstLevel: int = None,
    WTA_K: int = None,
    scoreType: int = None,
    patchSize: int = None,
    fastThreshold: int = None,
) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}
    parameters.pop("mask", None)
    parameters.pop("image", None)

    orb_features = cv2.ORB_create(**parameters)
    keypoints = orb_features.detect(image, mask)
    keypoints, descriptors = orb_features.compute(image, keypoints)
    return descriptors.flatten()
