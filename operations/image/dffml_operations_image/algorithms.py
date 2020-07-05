from typing import List
import cv2
import numpy as np

from dffml.df.base import op


@op
async def SIFT(
    image: List[int],
    mask: List[int] = None,
    nfeatures: int = None,
    nOctaveLayers: int = None,
    contrastThreshold: float = None,
    edgeThreshold: float = None,
    sigma: float = None,
) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}
    parameters.pop("mask", None)
    parameters.pop("image", None)

    sift_features = cv2.xfeatures2d.SIFT_create(**parameters)
    keypoints = sift_features.detect(image, mask)
    keypoints, descriptors = sift_features.compute(image, keypoints)
    return descriptors.flatten()


# @op
# async def HOG(
#     image: List[int],
#     winSize: List[int] = None,  # (64,64)
#     blockSize: List[int] = None,  # (16,16)
#     blockStride: List[int] = None,  # (8,8)
#     cellSize: List[int] = None,  # (8,8)
#     nbins: int = None,  # 9
#     derivAperture: int = None,  # 1
#     winSigma: float = None,  # 4.
#     histogramNormType: int = None,  # 0
#     L2HysThreshold: float = None,  # 2.0000000000000001e-01
#     gammaCorrection: int = None,  # 0
#     nlevels: int = None,  # 64
#     signedGradient: bool = False,
# ) -> List[int]:
#     """
#     """
#     parameters = {k: v for k, v in locals().items() if v is not None}
#     # parameters.pop("mask", None)
#     parameters.pop("image", None)

#     hog = cv2.HOGDescriptor(**parameters)
#     keypoints = hog.detect(image)
#     keypoints, descriptors = hog.compute(image, keypoints)
#     return descriptors.flatten()


@op
async def KAZE(
    image: List[int],
    # mask: List[int] = None,
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
async def SURF():
    pass


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
