from typing import List
import cv2
import mahotas
import numpy as np

from dffml.df.base import op


@op
async def flatten(array: List[int]) -> List[int]:
    return np.array(array).flatten()


@op
async def resize(
    src: List[int],
    dsize: List[int],
    fx: float = None,
    fy: float = None,
    interpolation: int = None,
) -> List[int]:
    """
    Resizes image array to the specified new dimensions

    - If the new dimensions are in 2D, the image is converted to grayscale.

    - To enlarge the image (src dimensions < dsize),
        it will resize the image with INTER_CUBIC interpolation.

    - To shrink the image (src dimensions > dsize),
        it will resize the image with INTER_AREA interpolation
    """
    if len(dsize) == 2:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        dsize = dsize[:2]
    dsize = tuple(dsize)
    if interpolation is None:
        if dsize > src.shape:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA

    parameters = {k: v for k, v in locals().items() if v is not None}

    resized_image = cv2.resize(**parameters)
    return resized_image


@op
async def convert_color(src: List[int], code: str,) -> List[int]:
    """
    Converts images from one color space to another
    """
    # TODO Create a mapping of color conversion names to their integer codes.
    # Reference: https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0

    code = getattr(cv2, "COLOR_" + code.upper())
    return cv2.cvtColor(src, code)


@op
async def calcHist(
    images: List[int],
    channels: List[int],
    mask: List[int],
    histSize: List[int],
    ranges: List[int],
) -> List[int]:
    """
    Calculates a histogram
    """
    return cv2.calcHist(**locals())


@op
async def HuMoments(m: List[int]) -> List[int]:
    """
    Calculates seven Hu invariants
    """
    # If image is not a single channel image convert it
    if len(m.shape) != 2:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = cv2.moments(m)
    hu_moments = cv2.HuMoments(m).flatten()

    return hu_moments


@op
async def Haralick(
    f: List[int],
    ignore_zeros: bool = False,
    preserve_haralick_bug: bool = False,
    compute_14th_feature: bool = False,
    return_mean: bool = False,
    return_mean_ptp: bool = False,
    use_x_minus_y_variance: bool = False,
    distance: int = 1,
) -> List[int]:
    """
    Computes Haralick texture features
    """
    return mahotas.features.haralick(**locals()).mean(axis=0)


@op
async def normalize(
    src: List[int],
    alpha: int = None,
    beta: int = None,
    norm_type: int = None,
    dtype: int = None,
    mask: List[int] = None,
) -> List[int]:
    """
    Normalizes arrays
    """
    src = src.astype("float")
    dst = np.zeros(src.shape)  # Output image array

    parameters = {k: v for k, v in locals().items() if v is not None}

    cv2.normalize(**parameters)
    return dst


@op
async def meanStdDev(image: List[int], mask: List[int] = None) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}

    mean, stdDev = cv2.meanStdDev(**parameters)
    return np.concatenate([mean, stdDev]).flatten()


@op
async def zernike_moments(
    im: List[int], radius: int, degree: int = 8, cm: List[float] = None
) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}

    return mahotas.features.zernike_moments(**parameters)


@op
async def threshold(
    src: List[int], thresh: int, maxval: int, type: int
) -> List[int]:
    """
    """
    return cv2.threshold(**locals())[1]


@op
async def adaptiveThreshold(
    src: List[int],
    maxValue: float,
    adaptiveMethod: int,
    thresholdType: int,
    blockSize: int,
    C: float,
) -> List[int]:
    """
    """
    return cv2.adaptiveThreshold(**locals())


@op
async def findContours(image: List[int], mode: int, method: int) -> List[int]:
    """
    """
    image, contours, hierarchy = cv2.findContours(**locals())
    return contours  # flatten it?


@op
async def boxFilter(
    src: List[int],
    ddepth: int,
    ksize: List[int],
    anchor: List[int] = None,
    normalize: bool = None,
    borderType: int = None,
) -> List[int]:
    """
    """
    ksize = tuple(ksize)
    if anchor:
        anchor = tuple(anchor)
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.boxFilter(**parameters)


@op
async def filter2D(
    src: List[int],
    ddepth: int,
    kernel: List[int],
    anchor: List[int] = None,
    delta: float = None,
    borderType: int = None,
) -> List[int]:
    """
    """
    if anchor:
        anchor = tuple(anchor)
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.filter2D(**parameters)


@op
async def GaussianBlur(
    src: List[int],
    ksize: List[int],
    sigmaX: float,
    sigmaY: float = None,
    borderType: int = None,
) -> List[int]:
    """
    """
    ksize = tuple(ksize)
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.GaussianBlur(**parameters)


@op
async def medianBlur(src: List[int], ksize: int) -> List[int]:
    """
    """
    return cv2.medianBlur(src, ksize)


@op
async def bilateralFilter(
    src: List[int],
    d: int,
    sigmaColor: float,
    sigmaSpace: float,
    borderType: int = None,
) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.bilateralFilter(**parameters)


@op
async def sepFilter2D(
    src: List[int],
    ddepth: int,
    kernelX: List[int],
    kernelY: List[int],
    anchor: List[int] = None,
    delta: float = None,
    borderType: int = None,
) -> List[int]:
    """
    """
    if anchor:
        anchor = tuple(anchor)
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.sepFilter2D(**parameters)


@op
async def morph(
    src: List[int],
    op: str,
    kernel: List[int],
    anchor: List[int] = None,
    iterations: int = None,
    borderType: int = None,
    borderValue: int = None,
) -> List[int]:
    """
    """
    op = getattr(cv2, "COLOR_" + op.upper())
    if anchor:
        anchor = tuple(anchor)
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.morphologyEx(**parameters)


@op
async def get_kernel(
    shape: List[int], ksize: List[int], anchor: List[int] = None
) -> List[int]:
    """
    """
    parameters = {k: tuple(v) for k, v in locals().items() if v is not None}
    print(parameters)
    return cv2.getStructuringElement(**parameters)


@op
async def Sobel(
    src: List[int],
    ddepth: int,
    dx: int,
    dy: int,
    ksize: int = None,
    scale: float = None,
    delta: float = None,
    borderType: int = None,
) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.Sobel(**parameters)


@op
async def Laplacian(
    src: List[int],
    ddepth: int,
    ksize: int,
    scale: float = None,
    delta: float = None,
    borderType: int = None,
) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.Laplacian(**parameters)


@op
async def Scharr(
    src: List[int],
    ddepth: int,
    dx: int,
    dy: int,
    scale: float = None,
    delta: float = None,
    borderType: int = None,
) -> List[int]:
    """
    """
    parameters = {k: v for k, v in locals().items() if v is not None}

    return cv2.Scharr(**parameters)


@op
async def calcBackProject(
    images: List[int],
    channels: List[int],
    hist: List[int],
    ranges: List[int],
    scale: float = 1,
) -> List[int]:
    """
    """
    return cv2.calcBackProject(**locals())


@op
async def equalizeHist(src: List[int]) -> List[int]:
    """
    """
    return cv2.equalizeHist(src)


@op
async def CLAHE(
    image: List[int], clipLimit: float, titleGridSize: List[int]
) -> List[int]:
    """
    """
    clahe = cv2.createCLAHE(clipLimit, tuple(titleGridSize))
    return clahe.apply(image)
