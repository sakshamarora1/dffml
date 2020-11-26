from typing import List

import cv2
import numpy as np

from dffml.df.base import op

NUM = 0


@op
def get_l_channel(src: List[int]) -> List[int]:
    return cv2.split(src)[0]


@op
def get_ab_channel(src: List[int]) -> List[int]:
    l, a, b = cv2.split(src)
    return cv2.merge([a, b])


@op
def merge_lab(gray: List[int], predicted: List[int]) -> List[int]:
    ab = predicted.squeeze(0).detach().cpu().numpy()
    ab = np.rollaxis(ab, 0, 3)
    return cv2.merge([gray, (ab * 255).astype(np.uint8)])


@op
def display(image: List[int]) -> None:
    cv2.imshow(image)
    cv2.waitKey(0)


@op
def save(image: List[int], directory: str, name: str) -> None:
    # cv2.imsave(directory + "/" + "ColoredImage" + NUM + ".jpg", image)
    cv2.imsave(directory + "/" + name + ".jpg", image)
