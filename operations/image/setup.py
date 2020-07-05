import os
import importlib.util
from setuptools import setup

# Boilerplate to load commonalities
spec = importlib.util.spec_from_file_location(
    "setup_common", os.path.join(os.path.dirname(__file__), "setup_common.py")
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

common.KWARGS["entry_points"] = {
    "dffml.operation": [
        f"resize = {common.IMPORT_NAME}.operations:resize",
        f"flatten = {common.IMPORT_NAME}.operations:flatten",
        f"calcHist = {common.IMPORT_NAME}.operations:calcHist",
        f"HuMoments = {common.IMPORT_NAME}.operations:HuMoments",
        f"Haralick = {common.IMPORT_NAME}.operations:Haralick",
        f"normalize = {common.IMPORT_NAME}.operations:normalize",
        f"convert_color = {common.IMPORT_NAME}.operations:convert_color",
        f"zernike_moments = {common.IMPORT_NAME}.operations:zernike_moments",
        f"threshold = {common.IMPORT_NAME}.operations:threshold",
        f"adaptiveThreshold = {common.IMPORT_NAME}.operations:adaptiveThreshold",
        f"findContours = {common.IMPORT_NAME}.operations:findContours",
        f"inRange = {common.IMPORT_NAME}.operations:inRange",
        f"boxFilter = {common.IMPORT_NAME}.operations:boxFilter",
        f"filter2D = {common.IMPORT_NAME}.operations:filter2D",
        f"GaussianBlur = {common.IMPORT_NAME}.operations:GaussianBlur",
        f"medianBlur = {common.IMPORT_NAME}.operations:medianBlur",
        f"bilateralFilter = {common.IMPORT_NAME}.operations:bilateralBlur",
        f"sepFilter2D = {common.IMPORT_NAME}.operations:sepFilter2D",
        f"morph = {common.IMPORT_NAME}.operations:morph",
        f"get_kernel = {common.IMPORT_NAME}.operations:get_kernel",
        f"Sobel = {common.IMPORT_NAME}.operations:Sobel",
        f"Laplacian = {common.IMPORT_NAME}.operations:Laplacian",
        f"Scharr = {common.IMPORT_NAME}.operations:Scharr",
        f"equalizeHist = {common.IMPORT_NAME}.operations:equalizeHist",
        f"calcBackProject = {common.IMPORT_NAME}.operations:calcBackProject",
        f"hog = {common.IMPORT_NAME}.operations:hog",
        # f"CLAHE = {common.IMPORT_NAME}.operations:CLAHE",
        # f"KAZE = {common.IMPORT_NAME}.algorithms:KAZE",
        # f"ORB = {common.IMPORT_NAME}.algorithms:ORB",
        # f"HOG = {common.IMPORT_NAME}.algorithms:HOG",
        # f"BRISK = {common.IMPORT_NAME}.algorithms:BRISK",
        # f"AKAZE = {common.IMPORT_NAME}.algorithms:AKAZE",
        # f"SURF = {common.IMPORT_NAME}.algorithms:SURF",
        # f"SIFT = {common.IMPORT_NAME}.algorithms:SIFT",
    ]
}

setup(**common.KWARGS)
