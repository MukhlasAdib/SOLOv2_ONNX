from typing import Tuple

import cv2
import numpy as np

NORM_MEAN = [123.675, 116.28, 103.53]
NORM_STD = [58.395, 57.12, 57.375]


def resize_and_pad(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """resize and pad image to the target size

    Args:
        img (np.ndarray): input original image in HWC
        target_size (Tuple[int, int]): inference size

    Returns:
        np.ndarray: processed image
    """
    # Resize and pad
    target_h, target_w = target_size
    ori_h, ori_w = img.shape[:2]
    ratio_w = target_w / ori_w
    ratio_h = target_h / ori_h

    # Follow output size of the side with smaller ratio
    if ratio_w < ratio_h:
        output_w = target_w
        output_h = int(ratio_w * ori_h)
        left = right = 0
        top = int((target_h - output_h) / 2)
        bottom = target_h - output_h - top
    else:
        output_h = target_h
        output_w = int(ratio_h * ori_w)
        left = int((target_w - output_w) / 2)
        right = target_w - output_w - left
        top = bottom = 0

    img = cv2.resize(img, (output_w, output_h))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[128, 128, 128]
    )
    return img


def solov2_preprocess(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess image for SOLOv2

    Args:
        img (np.ndarray): input original image in HWC
        target_size (Tuple[int, int]): inference size

    Returns:
        np.ndarray: preprocessed image aformatted in NCHW
    """
    img = resize_and_pad(img, target_size)
    img = img - NORM_MEAN
    img = img / NORM_STD

    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, 0)
    return img.astype(np.float32)
