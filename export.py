"""
Converter for SOLOv2 model into ONNX format.

This script is based on the mmdetection repository
https://github.com/open-mmlab/mmdetection
"""

import argparse

import cv2
import numpy as np
import torch

from mmdet.apis import init_detector

from patches import mock_get_results_single

torch.no_grad()


def generate_inputs(image_path, target_size=(800, 800)):
    """Generate input data
    target_size is in H x W"""
    norm_mean = [123.675, 116.28, 103.53]
    norm_std = [58.395, 57.12, 57.375]
    img = cv2.imread(image_path)

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

    img = img - norm_mean
    img = img / norm_std

    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img.astype(np.float32))
    img_meta = {"ori_shape": (*target_size, 3), "img_shape": (*target_size, 3)}

    return img, [img_meta]


def create_model(config_path, checkpoint_path, img_metas):
    model = init_detector(
        config_path,
        checkpoint_path,
        device="cpu",
    )

    def mock_get_results_single_wrapper(*args, **kwargs):
        return mock_get_results_single(model.mask_head, *args, **kwargs)

    model.mask_head._get_results_single = mock_get_results_single_wrapper

    def _forward(img):
        ret_backbone = model.backbone(img)
        ret_neck = model.neck(ret_backbone)
        ret_mask_head = model.mask_head.simple_test(
            ret_neck, img_metas, rescale=False, instances_list=None
        )
        return ret_mask_head[0]

    model.forward = _forward
    return model


def main(image_path, config_path, checkpoint_path, output_path, input_size):
    img, img_metas = generate_inputs(image_path, input_size)
    model = create_model(config_path, checkpoint_path, img_metas)
    torch.onnx.export(
        model,
        img,
        output_path,
        opset_version=14,
        input_names=["images"],
        output_names=["masks", "labels", "scores"],
        dynamic_axes=None,
    )

    print()
    print(f"Image metas: {img_metas}")
    print("DONE")


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX.")
    parser.add_argument("--cfg", help="model config path")
    parser.add_argument("--ckpt", help="model config path")
    parser.add_argument("--img", help="path to one test image")
    parser.add_argument("--out", help="path to the onnx output")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[800, 800],
        help="image size (h, w), divisible by 32",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.img, args.cfg, args.ckpt, args.out, tuple(args.imgsz))
