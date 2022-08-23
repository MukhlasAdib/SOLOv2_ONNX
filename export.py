"""
Converter for SOLOv2 model into ONNX format.

This script is based on the mmdetection repository
https://github.com/open-mmlab/mmdetection
"""

import argparse

import cv2
import torch

from mmdet.apis import init_detector

from utils.data_processing import solov2_preprocess
from utils.patches import mock_get_results_single


torch.no_grad()


def validate_data(infer_size):
    assert len(infer_size) == 2, "Image size must be in format of (H, W)"
    assert (
        infer_size[0] % 32 == 0 and infer_size[1] % 32 == 0
    ), "Image size must be divisible by 32"


def generate_inputs(image_path, target_size=(800, 800)):
    """Generate input data
    target_size is in H x W"""
    img = cv2.imread(image_path)
    img = solov2_preprocess(img, target_size)
    img = torch.from_numpy(img)
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
    validate_data(input_size)
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
