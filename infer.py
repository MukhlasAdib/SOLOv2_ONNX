import itertools
import os
import shutil
from typing import Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from mmdet.core.visualization.image import imshow_det_bboxes

INFER_SIZE = (800, 800)  # H x W
NORM_MEAN = [123.675, 116.28, 103.53]
NORM_STD = [58.395, 57.12, 57.375]

MODEL_PATH = "output.onnx"
IMAGE_PATH = "mmdetection/demo"
OUPUT_DIR = "results"


def format_results(
    scores: np.ndarray, labels: np.ndarray, masks: np.ndarray, num_classes: int = 81
):
    """Format inference results into bboxes and masks.
    Taken from https://github.com/open-mmlab/mmdetection/blob/v2.25.1/mmdet/models/detectors/single_stage_instance_seg.py
    """
    labels = labels.astype(int)
    mask_results = [[] for _ in range(num_classes)]

    num_masks = len(scores)
    if num_masks == 0:
        bbox_results = [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)]
        return bbox_results, mask_results

    # create dummy bbox results to store the scores
    det_bboxes = np.zeros((len(scores), 5))
    det_bboxes[:, 4] = scores
    bbox_results = [det_bboxes[labels == i, :] for i in range(num_classes)]
    for idx in range(num_masks):
        mask = masks[idx]
        mask_results[labels[idx]].append(mask)

    return bbox_results, mask_results


def show_result(
    img,
    result,
    score_thr=0.3,
    bbox_color=(72, 101, 241),
    text_color=(72, 101, 241),
    mask_color=None,
    thickness=2,
    font_size=13,
    win_name="",
    show=False,
    wait_time=0,
    out_file=None,
):
    """Visualize bboxes and masks.
    Taken from https://github.com/open-mmlab/mmdetection/blob/v2.25.1/mmdet/models/detectors/single_stage_instance_seg.py
    """
    assert isinstance(result, tuple)
    bbox_result, mask_result = result
    bboxes = np.vstack(bbox_result)
    img = img.copy()
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    if len(labels) == 0:
        bboxes = np.zeros([0, 5])
        masks = np.zeros([0, 0, 0])
    # draw segmentation masks
    else:
        masks = list(itertools.chain(*mask_result))
        masks = np.stack(masks, axis=0)
        # dummy bboxes
        if bboxes[:, :4].sum() == 0:
            num_masks = len(bboxes)
            x_any = masks.any(axis=1)
            y_any = masks.any(axis=2)
            for idx in range(num_masks):
                x = np.where(x_any[idx, :])[0]
                y = np.where(y_any[idx, :])[0]
                if len(x) > 0 and len(y) > 0:
                    bboxes[idx, :4] = np.array(
                        [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32
                    )
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        masks,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file,
    )

    if not (show or out_file):
        return img


def resize_and_pad(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    old_size = img.shape[:2]
    if old_size[0] > old_size[1]:
        ratio = target_size[0] / old_size[0]
    else:
        ratio = target_size[1] / old_size[1]
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [125, 125, 125]
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img


class ORTRunner:
    def __init__(self, model_path: str) -> None:
        test_model = onnx.load(model_path)
        onnx.checker.check_model(test_model)

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        self.sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def infer(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = self.preprocess(img)
        labels, masks, scores = self.sess.run(
            ("labels", "masks", "scores"), {"images": img}
        )
        return labels, masks, scores

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = resize_and_pad(img, INFER_SIZE)
        img = (img - NORM_MEAN) / NORM_STD
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)


def main():
    if os.path.isdir(OUPUT_DIR):
        shutil.rmtree(OUPUT_DIR)
    os.makedirs(OUPUT_DIR, exist_ok=True)
    image_files = os.listdir(IMAGE_PATH)
    image_files = [f for f in image_files if f.endswith(".jpg") or f.endswith(".png")]

    model = ORTRunner(MODEL_PATH)
    for f in image_files:
        img = cv2.imread(os.path.join(IMAGE_PATH, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels, masks, scores = model.infer(img)
        formatted = format_results(scores, labels, masks)

        vis = resize_and_pad(img, INFER_SIZE)
        vis = show_result(vis, formatted)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(OUPUT_DIR, f), vis)


if __name__ == "__main__":
    main()
