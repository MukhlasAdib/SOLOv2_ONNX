import argparse
import os
import shutil
import time
from typing import Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from utils.data_processing import resize_and_pad, solov2_preprocess
from utils.output_processing import format_results, show_result


class ORTRunner:
    def __init__(self, model_path: str) -> None:
        test_model = onnx.load(model_path)
        onnx.checker.check_model(test_model)  # type: ignore

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
        self.input_size = tuple(self.sess.get_inputs()[0].shape[-2:])
        sess_input_type = self.sess.get_inputs()[0].type
        if sess_input_type == "tensor(float16)":
            self.input_type = np.float16
        else:
            self.input_type = np.float32

    def infer(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = self.preprocess(img)
        labels, masks, scores = self.sess.run(
            ("labels", "masks", "scores"), {"images": img}
        )
        return labels, masks, scores

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = solov2_preprocess(img, self.input_size)
        return img.astype(self.input_type)


def main(model_path: str, input_dir: str, output_dir: str) -> None:
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    image_files = os.listdir(input_dir)
    image_files = [f for f in image_files if f.endswith(".jpg") or f.endswith(".png")]

    model = ORTRunner(model_path)
    times = []
    for f in image_files:
        img = cv2.imread(os.path.join(input_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = time.perf_counter()
        labels, masks, scores = model.infer(img)
        times.append(time.perf_counter() - start)
        formatted = format_results(scores, labels, masks)

        vis = resize_and_pad(img, model.input_size)
        vis = show_result(vis, formatted)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_dir, f), vis)
    avg_time = np.mean(times)
    print(f"Average latency: {avg_time} s")


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX.")
    parser.add_argument("--onnx", help="path to the ONNX model")
    parser.add_argument("--inputs", help="path to folder containing input images")
    parser.add_argument("--results", help="folder to save the results")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.onnx, args.inputs, args.results)
