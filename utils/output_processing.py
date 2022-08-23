import itertools

import numpy as np

from mmdet.core.visualization.image import imshow_det_bboxes


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
        score_thr=score_thr,  # type: ignore
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
