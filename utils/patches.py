import torch
import torch.nn.functional as F


def mock_mask_matrix_nms(
    masks,
    labels,
    scores,
    filter_thr=-1,
    nms_pre=-1,
    max_num=-1,
    kernel="gaussian",
    sigma=2.0,
    mask_area=None,
):
    """Exportable Matrix NMS.
    Based on https://github.com/open-mmlab/mmdetection/blob/v2.25.1/mmdet/core/post_processing/matrix_nms.py

    Modifications:
    - change len(tensor) to tensor.shape[0]
    - if return empty logic is replaced by assertion
    """
    assert len(labels) == len(masks) == len(scores)
    assert len(labels) != 0
    if mask_area is None:
        mask_area = masks.sum((1, 2)).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and sort_inds.shape[0] > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = labels.shape[0]
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (
        inter_matrix
        / (expanded_mask_area + expanded_mask_area.transpose(1, 0) - inter_matrix)
    ).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(f"{kernel} kernel is not supported in matrix nms!")
    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return (
                scores.new_zeros(0),
                labels.new_zeros(0),
                masks.new_zeros(0, *masks.shape[-2:]),
                labels.new_zeros(0),
            )
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and sort_inds.shape[0] > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds


class DynamicConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_feats, kernel_preds, stride=1):
        return F.conv2d(mask_feats, kernel_preds, stride=stride)

    @staticmethod
    def symbolic(g, mask_feats, kernel_preds, stride=1):
        return g.op("Conv", mask_feats, kernel_preds)


def mock_get_results_single(
    self, kernel_preds, cls_scores, mask_feats, img_meta, cfg=None
):
    """Exportable _get_results_single method for the SOLOv2 head.
    Based on https://github.com/open-mmlab/mmdetection/blob/v2.25.1/mmdet/models/dense_heads/solov2_head.py

    Modifications:
    - change F.conv2d to an autograd function with explicit symbolic
    - if return empty logic is replaced by assertion
    - do not use InstanceData
    """
    cfg = self.test_cfg if cfg is None else cfg
    assert len(kernel_preds) == len(cls_scores)

    featmap_size = mask_feats.size()[-2:]

    img_shape = img_meta["img_shape"]
    ori_shape = img_meta["ori_shape"]

    # overall info
    h, w, _ = img_shape
    upsampled_size = (
        featmap_size[0] * self.mask_stride,
        featmap_size[1] * self.mask_stride,
    )

    # process.
    score_mask = cls_scores > cfg.score_thr
    cls_scores = cls_scores[score_mask]
    assert len(cls_scores) != 0

    # cate_labels & kernel_preds
    inds = score_mask.nonzero()
    cls_labels = inds[:, 1]
    kernel_preds = kernel_preds[inds[:, 0]]

    # trans vector.
    lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
    strides = kernel_preds.new_ones(lvl_interval[-1])

    strides[: lvl_interval[0]] *= self.strides[0]
    for lvl in range(1, self.num_levels):
        strides[lvl_interval[lvl - 1] : lvl_interval[lvl]] *= self.strides[lvl]
    strides = strides[inds[:, 0]]

    # mask encoding.
    kernel_preds = kernel_preds.view(
        kernel_preds.size(0), -1, self.dynamic_conv_size, self.dynamic_conv_size
    )
    mask_preds = DynamicConv.apply(mask_feats, kernel_preds, 1).squeeze(0).sigmoid()

    # mask.
    masks = mask_preds > cfg.mask_thr
    sum_masks = masks.sum((1, 2)).float()
    keep = sum_masks > strides
    assert keep.sum() != 0
    masks = masks[keep]
    mask_preds = mask_preds[keep]
    sum_masks = sum_masks[keep]
    cls_scores = cls_scores[keep]
    cls_labels = cls_labels[keep]

    # maskness.
    mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
    cls_scores *= mask_scores

    scores, labels, _, keep_inds = mock_mask_matrix_nms(
        masks,
        cls_labels,
        cls_scores,
        mask_area=sum_masks,
        nms_pre=cfg.nms_pre,
        max_num=cfg.max_per_img,
        kernel=cfg.kernel,
        sigma=cfg.sigma,
        filter_thr=cfg.filter_thr,
    )
    mask_preds = mask_preds[keep_inds]
    mask_preds = F.interpolate(
        mask_preds.unsqueeze(0),
        size=upsampled_size,
        mode="bilinear",
        align_corners=False,
    )[:, :, :h, :w]
    mask_preds = F.interpolate(
        mask_preds, size=ori_shape[:2], mode="bilinear", align_corners=False
    ).squeeze(0)
    masks = mask_preds > cfg.mask_thr
    return masks, labels, scores
