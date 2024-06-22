#!/usr/bin/env python

import os
import sys
from typing import Tuple

import imgviz
import numpy as np
import torch
import torchvision
from loguru import logger
from mmdet.apis import init_detector
from mmengine.config import ConfigDict

import _shared

here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(here, "src/YOLO-World/deploy"))
import easydeploy.model  # noqa: E402


def load_model(deploy_model_cls=None) -> Tuple[torch.nn.Module, int]:
    if deploy_model_cls is None:
        deploy_model_cls = easydeploy.model.DeployModel

    config = os.path.join(
        here,
        "src/YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",  # noqa: E501
    )
    checkpoint = os.path.join(
        here,
        "checkpoints/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth",
    )
    logger.info("Loading model: config={!r}, checkpoint={!r}", config, checkpoint)
    base_model = init_detector(config=config, checkpoint=checkpoint, device="cpu")

    model = deploy_model_cls(
        baseModel=base_model,
        backend=easydeploy.model.MMYOLOBackend.ONNXRUNTIME,
        postprocess_cfg=ConfigDict(
            pre_top_k=1000,
            keep_top_k=100,
            iou_threshold=0.7,
            score_threshold=0.1,
        ),
        with_nms=False,
        without_bbox_decoder=False,
    )
    model.eval()

    image_size = 640
    return model, image_size


def non_maximum_suppression_pytorch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    score_threshold: float,
    max_num_detections: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_list = []
    labels_list = []
    bboxes_list = []
    for cls_id in range(scores.shape[1]):
        cls_scores = scores[:, cls_id]
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        keep_idxs = torchvision.ops.nms(boxes, cls_scores, iou_threshold=iou_threshold)
        cur_bboxes = boxes[keep_idxs]
        cls_scores = cls_scores[keep_idxs]
        labels = labels[keep_idxs]
        scores_list.append(cls_scores)
        labels_list.append(labels)
        bboxes_list.append(cur_bboxes)
    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    bboxes = torch.cat(bboxes_list, dim=0)

    keep_idxs = scores > score_threshold
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]

    if len(keep_idxs) > max_num_detections:
        _, sorted_idx = torch.sort(scores, descending=True)
        keep_idxs = sorted_idx[:max_num_detections]
        bboxes = bboxes[keep_idxs]
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]

    return bboxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()


def main():
    parser = _shared.get_argument_parser()
    args = parser.parse_args()

    image = imgviz.io.imread(args.image_file)
    class_names = args.class_names

    model, image_size = load_model()
    model.baseModel.reparameterize(
        [[class_name] for class_name in class_names] + [[" "]]
    )
    model.baseModel.text_feats = model.baseModel.text_feats.permute(1, 0, 2)

    input_image, original_image_hw, padding_hw = _shared.transform_image(
        image=image, image_size=image_size
    )
    with torch.no_grad():
        scores, bboxes = model(inputs=torch.Tensor(input_image[None]))
        scores = scores[0]
        bboxes = bboxes[0]
    bboxes, scores, labels = non_maximum_suppression_pytorch(
        boxes=bboxes,
        scores=scores,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_num_detections=args.max_num_detections,
    )
    bboxes = _shared.untransform_bboxes(
        bboxes=bboxes,
        image_size=image_size,
        original_image_hw=original_image_hw,
        padding_hw=padding_hw,
    )

    captions = [
        f"{class_names[label]}: {score:.2f}" for label, score in zip(labels, scores)
    ]
    viz = imgviz.instances2rgb(
        image=image,
        bboxes=bboxes[:, [1, 0, 3, 2]],
        labels=labels + 1,
        captions=captions,
        font_size=image.shape[0] // 80,
        line_width=1,
    )
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
