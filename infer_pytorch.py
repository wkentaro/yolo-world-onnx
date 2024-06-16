#!/usr/bin/env python

import os
import sys
from typing import Tuple

import cv2
import imgviz
import imshow
import numpy as np
import torch
import torchvision
from loguru import logger
from mmdet.apis import init_detector
from mmengine.config import ConfigDict

here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(here, "src/YOLO-World/deploy"))
from easydeploy.model import DeployModel  # noqa: E402
from easydeploy.model import MMYOLOBackend  # noqa: E402


def load_model() -> Tuple[torch.nn.Module, int]:
    config = "configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"  # noqa: E501
    checkpoint = "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    logger.info("Loading model: config={!r}, checkpoint={!r}", config, checkpoint)
    os.chdir(os.path.join(here, "src/YOLO-World"))
    base_model = init_detector(config=config, checkpoint=checkpoint, device="cpu")
    os.chdir(here)

    model = DeployModel(
        baseModel=base_model,
        backend=MMYOLOBackend.ONNXRUNTIME,
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


def postprocess_detections(
    ori_bboxes: torch.Tensor,
    ori_scores: torch.Tensor,
    nms_thr: float,
    score_thr: float,
    max_dets: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_list = []
    labels_list = []
    bboxes_list = []
    # class-specific NMS
    for cls_id in range(ori_scores.shape[1]):
        cls_scores = ori_scores[:, cls_id]
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        keep_idxs = torchvision.ops.nms(ori_bboxes, cls_scores, iou_threshold=nms_thr)
        cur_bboxes = ori_bboxes[keep_idxs]
        cls_scores = cls_scores[keep_idxs]
        labels = labels[keep_idxs]
        scores_list.append(cls_scores)
        labels_list.append(labels)
        bboxes_list.append(cur_bboxes)

    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    bboxes = torch.cat(bboxes_list, dim=0)

    keep_idxs = scores > score_thr
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]
    if len(keep_idxs) > max_dets:
        _, sorted_idx = torch.sort(scores, descending=True)
        keep_idxs = sorted_idx[:max_dets]
        bboxes = bboxes[keep_idxs]
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]

    return bboxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()


def transform_image(
    image: np.ndarray, image_size: int
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    height, width = image.shape[:2]

    scale = image_size / max(height, width)
    image_resized = cv2.resize(
        image,
        dsize=(int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    pad_height = image_size - image_resized.shape[0]
    pad_width = image_size - image_resized.shape[1]
    image_resized = np.pad(
        image_resized,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
            (0, 0),
        ),
        mode="constant",
        constant_values=114,
    )
    input_image = torch.tensor(
        image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    )
    return input_image, (height, width), (pad_height, pad_width)


def untransform_bboxes(
    bboxes: np.ndarray,
    image_size: int,
    original_image_hw: Tuple[int, int],
    padding_hw: Tuple[int, int],
):
    bboxes -= np.array([padding_hw[1] // 2, padding_hw[0] // 2] * 2)
    bboxes /= image_size / max(original_image_hw)
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, original_image_hw[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, original_image_hw[0])
    bboxes = bboxes.round().astype(int)
    return bboxes


def main():
    model, image_size = load_model()

    image = cv2.imread("src/YOLO-World/demo/sample_images/bus.jpg")[:, :, ::-1]

    class_names = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"  # noqa: E501
    class_names = class_names.split(",")

    model.baseModel.reparameterize(
        [[class_name] for class_name in class_names] + [[" "]]
    )
    model.baseModel.text_feats = model.baseModel.text_feats.permute(1, 0, 2)

    input_image, original_image_hw, padding_hw = transform_image(
        image=image, image_size=image_size
    )
    with torch.no_grad():
        scores, bboxes = model(inputs=input_image[None])
        scores = scores[0]
        bboxes = bboxes[0]

    bboxes, scores, labels = postprocess_detections(
        ori_bboxes=bboxes,
        ori_scores=scores,
        nms_thr=0.7,
        score_thr=0.1,
        max_dets=100,
    )
    bboxes = untransform_bboxes(
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
        font_size=15,
        line_width=1,
    )
    imshow.imshow(
        [viz],
        get_title_from_item=lambda x: f"shape={x.shape}, mean={x.mean(axis=(0, 1)).astype(int).tolist()}",  # noqa: E501
    )


if __name__ == "__main__":
    main()
