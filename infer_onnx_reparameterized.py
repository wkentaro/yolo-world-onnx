#!/usr/bin/env python

import os
from typing import Tuple

import imgviz
import numpy as np
import onnxruntime

from _shared import get_coco_class_names
from _shared import transform_image
from _shared import untransform_bboxes

here = os.path.dirname(os.path.abspath(__file__))


def load_model():
    onnx_file = os.path.join(
        here,
        "checkpoints/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival_reparameterized.onnx",  # noqa: E501
    )
    inference_session = onnxruntime.InferenceSession(path_or_bytes=onnx_file)
    image_size = 640
    return inference_session, image_size


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
    max_num_detections: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    onnx_file = os.path.join(here, "checkpoints/non_maximum_suppression.onnx")
    inference_session = onnxruntime.InferenceSession(path_or_bytes=onnx_file)

    selected_indices = inference_session.run(
        output_names=["selected_indices"],
        input_feed={
            "boxes": boxes[None, :, :],
            "scores": scores[None, :, :].transpose(0, 2, 1),
            "max_output_boxes_per_class": np.array(
                [max_num_detections], dtype=np.int64
            ),
            "iou_threshold": np.array([iou_threshold], dtype=np.float32),
            "score_threshold": np.array([score_threshold], dtype=np.float32),
        },
    )[0]
    labels = selected_indices[:, 1]
    box_indices = selected_indices[:, 2]
    boxes = boxes[box_indices]
    scores = scores[box_indices, labels]

    if len(boxes) > max_num_detections:
        keep_indices = np.argsort(scores)[-max_num_detections:]
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

    return boxes, scores, labels


def main():
    image = imgviz.io.imread(
        os.path.join(here, "src/YOLO-World/demo/sample_images/bus.jpg")
    )
    class_names = get_coco_class_names()

    inference_session, image_size = load_model()

    input_image, original_image_hw, padding_hw = transform_image(
        image=image, image_size=image_size
    )
    #
    scores, bboxes = inference_session.run(
        output_names=["scores", "boxes"],
        input_feed={"images": input_image[None]},
    )
    scores = scores[0]
    bboxes = bboxes[0]
    #
    bboxes, scores, labels = non_maximum_suppression(
        boxes=bboxes,
        scores=scores,
        iou_threshold=0.7,
        score_threshold=0.1,
        max_num_detections=100,
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
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
