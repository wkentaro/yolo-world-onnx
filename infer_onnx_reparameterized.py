#!/usr/bin/env python

import os
from typing import Tuple

import imgviz
import numpy as np
import onnxruntime

import _shared

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
    parser = _shared.get_argument_parser(class_names=False)
    args = parser.parse_args()

    image = imgviz.io.imread(args.image_file)

    inference_session, image_size = load_model()
    class_names = _shared.get_coco_class_names()

    input_image, original_image_hw, padding_hw = _shared.transform_image(
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
    _shared.visualize_bboxes(
        image=image,
        bboxes=bboxes,
        labels=labels,
        scores=scores,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
