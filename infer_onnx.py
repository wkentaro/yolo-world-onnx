#!/usr/bin/env python3

import os

import imgviz
import numpy as np
import onnxruntime

import _shared
from _shared import clip
from infer_onnx_reparameterized import non_maximum_suppression

here = os.path.dirname(os.path.abspath(__file__))


def load_model():
    onnx_file = os.path.join(
        here,
        "checkpoints/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",  # noqa: E501
    )
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(
            f"File not found: {onnx_file}, download it from "
            "https://github.com/wkentaro/yolo-world-onnx/releases/latest"
        )
    inference_session = onnxruntime.InferenceSession(path_or_bytes=onnx_file)
    image_size = 640
    return inference_session, image_size


def main():
    parser = _shared.get_argument_parser()
    args = parser.parse_args()

    image = imgviz.io.imread(args.image_file)
    class_names = args.class_names

    yolo_world_session, image_size = load_model()

    input_image, original_image_hw, padding_hw = _shared.transform_image(
        image=image, image_size=image_size
    )
    #
    token = clip.tokenize(class_names + [" "])
    textual_session = onnxruntime.InferenceSession(
        os.path.join(here, "checkpoints/vitb32-textual.onnx")
    )
    (text_feats,) = textual_session.run(None, {"input": token})
    text_feats = text_feats / np.linalg.norm(text_feats, ord=2, axis=1, keepdims=True)
    #
    scores, bboxes = yolo_world_session.run(
        output_names=["scores", "boxes"],
        input_feed={
            "images": input_image[None],
            "text_features": text_feats[None],
        },
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
