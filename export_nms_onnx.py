#!/usr/bin/env python

import os

import onnx
import onnx.helper as helper
from loguru import logger

here = os.path.dirname(os.path.abspath(__file__))


def main():
    node = helper.make_node(
        "NonMaxSuppression",
        inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        outputs=["selected_indices"],
    )

    boxes = helper.make_tensor_value_info(
        "boxes", onnx.TensorProto.FLOAT, [None, None, 4]
    )
    scores = helper.make_tensor_value_info(
        "scores", onnx.TensorProto.FLOAT, [None, None, None]
    )
    max_output_boxes_per_class = helper.make_tensor_value_info(
        "max_output_boxes_per_class", onnx.TensorProto.INT64, [1]
    )
    iou_threshold = helper.make_tensor_value_info(
        "iou_threshold", onnx.TensorProto.FLOAT, [1]
    )
    score_threshold = helper.make_tensor_value_info(
        "score_threshold", onnx.TensorProto.FLOAT, [1]
    )
    selected_indices = helper.make_tensor_value_info(
        "selected_indices", onnx.TensorProto.INT64, [None, 3]
    )

    graph = helper.make_graph(
        [node],
        "NonMaxSuppressionGraph",
        [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
        [selected_indices],
    )

    model = helper.make_model(graph, producer_name="nms")

    onnx_file = os.path.join(here, "checkpoints/non_maximum_suppression.onnx")
    onnx.save(model, onnx_file)
    logger.info(f"Saved model to {onnx_file}")


if __name__ == "__main__":
    main()
