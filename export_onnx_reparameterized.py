#!/usr/bin/env python

import io
import os

import onnx
import onnxsim
import torch
from loguru import logger

from _shared import get_coco_class_names
from infer_pytorch import load_model

here = os.path.dirname(os.path.abspath(__file__))


def main():
    model, image_size = load_model()
    class_names = get_coco_class_names()

    fake_images = torch.randn(1, 3, image_size, image_size)

    logger.info("Reparameterizing model.")
    model.baseModel.reparameterize(
        [[class_name] for class_name in class_names] + [[" "]]
    )
    model.baseModel.text_feats = model.baseModel.text_feats.permute(1, 0, 2)

    logger.info("Exporting ONNX model.")
    with io.BytesIO() as f:
        torch.onnx.export(
            model,
            fake_images,
            f,
            input_names=["images"],
            output_names=["scores", "boxes"],
            opset_version=12,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)

    onnxsim.simplify(onnx_model)
    onnx_file = os.path.join(
        here,
        "checkpoints/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival_reparameterized.onnx",  # noqa: E501
    )
    onnx.save(onnx_model, onnx_file)
    logger.info("ONNX model saved to {!r}.", onnx_file)


if __name__ == "__main__":
    main()
