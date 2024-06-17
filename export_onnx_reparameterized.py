#!/usr/bin/env python

import io
import os

import onnx
import onnxsim
import torch
from loguru import logger

from infer_pytorch import load_model

here = os.path.dirname(os.path.abspath(__file__))


def main():
    model, image_size = load_model()

    fake_images = torch.randn(1, 3, image_size, image_size)

    class_names = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"  # noqa: E501
    class_names = class_names.split(",")

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
