#!/usr/bin/env python

import io
import os
import sys

import onnx
import onnxsim
import torch
from loguru import logger
from mmdet.apis import init_detector
from mmengine.config import ConfigDict

here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(here, "src/YOLO-World/deploy"))
from easydeploy.model import DeployModel  # noqa: E402
from easydeploy.model import MMYOLOBackend  # noqa: E402


def load_model():
    config = "configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"  # noqa: E501
    checkpoint = "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    logger.info("Loading model: config={!r}, checkpoint={!r}", config, checkpoint)
    os.chdir(os.path.join(here, "src/YOLO-World"))
    model = init_detector(config=config, checkpoint=checkpoint, device="cpu")
    os.chdir(here)
    return model


def main():
    model = load_model()
    model.eval()

    fake_images = torch.randn(1, 3, 640, 640)
    # fake_texts = torch.rand(1, 80, 512)
    # fake_texts = fake_texts / fake_texts.norm(p=2, dim=-1, keepdim=True)

    texts = [["person", "bus"]]
    logger.info("Reparameterizing model with texts={!r}.", texts)
    model.reparameterize(texts=texts)

    postprocess_cfg = ConfigDict(
        pre_top_k=1000,
        keep_top_k=100,
        iou_threshold=0.5,
        score_threshold=0.05,
    )
    deploy_model = DeployModel(
        baseModel=model,
        backend=MMYOLOBackend.ONNXRUNTIME,
        postprocess_cfg=postprocess_cfg,
        with_nms=False,
        without_bbox_decoder=False,
    )
    deploy_model.eval()

    logger.info("Exporting ONNX model.")
    with io.BytesIO() as f:
        torch.onnx.export(
            deploy_model,
            # (fake_images, fake_texts),
            fake_images,
            f,
            # input_names=["images", "texts"],
            input_names=["images"],
            output_names=["scores", "boxes"],
            opset_version=12,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)

    onnxsim.simplify(onnx_model)
    onnx_file = "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx"  # noqa: E501
    onnx.save(onnx_model, onnx_file)
    logger.info("ONNX model saved to {!r}.", onnx_file)


if __name__ == "__main__":
    main()
