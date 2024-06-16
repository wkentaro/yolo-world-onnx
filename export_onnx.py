#!/usr/bin/env python

import io
import os
import sys
from typing import Tuple

import onnx
import onnxsim
import torch
from loguru import logger
from mmdet.apis import init_detector
from mmengine.config import ConfigDict

here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(here, "src/YOLO-World/deploy"))
import easydeploy.model  # noqa: E402


class DeployModel(easydeploy.model.DeployModel):
    def forward(self, images: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        # self.baseModel._forward {{
        # self.baseModel.extract_feat {{
        img_feats = self.baseModel.backbone.forward_image(images)
        if self.baseModel.with_neck:
            img_feats = self.baseModel.neck(img_feats, text_feats)
        # }} self.baseModel.extract_feat
        neck_outputs = self.baseModel.bbox_head.forward(img_feats, text_feats)
        # }} self.baseModel._forward

        assert self.with_postprocess
        return self.pred_by_feat(*neck_outputs)


def load_model() -> Tuple[torch.nn.Module, int]:
    config = "configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"  # noqa: E501
    checkpoint = "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    logger.info("Loading model: config={!r}, checkpoint={!r}", config, checkpoint)
    os.chdir(os.path.join(here, "src/YOLO-World"))
    base_model = init_detector(config=config, checkpoint=checkpoint, device="cpu")
    os.chdir(here)

    model = DeployModel(
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


def main():
    model, image_size = load_model()

    images = torch.randn(1, 3, image_size, image_size)
    text_feats = torch.rand(1, 81, 512).float()
    text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)

    logger.info("Exporting ONNX model.")
    with io.BytesIO() as f:
        torch.onnx.export(
            model,
            (images, text_feats),
            f,
            input_names=["images", "text_features"],
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
