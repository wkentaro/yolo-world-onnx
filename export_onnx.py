#!/usr/bin/env python

import io
import os
import sys

import onnx
import onnxsim
import torch
from loguru import logger

from infer_pytorch import load_model

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


def main():
    model, image_size = load_model(deploy_model_cls=DeployModel)

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
            dynamic_axes={
                "images": {0: "batch"},
                "text_features": {0: "batch", 1: "classes"},
                "scores": {0: "batch", 2: "classes"},
                "boxes": {0: "batch"},
            },
            opset_version=12,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)

    onnxsim.simplify(onnx_model)
    onnx_file = os.path.join(
        here,
        "checkpoints/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",  # noqa: E501,
    )
    onnx.save(onnx_model, onnx_file)
    logger.info("ONNX model saved to {!r}.", onnx_file)


if __name__ == "__main__":
    main()
