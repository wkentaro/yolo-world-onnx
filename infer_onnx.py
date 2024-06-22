#!/usr/bin/env python3

import os

import clip
import imgviz
import onnxruntime
import torch

from infer_pytorch import get_coco_class_names
from infer_pytorch import postprocess_detections
from infer_pytorch import transform_image
from infer_pytorch import untransform_bboxes

here = os.path.dirname(os.path.abspath(__file__))


def load_model():
    onnx_file = os.path.join(
        here,
        "checkpoints/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",  # noqa: E501
    )
    inference_session = onnxruntime.InferenceSession(path_or_bytes=onnx_file)
    image_size = 640
    return inference_session, image_size


def main():
    image = imgviz.io.imread(
        os.path.join(here, "src/YOLO-World/demo/sample_images/bus.jpg")
    )
    class_names = get_coco_class_names()

    yolo_world_session, image_size = load_model()

    input_image, original_image_hw, padding_hw = transform_image(
        image=image, image_size=image_size
    )
    #
    token = clip.tokenize(class_names + [" "]).numpy().astype(int)
    textual_session = onnxruntime.InferenceSession(
        os.path.join(here, "checkpoints/vitb32-textual.onnx")
    )
    (text_feats,) = textual_session.run(None, {"input": token})
    text_feats = torch.from_numpy(text_feats)
    text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
    #
    scores, bboxes = yolo_world_session.run(
        output_names=["scores", "boxes"],
        input_feed={
            "images": input_image.numpy()[None],
            "text_features": text_feats.numpy()[None],
        },
    )
    scores = scores[0]
    bboxes = bboxes[0]
    #
    bboxes, scores, labels = postprocess_detections(
        ori_bboxes=torch.from_numpy(bboxes),
        ori_scores=torch.from_numpy(scores),
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
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
