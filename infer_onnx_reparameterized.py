#!/usr/bin/env python

import cv2
import imgviz
import imshow
import onnxruntime
import torch

from infer_pytorch import postprocess_detections
from infer_pytorch import transform_image
from infer_pytorch import untransform_bboxes


def main():
    onnx_file = "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx"  # noqa: E501
    image_size = 640

    class_names = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"  # noqa: E501
    class_names = class_names.split(",")

    inference_session = onnxruntime.InferenceSession(path_or_bytes=onnx_file)

    image = cv2.imread("src/YOLO-World/demo/sample_images/bus.jpg")[:, :, ::-1]

    input_image, original_image_hw, padding_hw = transform_image(
        image=image, image_size=image_size
    )

    scores, bboxes = inference_session.run(
        output_names=["scores", "boxes"],
        input_feed={"images": input_image.numpy()[None]},
    )
    scores = scores[0]
    bboxes = bboxes[0]

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
    imshow.imshow(
        [viz],
        get_title_from_item=lambda x: f"shape={x.shape}, mean={x.mean(axis=(0, 1)).astype(int).tolist()}",  # noqa: E501
    )


if __name__ == "__main__":
    main()