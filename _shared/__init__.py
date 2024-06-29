import argparse
import os
from typing import List
from typing import Tuple

import imgviz
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def get_argument_parser(class_names=True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-file",
        type=str,
        default=os.path.join(here, "../src/YOLO-World/demo/sample_images/bus.jpg"),
        help="image file",
    )

    if class_names:

        def comma_separated_string(value):
            return value.split(",")

        parser.add_argument(
            "--class-names",
            type=comma_separated_string,
            default=get_coco_class_names(),
            help="class names",
        )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="IoU threshold",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="score threshold",
    )
    parser.add_argument(
        "--max-num-detections",
        type=int,
        default=100,
        help="maximum number of detections",
    )
    return parser


def get_coco_class_names() -> List[str]:
    class_names_str: str = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"  # noqa: E501
    class_names: List[str] = class_names_str.split(",")
    return class_names


def transform_image(
    image: np.ndarray, image_size: int
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    height, width = image.shape[:2]

    scale = image_size / max(height, width)
    image_resized = imgviz.resize(
        image,
        height=int(height * scale),
        width=int(width * scale),
        interpolation="linear",
    )
    pad_height = image_size - image_resized.shape[0]
    pad_width = image_size - image_resized.shape[1]
    image_resized = np.pad(
        image_resized,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
            (0, 0),
        ),
        mode="constant",
        constant_values=114,
    )
    input_image = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return input_image, (height, width), (pad_height, pad_width)


def untransform_bboxes(
    bboxes: np.ndarray,
    image_size: int,
    original_image_hw: Tuple[int, int],
    padding_hw: Tuple[int, int],
) -> np.ndarray:
    bboxes -= np.array([padding_hw[1] // 2, padding_hw[0] // 2] * 2)
    bboxes /= image_size / max(original_image_hw)
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, original_image_hw[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, original_image_hw[0])
    bboxes = bboxes.round().astype(int)
    return bboxes


def visualize_bboxes(
    image: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: np.ndarray,
) -> None:
    captions = [
        f"{class_names[label]}: {score:.2f}" for label, score in zip(labels, scores)
    ]
    font_size = image.shape[0] // 80
    line_width = max(1, font_size // 10)
    viz = imgviz.instances2rgb(
        image=image,
        bboxes=bboxes[:, [1, 0, 3, 2]],
        labels=labels + 1,
        captions=captions,
        font_size=font_size,
        line_width=line_width,
    )
    imgviz.io.pil_imshow(viz)
