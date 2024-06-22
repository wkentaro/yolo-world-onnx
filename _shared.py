from typing import List
from typing import Tuple

import imgviz
import numpy as np


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
