<div align="center">
  <h1>yolo-world-onnx</h1>
  <a href="#installation"><b>Installation</b></a>
  | <a href="#usage"><b>Usage</b></a>
  | <a href="#license"><b>License</b></a>
</div>

<br/>

<div align="center">
    <img src="README.md.files/teaser.jpg" height="300px" />
</div>

## ONNX Models

[Download](https://github.com/wkentaro/yolo-world-onnx/releases/latest)

## Installation

Install dependencies and downloads the pre-trained model checkpoints:

```
make install
```

## Usage

### Inference with PyTorch

Run inference using the PyTorch pre-trained model:

```
./infer_pytorch.py
./infer_pytorch.py --class-names person,shoes
./infer_pytorch.py --image-file data/images/dogs.jpg \
                   --class-names dog,eye,nose,ear,tail \
                   --iou-threshold 0.5 \
                   --score-threshold 0.09
```

### Export ONNX

Export YOLO-World and NMS models to ONNX format:

```
./export_onnx.py
./export_nms_onnx.py
```

### Inference with ONNX

Run inference using the exported ONNX model:

```
./infer_onnx.py
./infer_onnx.py --class-names person,shoes
./infer_onnx.py --image-file data/images/dogs.jpg \
                --class-names dog,eye,nose,ear,tail \
                --iou-threshold 0.5 \
                --score-threshold 0.09
```

## License

GPLv3
