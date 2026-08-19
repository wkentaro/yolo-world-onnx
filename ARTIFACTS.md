# Release artifact provenance

This document maps the binary artifacts published by this repository to their
corresponding source, build inputs, and export commands. It records provenance;
it does not replace the license notices in this repository or its upstream
projects.

## v0.1.0

- Release: <https://github.com/wkentaro/yolo-world-onnx/releases/tag/v0.1.0>
- Corresponding source: commit
  [`d1da5883e2ce61ff50e00237fee44a7907b4017b`](https://github.com/wkentaro/yolo-world-onnx/tree/d1da5883e2ce61ff50e00237fee44a7907b4017b)
  (tag `v0.1.0`)
- YOLO-World source: commit
  [`b449b98202e931590513c16e4830318be2dde946`](https://github.com/AILab-CVC/YOLO-World/tree/b449b98202e931590513c16e4830318be2dde946),
  pinned by the `src/YOLO-World` submodule
- License: [GNU General Public License v3.0](LICENSE)

| Artifact | SHA-256 | Export source |
| --- | --- | --- |
| `yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx` | `92660c6456766439a2670cf19a8a258ccd3588118622a15959f39e253731c05d` | [`export_onnx.py`](https://github.com/wkentaro/yolo-world-onnx/blob/v0.1.0/export_onnx.py) |
| `non_maximum_suppression.onnx` | `328310ba8fdd386c7ca63fc9df3963cc47b1268909647abd469e8ebdf7f3d20a` | [`export_nms_onnx.py`](https://github.com/wkentaro/yolo-world-onnx/blob/v0.1.0/export_nms_onnx.py) |

The YOLO-World export uses the following pretrained checkpoint:

- URL: <https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth>
- SHA-256: `5daf1395eb25b6f5adf27781022add7f20b70afdb107e725ccffc5ecc471dc7d`
- Configuration:
  [`configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py`](https://github.com/AILab-CVC/YOLO-World/blob/b449b98202e931590513c16e4830318be2dde946/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py)

To inspect the corresponding source and run the export commands:

```console
git clone --recurse-submodules --branch v0.1.0 \
  https://github.com/wkentaro/yolo-world-onnx.git
cd yolo-world-onnx
make install
./export_onnx.py
./export_nms_onnx.py
sha256sum checkpoints/*.onnx
```

Package solver results and exporter versions can affect ONNX serialization, so
the commands are not claimed to reproduce the published files byte for byte on
every current platform. The table above provides the authoritative hashes for
the distributed `v0.1.0` artifacts.
