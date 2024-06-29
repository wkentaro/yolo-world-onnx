#!/bin/bash

log_info() {
  echo -e "\033[1m$(basename $0):$LINENO - $1\033[0m"
}
cached_download() {
    url=$1
    path=$2
    sha256=$3
    if test -e $path && $(sha256sum $path | grep ^$sha256 &>/dev/null); then
        log_info "Already downloaded '$url' at '$path'."
    else
        log_info "Downloading from '$url' to '$path'."
        set -x
        mkdir -p $(dirname $path)
        curl -L $url -o $path
        { set +x; } 2>/dev/null
    fi
}

log_info "Downloading checkpoints."
urls=https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth
path=checkpoints/$(basename $urls)
sha256=5daf1395eb25b6f5adf27781022add7f20b70afdb107e725ccffc5ecc471dc7d
cached_download $urls $path $sha256

url=https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json
path=data/coco/lvis/$(basename $url)
sha256=02301f6ccd89d1ee3d35112cb57d000c3396f34e4073066c90b2c1fbf47b55ce
cached_download $url $path $sha256

url=https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx
path=checkpoints/vitb32-textual.onnx
sha256=55c85d8cbb096023781c1d13c557eb95d26034c111bd001b7360fdb7399eec68
cached_download $url $path $sha256

url=https://github.com/wkentaro/yolo-world-onnx/releases/download/v0.1.0/non_maximum_suppression.onnx
path=checkpoints/non_maximum_suppression.onnx
sha256=328310ba8fdd386c7ca63fc9df3963cc47b1268909647abd469e8ebdf7f3d20a
cached_download $url $path $sha256

url=https://github.com/wkentaro/yolo-world-onnx/releases/download/v0.1.0/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx
path=checkpoints/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx
sha256=92660c6456766439a2670cf19a8a258ccd3588118622a15959f39e253731c05d
cached_download $url $path $sha256
