#!/bin/bash -e

log_info() {
  echo -e "\033[1m$(basename $0):$LINENO - $1\033[0m"
}
log_error() {
  echo -e "\033[31m$(basename $0):$LINENO - $1\033[0m"
}

source $(pwd)/.conda/bin/activate

log_info "Installing YOLO-World."

export PS4='+ $(basename $0):$LINENO - '

set -x
git submodule update --init --recursive src/YOLO-World

cd src/YOLO-World

pip install -q torch==2.3.1 lvis==0.5.3
{ set +x; } 2>/dev/null

if pip show yolo_world | grep "Editable project location: $(pwd)"; then
    log_info "YOLO-World is already installed in editable mode."
else
    set -x
    pip install -q -e .
    { set +x; } 2>/dev/null
fi

log_info "YOLO-World recommends mmdet==3.0.0 and mmcv==2.0.0, but can't install mmcv==2.0.0, so installing mmdet==3.2.0 and mmcv==2.1.0 instead."
set -x
pip install -q mmdet==3.2.0
pip install -q mmcv==2.1.0
{ set +x; } 2>/dev/null

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

log_info "Downloading data."
urls=https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth
path=$(basename $urls)
sha256=5daf1395eb25b6f5adf27781022add7f20b70afdb107e725ccffc5ecc471dc7d
cached_download $urls $path $sha256

url=https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json
path=data/coco/lvis/$(basename $url)
sha256=02301f6ccd89d1ee3d35112cb57d000c3396f34e4073066c90b2c1fbf47b55ce
cached_download $url $path $sha256
