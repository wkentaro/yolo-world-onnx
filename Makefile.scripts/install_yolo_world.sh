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
