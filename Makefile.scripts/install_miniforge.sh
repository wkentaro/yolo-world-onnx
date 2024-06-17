#!/bin/bash -e

log_info() {
  echo -e "\033[1m$(basename $0):$LINENO - $1\033[0m"
}
log_error() {
  echo -e "\033[31m$(basename $0):$LINENO - $1\033[0m"
}

if [ $# -lt 1 ]; then
  echo "Usage: $0 INSTALL_DIR [MINIFORGE_VERSION]"
  exit 1
fi

INSTALL_DIR=$1
INSTALL_DIR=$(cd $INSTALL_DIR && pwd)
VERSION=${2:-latest}

if [ -e $INSTALL_DIR/.conda ]; then
  log_info "Miniforge is already installed at '$INSTALL_DIR/.conda'."
  exit 0
fi

log_info "Installing Miniforge: $VERSION"

export PS4='+ $(basename $0):$LINENO: '
set -x

TMPDIR=$(mktemp -d)
cd $TMPDIR

if [ "${VERSION}" == "latest" ]; then
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
else
    URL="https://github.com/conda-forge/miniforge/releases/download/${VERSION}/Miniforge3-$(uname)-$(uname -m).sh"
fi

if which wget &>/dev/null; then
  wget --no-check-certificate -q $URL -O miniforge.sh
else
  curl -s -L $URL -o miniforge.sh
fi

unset PYTHONPATH

bash ./miniforge.sh -p $INSTALL_DIR/.conda -b

cd -
rm -rf $TMPDIR
