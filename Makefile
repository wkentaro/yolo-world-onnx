all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

install_miniforge:
	@Makefile.scripts/install_miniforge.sh . 24.3.0-0

install_yolo_world: install_miniforge
	@Makefile.scripts/install_yolo_world.sh

install_checkpoints:
	@Makefile.scripts/install_checkpoints.sh

install: install_yolo_world install_checkpoints

install_for_onnx: install_miniforge install_checkpoints
	@.conda/bin/pip install imgviz loguru onnxruntime

lint:
	ruff check
	ruff format --diff
	mypy infer_*.py export_*.py

format:
	ruff check --fix
	ruff format
