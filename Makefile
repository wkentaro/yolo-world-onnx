all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

install_miniforge:
	@Makefile.scripts/install_miniforge.sh . 24.3.0-0

install_yolo_world: install_miniforge
	@Makefile.scripts/install_yolo_world.sh

lint:
	ruff check
	ruff format --diff
	mypy infer_pytorch.py export_onnx_reparameterized.py

format:
	ruff check --fix
	ruff format
