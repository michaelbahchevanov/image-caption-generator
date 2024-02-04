.PHONY: setup/env setup/models setup/install setup install-no-poetry clean jupyter

POETRY_VERSION := $(shell poetry --version 2>/dev/null)

setup/env:
	@echo "Creating virtual environment"
	@sleep 1
	@python -m venv .venv
	@echo "Run 'source .venv/bin/activate' before you continue."

setup/models:
	@wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./models/sam/weights/
	@wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -P ./models/yolo/weights/
	@wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P ./models/dino/weights/
	@wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py -P ./models/dino/config/

setup/install:
ifdef POETRY_VERSION
	@echo "Poetry found: $(POETRY_VERSION)"
	@sleep 1
else
	@echo "Poetry has not been found, installing..."
	@sleep 1
	@pip install poetry
endif
	@poetry check
	@poetry install

setup: setup/models setup/install
	$(eval SITE_PACKAGES_PTH := $(shell find .venv -type d -name "site-packages" -print -quit)/imagecap.pth)
	@echo "$(PWD)/src" > $(SITE_PACKAGES_PTH)

install-no-poetry: requirements.txt
	@pip install -r requirements.txt

clean:
	@rm -rf __pycache__

jupyter: exploration
	@jupyter notebook exploration