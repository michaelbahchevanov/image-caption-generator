download_models:
	@wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./models/sam/weights/
	@wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -P ./models/yolo/weights/
	@wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P ./models/dino/weights/
	@wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py -P ./models/dino/config/

install: requirements.txt
	@pip install -r requirements.txt

clean:
	@rm -rf __pycache__

jupyter: exploration
	@jupyter notebook exploration