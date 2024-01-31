from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# model.info()
# Train the model
# model.train(data='ultralytics/cfg/datasets/coco128.yaml', batch =16, epochs=10, imgsz=640)
model.train(data='my_dataset/bolt1/bolt_class.yaml', 
            batch =4, 
            epochs=10,
            imgsz=640,
            save_dir = 'runs/yolov8s_10epoch'
            )

