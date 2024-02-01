from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/v5s_30epoch/weights/best.pt')  # load a pretrained model (recommended for training)


model.val(data='my_dataset/bolt1/bolt_class.yaml',
                    split = 'val',
                    batch = 1,
                    imgsz = 640
            )

