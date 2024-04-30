from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='./datasets/_GENERATED/data.yml', epochs=100, imgsz=480)