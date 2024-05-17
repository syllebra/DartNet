from ultralytics import YOLO
import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a realistic random darts dataset.')
    parser.add_argument('dataset', metavar='Dataset', type=str, help='Dataset name in datasets_dir')
    parser.add_argument("-s",'--imsize', type=int, default=640, help="Image size")
    parser.add_argument("-d", "--datasets_dir", type=str, default="datasets", help="Datasets directory")
    parser.add_argument("-m",'--model', type=str, default="yolov8n.pt", help="Base model to train.")
    parser.add_argument("-e",'--epochs', type=int, default=500, help="Number of epochs")

    #parser.add_argument("-t", "--type", type=str, default= "normal", choices=['normal', 'temporal'], help="Dataset type")
    
    args = parser.parse_args()
    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    # Train the model
    dataset = "temporal_test"
    results = model.train(data=os.path.abspath(f'./{args.datasets_dir}/{args.dataset}/data.yml'), epochs=args.epochs, patience=args.epochs, imgsz=args.imsize, project="training",name=args.dataset, batch=-1,plots=True)