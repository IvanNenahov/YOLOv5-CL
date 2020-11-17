
cd G:/projects/yolov5_CL
python train.py --cfg models/yolov5s.yaml --data ../core50_350_1f/data.yaml --weights weights/yolov5s.pt --epochs_init 10 --epochs_iter 4 --img-size 416 --batch-size 4 
