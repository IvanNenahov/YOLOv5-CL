
cd G:/projects/yolov5_CL
python train.py --cfg models/yolov5s.yaml --reg-lambda 0 --data ../core50_350_1f/data.yaml --weights weights/yolov5s.pt --epochs_init 40 --epochs_iter 5 --img-size 416 --batch-size 4 --logdir runs/ext_memory_21_11
