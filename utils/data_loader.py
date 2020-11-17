import os
import pandas as pd
import torch
from PIL import Image
import numpy as np
import cv2 as cv
from tqdm import tqdm

class CORE50s(torch.utils.data.Dataset):
    def __init__(self, pathfile='G:/projects/core50_350_1f/train.txt', imsize=(416, 416)):
        with open(pathfile, 'r') as f:
            self.train_paths = [r[:-1] for r in f.readlines()]
        self.imsize=imsize

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, n_batch):
        self.n_batch = n_batch
        # reproduce CORE50 NIC experiment
        if n_batch == 0:
            paths = self.train_paths[:3000]
        else:
            start = 3000 + (n_batch - 1) * 300
            end = 3000 + n_batch * 300

            paths = self.train_paths[start: end]
        # load images
        x = np.zeros((len(paths), 3, *self.imsize), dtype=np.uint8)
        y = np.zeros((len(paths), 1, 6), dtype=np.float32)

        # for i, path in tqdm(enumerate(paths), total=len(paths)):
        #     im = np.array(Image.open(path))
        #     im = cv.resize(im, dsize=(416, 416))
        #     x[i] = im
        #


        # load labels

        # y = np.zeros((len(paths), 1, 6), dtype=np.float32)
        # for i, path in enumerate(paths):
        #     y[i, 0, 1:] = np.loadtxt(path.replace('images', 'labels').replace('png', 'txt'))
        #                # y = np.zeros((len(paths), 6), dtype=np.float32)
        #         # for i, path in enumerate(paths):
        #         #     y[i, 1:] = np.loadtxt(path.replace('images', 'labels').replace('png', 'txt'))


        for index, path in tqdm(enumerate(paths), total=len(paths)):
            img, (h0, w0), (h, w) = load_image(self, path)

            # Letterbox
            #shape = self.batch_shapes[self.batch[index]] #if self.rect else self.imsize[0]  # final letterboxed shape
            shape = self.imsize
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=False)

            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels

            labels = np.loadtxt(path.replace('images', 'labels').replace('png', 'txt'))
            labels = np.expand_dims(labels, 0)
            l = []
            if labels.size > 0:
                # Normalized xywh to pixel xyxy format
                l = labels.copy()
                l[:, 1] = ratio[0] * w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]  # pad width
                l[:, 2] = ratio[1] * h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]  # pad height
                l[:, 3] = ratio[0] * w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
                l[:, 4] = ratio[1] * h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
            nL = len(labels)
            labels_out = torch.zeros((nL, 6))
            if nL:
                labels_out[:, 1:] = torch.from_numpy(l)

            if nL:
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
                labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
                labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

            x[index] = img
            y[index] = labels_out




        return torch.from_numpy(x).permute(0, 3, 1, 2), torch.from_numpy(y)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def load_image(dset, path):
    # loads 1 image from dataset, returns img, original hw, resized hw

   # path = self.img_files[index]
    img = cv.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = dset.imsize[0] / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv.INTER_AREA if r < 1 and not dset.augment else cv.INTER_LINEAR
        img = cv.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def main():
    dset = CORE50s()
    print(dset.train_paths)
    a = dset.__getitem__(0)
    print(a)




if __name__ == '__main__':
    main()

