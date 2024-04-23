import cv2
import torch
from pathlib import Path
import os
from threading import Thread
import pyzed.sl as sl
from yolov5.utils.general import (LOGGER, clean_str, cv2, is_colab, is_kaggle)
import numpy as np
from yolov5.utils.augmentations import letterbox
import math


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='file.streams', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads, self.imgs_R= [None] * n, [0] * n, [0] * n, [None] * n,[None] * n

        self.zed = sl.Camera()
        # 设置相机的分辨率1080和采集帧率30fps
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        self.init_params.camera_fps = 30  # fps可选：15、30、60、100
        err = self.zed.open(self.init_params)  # 根据自定义参数打开相机

        if err != sl.ERROR_CODE.SUCCESS:
            print(34)
            exit(1)
        self.runtime_parameters = sl.RuntimeParameters()  # 设置相机获取参数   
        self.image_L = sl.Mat()
        self.image_R = sl.Mat()
        self.f1position=[0,0]
        self.f2position=[0,0]
        self.Z4=0



        
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS: 
                # 读取摄像机的视频图像
                #success, img = cap.read()
                self.zed.retrieve_image(self.image_L, sl.VIEW.LEFT)
                self.zed.retrieve_image(self.image_R, sl.VIEW.RIGHT)
                img_L1=self.image_L.get_data()
                img_R1=self.image_R.get_data()
                
                b, g, r, a = cv2.split(img_L1)
                img_L = cv2.merge([b, g, r])
                b, g, r, a = cv2.split(img_R1)
                img_R = cv2.merge([b, g, r])
                self.imgs[i] = img_L
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f'{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array

        dx1=list(np.zeros([20,3]))
        dx2=list(np.zeros([20,3]))
        dx=list(np.ones([20,1]))

        while True:
            #print("sl.ERROR_CODE.SUCCESS!!!!!!!!!")
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS: 
                # 读取摄像机的视频图像
                #success, img = cap.read()
                self.zed.retrieve_image(self.image_L, sl.VIEW.LEFT)
                self.zed.retrieve_image(self.image_R, sl.VIEW.RIGHT)
                img_L1=self.image_L.get_data()
                img_R1=self.image_R.get_data()
                
                b, g, r, a = cv2.split(img_L1)
                img_L = cv2.merge([b, g, r])
                b, g, r, a = cv2.split(img_R1)
                img_R = cv2.merge([b, g, r])

                self.imgs[i] = img_L
                self.imgs_R[i] = img_R

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        # if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
        #     cv2.destroyAllWindows()
        #     raise StopIteration

        im0 = self.imgs.copy()
        im0_R= self.imgs_R.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, '', im0_R

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
