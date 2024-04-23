# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import mediapipe as mp
import pyzed.sl as sl
import argparse
import os
import re
import platform
import sys
from pathlib import Path
from datetime import datetime
import torch
import cv2
import random
import cnn
import torchvision.transforms as transforms
import threading
import copy
import time
import multiprocessing
import subprocess
# from Controller import Controller
from threading import Thread
from time import sleep, ctime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots#, LoadStreams
from dataloaders2 import LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import sys
# import UI
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from Mysql_Setting import db, connection
from datetime import datetime
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer
import argparse
import os
import platform
import sys
from argparse import Namespace
from pathlib import Path
import torch
import cv2
import numpy as np
import mediapipe as mp
import time
from ultralytics import YOLO
import pymysql
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QObject, Qt, pyqtSignal, QThread, QTimer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QTimer, QObject, pyqtSignal
#log:time writing
cur = db.cursor()
Worker = 'Worker'
Records = 'Records'

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

sentence = []
threshold = 0.8
actions = np.array(['CatchingSmall', 'CatchingBig', 'Screwing', 'Done'])

# Specify certain parameters.
model1 = Sequential()
model1.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
model1.add(LSTM(128, return_sequences=True, activation='relu'))
model1.add(LSTM(64, return_sequences=False, activation='relu'))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(actions.shape[0], activation='softmax'))
model1.load_weights('action.h5')
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
colors1 = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 117, 245)]  # 四个动作的框框，要增加动作数目，就多加RGB元组

model_ang = cnn.CNN()
model_ang.load_state_dict(torch.load("bestangle.pt"))
model_ang.eval()
shape_of_img = 20
img_set = []
acf_list = []
obj_angle_list = []
det_list = []

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_styled_landmarks(image, results):
    # Draw face connections
    """
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    """
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

def auto_get_file(path):
  directory, file_name = os.path.split(path)
  while os.path.isfile(path):
    pattern = '(\d+)\)\.'
    if re.search(pattern, file_name) is None:
      file_name = file_name.replace('.', '(0).')
    else:
      current_number = int(re.findall(pattern, file_name)[-1])
      new_number = current_number + 1
      file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
    path = os.path.join(directory + os.sep + file_name)
  return path

def rot_img(img,angle):
    w, h, c = img.shape
    box = [0, 0, w, h]   # 创建一个列表，共4项，前两项表示原点，第三项为宽，第四项为高
    print(box)
    bbox = BBox(box)     # 创建BBox对象
    center = ((bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2)
    print(center)
    rot_mat=cv2.getRotationMatrix2D(center,angle,1)
    img_out=cv2.warpAffine(img,rot_mat,(h,w))
    return img_out,angle

def detect_ang(img_in):
    model_ang = cnn.CNN()
    opt = parse_opt()
    model_ang.load_state_dict(torch.load("bestangle.pt"))
    model_ang.eval()
    shape_of_img = 20
    img_set = []

    img = img_in
    angle = random.randint(-30,30)
    img1 = img
    imgout, det = my_detect.run(img1,**vars(opt))
    if det != []:
        x1=det[0]
        if abs(x1[1]-x1[3])>abs(x1[0]-x1[2]):
            x1[2]=x1[0]+abs(x1[1]-x1[3])
        else:
            x1[3]=x1[1]+abs(x1[0]-x1[2])
        cropped_image = img1[int(x1[1]):int(x1[3]), int(x1[0]):int(x1[2])]
        cv2.imwrite("cropped_image.jpg", cropped_image)


        in_img = cv2.resize(cropped_image,(100,100))
        transf = transforms.ToTensor()
        in_img = transf(in_img)  # tensor数据格式是torch(C,H,W)

        img_set.append(in_img)

        #print(angle_set_in,angle_set_in.size())
        img_set_in = torch.stack(img_set,dim=0)
        print("shape", img_set_in.size())
        output = model_ang(img_set_in)         #调用模型预测

        #    print(cnn_train.weights)
        print(output)

error_tip = 0

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1747, 600)
        MainWindow.setStyleSheet("*{font:16px}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.btn_quit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_quit.setObjectName("btn_quit")
        self.horizontalLayout_12.addWidget(self.btn_quit)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_12)
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.label = QtWidgets.QLabel(self.page)
        self.label.setStyleSheet("font:bold 30px")
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.worker_enter_btn = QtWidgets.QPushButton(self.page)
        self.worker_enter_btn.setObjectName("worker_enter_btn")
        self.horizontalLayout.addWidget(self.worker_enter_btn)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem7)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem8)
        self.manager_enter_btn = QtWidgets.QPushButton(self.page)
        self.manager_enter_btn.setObjectName("manager_enter_btn")
        self.horizontalLayout_2.addWidget(self.manager_enter_btn)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem9)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        spacerItem10 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem10)
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.page_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem11)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem12)
        self.label_4 = QtWidgets.QLabel(self.page_2)
        self.label_4.setStyleSheet("font:bold 30px")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem13)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        spacerItem14 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem14)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem15)
        self.label_2 = QtWidgets.QLabel(self.page_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.lineEdit_worker_account = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_worker_account.setObjectName("lineEdit_worker_account")
        self.horizontalLayout_4.addWidget(self.lineEdit_worker_account)
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem16)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        spacerItem17 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem17)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem18 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem18)
        self.label_3 = QtWidgets.QLabel(self.page_2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.lineEdit_worker_password = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_worker_password.setObjectName("lineEdit_worker_password")
        self.horizontalLayout_5.addWidget(self.lineEdit_worker_password)
        spacerItem19 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem19)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        spacerItem20 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem20)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        spacerItem21 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem21)
        self.worker_register_btn = QtWidgets.QPushButton(self.page_2)
        self.worker_register_btn.setObjectName("worker_register_btn")
        self.horizontalLayout_6.addWidget(self.worker_register_btn)
        spacerItem22 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem22)
        self.start_work_btn = QtWidgets.QPushButton(self.page_2)
        self.start_work_btn.setObjectName("start_work_btn")
        self.horizontalLayout_6.addWidget(self.start_work_btn)
        spacerItem23 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem23)
        self.end_work_btn = QtWidgets.QPushButton(self.page_2)
        self.end_work_btn.setObjectName("end_work_btn")
        self.horizontalLayout_6.addWidget(self.end_work_btn)
        spacerItem24 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem24)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        spacerItem25 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem25)
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.page_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem26 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem26)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        spacerItem27 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem27)
        self.label_5 = QtWidgets.QLabel(self.page_3)
        self.label_5.setStyleSheet("font:bold 30px")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_11.addWidget(self.label_5)
        spacerItem28 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem28)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        spacerItem29 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem29)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        spacerItem30 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem30)
        self.label_6 = QtWidgets.QLabel(self.page_3)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_8.addWidget(self.label_6)
        self.lineEdit_manager_account = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_manager_account.setObjectName("lineEdit_manager_account")
        self.horizontalLayout_8.addWidget(self.lineEdit_manager_account)
        spacerItem31 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem31)
        self.verticalLayout_4.addLayout(self.horizontalLayout_8)
        spacerItem32 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem32)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem33 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem33)
        self.label_7 = QtWidgets.QLabel(self.page_3)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_10.addWidget(self.label_7)
        self.lineEdit_manager_password = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_manager_password.setObjectName("lineEdit_manager_password")
        self.horizontalLayout_10.addWidget(self.lineEdit_manager_password)
        spacerItem34 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem34)
        self.verticalLayout_4.addLayout(self.horizontalLayout_10)
        spacerItem35 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem35)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        spacerItem36 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem36)
        self.manager_login = QtWidgets.QPushButton(self.page_3)
        self.manager_login.setObjectName("manager_login")
        self.horizontalLayout_9.addWidget(self.manager_login)
        spacerItem37 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem37)
        self.verticalLayout_4.addLayout(self.horizontalLayout_9)
        spacerItem38 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem38)
        self.stackedWidget.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.page_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        spacerItem39 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_6.addItem(spacerItem39)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_8 = QtWidgets.QLabel(self.page_4)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_5.addWidget(self.label_8)
        self.tableview = QtWidgets.QTableView(self.page_4)
        self.tableview.setMinimumSize(QtCore.QSize(330, 0))
        self.tableview.setMaximumSize(QtCore.QSize(350, 16777215))
        self.tableview.setObjectName("tableview")
        self.verticalLayout_5.addWidget(self.tableview)
        spacerItem40 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_5.addItem(spacerItem40)
        self.horizontalLayout_13.addLayout(self.verticalLayout_5)
        spacerItem41 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_13.addItem(spacerItem41)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setContentsMargins(0, -1, -1, -1)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_worker = QtWidgets.QLabel(self.page_4)
        self.label_worker.setMinimumSize(QtCore.QSize(750, 450))
        self.label_worker.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_worker.setText("")
        self.label_worker.setObjectName("label_worker")
        self.verticalLayout_7.addWidget(self.label_worker)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_view_worker = QtWidgets.QLabel(self.page_4)
        self.label_view_worker.setMinimumSize(QtCore.QSize(300, 0))
        self.label_view_worker.setObjectName("label_view_worker")
        self.horizontalLayout_14.addWidget(self.label_view_worker)
        self.label_11 = QtWidgets.QLabel(self.page_4)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_14.addWidget(self.label_11)
        self.label_error = QtWidgets.QLabel(self.page_4)
        self.label_error.setMinimumSize(QtCore.QSize(300, 0))
        self.label_error.setText("")
        self.label_error.setObjectName("label_error")
        self.horizontalLayout_14.addWidget(self.label_error)
        spacerItem42 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem42)
        self.verticalLayout_7.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_13.addLayout(self.verticalLayout_7)
        self.horizontalLayout_13.setStretch(2, 1)
        self.verticalLayout_6.addLayout(self.horizontalLayout_13)
        self.stackedWidget.addWidget(self.page_4)
        self.verticalLayout.addWidget(self.stackedWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Intelligent manufacturing factory"))
        self.btn_quit.setText(_translate("MainWindow", "Quit"))
        self.label.setText(_translate("MainWindow", "Please select your identity"))
        self.worker_enter_btn.setText(_translate("MainWindow", "Worker"))
        self.manager_enter_btn.setText(_translate("MainWindow", "Manager"))
        self.label_4.setText(_translate("MainWindow", "Worker identity verification"))
        self.label_2.setText(_translate("MainWindow", "Accoun："))
        self.lineEdit_worker_account.setPlaceholderText(_translate("MainWindow", "Please enter your account number"))
        self.label_3.setText(_translate("MainWindow", "Password："))
        self.lineEdit_worker_password.setPlaceholderText(_translate("MainWindow", "Please enter password"))
        self.worker_register_btn.setText(_translate("MainWindow", "Register"))
        self.start_work_btn.setText(_translate("MainWindow", "Start work"))
        self.end_work_btn.setText(_translate("MainWindow", "Finish work"))
        self.label_5.setText(_translate("MainWindow", "Manager identity verification"))
        self.label_6.setText(_translate("MainWindow", "Account:"))
        self.lineEdit_manager_account.setPlaceholderText(_translate("MainWindow", "Please enter your account number"))
        self.label_7.setText(_translate("MainWindow", "Password:"))
        self.lineEdit_manager_password.setPlaceholderText(_translate("MainWindow", "Please enter password"))
        self.manager_login.setText(_translate("MainWindow", "Login"))
        self.label_8.setText(_translate("MainWindow", "Working"))
        self.label_view_worker.setText(_translate("MainWindow", "Please select to view workers"))
        self.label_11.setText(_translate("MainWindow", "Error type:"))

class Controller(object):
    # def __init__(self):
    #     self.error_tip = 0

    stage_flag = 0
    flag_0 = 0
    flag_1 = 0
    flag_2 = 0
    flag_3 = 0
    flag_4 = 0
    flag_5 = 0
    flag_6 = 0
    flag_7 = 0
    flag_8 = 0
    flag_9 = 0


    def get_shared_data(self,acf_list,obj_angle_list,det_list):
        global error_tip
        acf = 0
        obj_angle = 0
        det = 0
        flag = 0
        while True:
            print(error_tip,"error_tip")
            if len(acf_list):
                flag |= 1
            acf = acf_list.pop(-1) if len(acf_list) else acf
            print(type(acf))
            print("from controller act: ", acf)

            if len(obj_angle_list):
                flag |= 2
            obj_angle = obj_angle_list.pop(-1) if len(obj_angle_list) else obj_angle
            print("from controller obj_angel: ", obj_angle)

            if len(det_list):
                flag |= 4
            det = det_list.pop(-1) if len(det_list) else det
            print("from controller det_list: ", det)

            if flag >= 4:
                time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.states_controller(acf, obj_angle, det, time_now)

            time.sleep(0.5)

    def states_controller(self, acf, obj_angle, det, actual_time):
            global error_tip
        # with open("output.txt", "a+", encoding="utf-8") as file:
            target_0 = [0, 0, 0, 0, 0, 0]
            target_1 = [0, 0, 0, 0, 0, 0]
            target_2 = [0, 0, 0, 0, 0, 0]
            target_3 = [0, 0, 0, 0, 0, 0]
            target_4 = [0, 0, 0, 0, 0, 0]
            target_5 = [0, 0, 0, 0, 0, 0]
            target_6 = [0, 0, 0, 0, 0, 0]
            target_7 = [0, 0, 0, 0, 0, 0]
            target_8 = [0, 0, 0, 0, 0, 0]
            target_9 = [0, 0, 0, 0, 0, 0]

            target_5_temp = [0, 0, 0, 0, 0, 0]

            for target in det:
                if target[5] == 0:
                    target_0 = target
                    self.flag_0 = 1
                    print("0 Success")

                if target[5] == 1:
                    target_1 = target
                    self.flag_1 = 1#改吧
                    print("1 Success")

                if target[5] == 2:
                    target_2 = target
                    self.flag_2 = 1
                    print("2 Success")

                if target[5] == 3:
                    target_3 = target
                    self.flag_3 = 1
                    print("3 Success")

                if target[5] == 4:
                    target_4 = target
                    self.flag_4 = 1
                    print("4 Success")

                if target[5] == 5:
                    target_5 = target
                    self.flag_5 = 1
                    print("5 Success")

                if target[5] == 6:
                    target_6 = target
                    self.flag_6 = 1
                    print("6 Success")

                if target[5] == 7:
                    target_7 = target
                    self.flag_7 = 1
                    print("7 Success")

                if target[5] == 8:
                    target_8 = target
                    self.flag_8 = 1
                    print("8 Success")

                if target[5] == 9:
                    target_9 = target
                    self.flag_9 = 1
                    print("9 Success")

            if self.flag_5 and self.stage_flag == 0:
                print("Assembly start")
                # file.write(actual_time + " Assembly start\n")
                self.stage_flag += 1

            if self.stage_flag == 1:
                if target_1[0] >= target_5[0] and target_1[1] >= target_5[1] and target_1[2] <= target_5[2] and target_1[3] <= target_5[3]:
                    print("ActuatorBase assembled")
                    # file.write(actual_time + " ActuatorBase assembled\n")
                    error_tip = 1
                    self.stage_flag += 1

            if target_0[0] >= target_5[0] and target_0[1] >= target_5[1] and target_0[2] <= target_5[2] and target_0[3] <= target_5[3] and (target_5[1]-target_3[1]) >= 100 and self.stage_flag == 2:
                print("Arm & Electro assembled, wait to be screwed\n")
                # file.write(actual_time + " Arm & Electro assembled, wait to be screwed\n")
                error_tip = 2
                self.stage_flag += 1

            if self.stage_flag == 3:
                if acf[2] >= 0.5 or acf[1] >= 0.5:
                    print("Electro is under screwing\n")
                    # file.write(actual_time + " Electro is under screwing\n")
                    self.stage_flag += 1

            if target_2[0] >= target_5[0] and target_2[1] >= target_5[1] and target_2[2] <= target_5[2] and target_2[3] <= target_5[3] and self.stage_flag == 4:
                print("Actuator cover assembled, wait to be screwed")
                # file.write(actual_time + " Actuator cover assembled, wait to be screwed\n")
                error_tip = 3
                self.stage_flag += 1

            if self.stage_flag == 5:
                if acf[2] >= 0.5 or acf[1] >= 0.5:
                    print("Actuator cover is under screwing\n")
                    # file.write(actual_time + " Actuator cover is under screwing\n")
                    self.stage_flag += 1

            if -2 <= obj_angle <= 12 and target_7[0] >= target_5[0] and target_7[1] >= target_5[1] and target_7[2] <= target_5[2] and target_7[3] <= target_5[3] and self.stage_flag == 6:
                print("Platter assembled\n")
                # file.write(actual_time + " Platter assembled\n")
                error_tip = 4
                self.stage_flag += 1

            if target_9[0] >= target_7[0] and target_9[1] >= target_7[1] and target_9[2] <= target_7[2] and target_9[3] <= target_7[3] and self.stage_flag == 7:
                print("Spindle assembled, wait to be screwed\n")
                # file.write(actual_time + " Spindle assembled, wait to be screwed\n")
                error_tip = 5
                target_5_temp = target_5
                self.stage_flag += 1

            if self.stage_flag == 8:
                if acf[2] >= 0.5 or acf[1] >= 0.5:
                    print("Spindle is under screwing\n")
                    # file.write(actual_time + " Spindle is under screwing\n")
                    self.stage_flag += 1

            if self.flag_4 == 1 and target_4[0] - 10 <= target_5_temp[0] <= target_4[0] + 10 and self.stage_flag == 9:
                print("Case cover assembled, wait to be screwed\n")
                # file.write(actual_time + " Case cover assembled, wait to be screwed\n")
                error_tip = 6
                self.stage_flag += 1

            if self.stage_flag == 10:
                if acf[2] >= 0.5 or acf[1] >= 0.5:
                    print("Case cover is under screwing\n")
                    # file.write(actual_time + " Case cover is under screwing\n")
                    self.stage_flag += 1

            if target_6[2] <= target_5[2] and self.stage_flag == 11:
                print("Logi board assembled, wait to be screwed\n")
                # file.write(actual_time + " Logi board assembled, wait to be screwed\n")
                error_tip = 7
                self.stage_flag += 1

            if self.stage_flag == 12:
                if acf[2] >= 0.5 or acf[1] >= 0.5:
                    print("Logi board is under screwing\n")
                    # file.write(actual_time + " Logi board is under screwing\n")
                    self.stage_flag += 1

            if self.flag_4 == 1 and self.stage_flag == 13:
                print("Assembly Verified. Task total time: ")
                # file.write(actual_time + " Assembly Verified. Task total time: ")
                self.stage_flag += 1

class winlogin(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(winlogin, self).__init__()
        self.setupUi(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(1000)  # 每秒更新一次 UI
        '''界面居中显示'''
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))
        self.initial()
        
    def update_ui(self):
        global error_tip
        if error_tip == 1:
            self.Worker_error2()
            self.label_error.setText('ActuatorBase assembled')
        if error_tip == 2:
            self.label_error.setText('Arm & Electro assembled, wait to be screwed')
        if error_tip == 3:
            self.label_error.setText('Actuator cover assembled, wait to be screwed')
        if error_tip == 4:
            self.label_error.setText('Platter assembled')
        if error_tip == 5:
            self.label_error.setText('Spindle assembled, wait to be screwed')
        if error_tip == 6:
            self.label_error.setText('Case cover assembled, wait to be screwed')
        if error_tip == 7:
            self.label_error.setText('Logi board assembled, wait to be screwed')

    # 函数定义
    def initial(self):
        # 设置首页面
        self.stackedWidget.setCurrentIndex(0)
        self.btn_quit.setVisible(0)
        # 进入工人验证页面
        self.worker_enter_btn.clicked.connect(lambda: (self.stackedWidget.setCurrentIndex(1), self.lineEdit_worker_account.clear(), self.lineEdit_worker_password.clear(), self.btn_quit.setVisible(1)))
        # 进入主管验证页面
        self.manager_enter_btn.clicked.connect(lambda: (self.stackedWidget.setCurrentIndex(2), self.lineEdit_manager_account.clear(), self.lineEdit_manager_password.clear(), self.btn_quit.setVisible(1)))
        # 退出按钮返回首页
        self.btn_quit.clicked.connect(lambda: (self.stackedWidget.setCurrentIndex(0), self.btn_quit.setVisible(0)))
        # 注册
        try:
            self.worker_register_btn.clicked.disconnect()  # 尝试断开之前的连接
        except TypeError:
            pass  # 如果连接不存在，就不进行断开
        self.viewing_worker = ''
        self.worker_register_btn.clicked.connect(self.Worker_register)
        self.start_work_btn.clicked.connect(self.StartWork)
        self.end_work_btn.clicked.connect(self.EndWork)
        self.manager_login.clicked.connect(self.Manager_login)

    # 主管登录 设置主管manager密码123
    def Manager_login(self):
        manager_account = self.lineEdit_manager_account.text()
        manager_password = self.lineEdit_manager_password.text()
        # print(manager_password, manager_account)
        if manager_account == 'manager' and manager_password == '123':
            self.stackedWidget.setCurrentIndex(3)
            self.model = QStandardItemModel()
            self.tableview.setModel(self.model)
            self.tableview.clicked.connect(self.on_tableview_clicked)
            # 更新列表
            self.populate_table()
            # self.Worker_error()
        else:
            QMessageBox.information(self, 'Error', 'The account password is incorrect')

    # 更新正在工作的工人列表 获取的是正在工作中的工作列表 Worker表对应的Work值为1
    def populate_table(self):
        try:
            with db.cursor() as cur:
                # 执行 SQL 查询以获取工人信息
                query = "SELECT workername, starttime FROM Worker WHERE work = '1'"
                cur.execute(query)
                worker_data = cur.fetchall()

                # 设置带有列和标题的模型
                self.model.setColumnCount(3)
                self.model.setHorizontalHeaderLabels(['Worker', 'Time-on', 'Action'])

                # 使用工人信息填充表格
                for row, worker_info in enumerate(worker_data):
                    worker_name, start_time = worker_info

                    # 在相应列中设置工人和开始时间
                    worker_item = QStandardItem(worker_name)
                    start_time_item = QStandardItem(start_time)
                    self.model.setItem(row, 0, worker_item)
                    self.model.setItem(row, 1, start_time_item)

                    # 在第三列创建一个按钮
                    button = QPushButton("View")
                    button.clicked.connect(lambda _, name=worker_name: self.view_worker_info(name))
                    self.model.setItem(row, 2, QStandardItem())
                    self.tableview.setIndexWidget(self.model.index(row, 2), button)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error: {str(e)}")
        # finally:
        #     # 关闭游标和连接
        #     db.close()

    def on_tableview_clicked(self, index):
        # 检查点击的项是否是“操作”列中的按钮
        if index.column() == 2:
            worker_name = self.model.item(index.row(), 0).text()
            self.view_worker_info(worker_name)

    # 界面显示正在查看的工人信息
    def view_worker_info(self, worker_name):
        self.viewing_worker = worker_name
        # 单击按钮时显示工人信息
        self.label_view_worker.setText(f"Viewing workers：{worker_name}")

        @smart_inference_mode()
        def run(
                weights= 'yolov5s.pt',  # model path or triton URL
                source= '0',  # file/dir/URL/glob/screen/0(webcam)
                data= 'data/coco.yaml',  # dataset.yaml path
                imgsz=(640, 640),  # inference size (height, width)
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                view_img=False,  # show results
                save_txt=False,  # save results to *.txt
                save_conf=False,  # save confidences in --save-txt labels
                save_crop=False,  # save cropped prediction boxes
                nosave=False,  # do not save images/videos
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
                update=False,  # update all models
                project=ROOT / 'runs/detect',  # save results to project/name
                name='exp',  # save results to project/name
                exist_ok=False,  # existing project/name ok, do not increment
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                half=False,  # use FP16 half-precision inference
                dnn=False,  # use OpenCV DNN for ONNX inference
                vid_stride=1,  # video frame-rate stride
        ):
            source = str(source)
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
            is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
            webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
            screenshot = source.lower().startswith('screen')
            if is_url and is_file:
                source = check_file(source)  # download

            # Directories
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            device = select_device(device)
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Dataloader
            bs = 1  # batch_size
            if webcam:
                view_img = check_imshow(warn=True)
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
            elif screenshot:
                dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            sequence = []

            for path, im, im0s, vid_cap, s, imR in dataset:
                # img_orr = copy.deepcopy(im)
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Process predictions

                for i, det in enumerate(pred):  # per image
                    seen += 1
                    shared_data = multiprocessing.Array('d', 3)
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        img_orr = im0.copy()
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                    img = im0.copy() if save_crop else im0
                    image_mp, results = mediapipe_detection(img, holistic)

                    # Mediapipe part: Draw landmarks
                    draw_styled_landmarks(image_mp, results)
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    #when the last 30 frames are collected
                    if len(sequence) == 30:
                        acf = model1.predict(np.expand_dims(sequence, axis=0))[0]#你这里出来的不是int吗？是，就是很奇怪啊不懂所以说他是从int->ndarray->int?两边一样的写法，结果不一样
                        print("acf:::",type(acf))#对不起看不懂 这真的合理吗
                        acf_list.append(acf)  #但如果acf是ndarray，那append进去不是也是array吗
                        # print(type(acf_list))
                        # print('Actions confidence:', acf)
                        # cv2.imshow('mediapipe', image_mp)

                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    # Stream results
                    im0 = annotator.result()
                    img_ang = im0.copy()
                    det_rev = reversed(det)
                    det_list.append(det_rev)
                    # print(det_list)
                    for num111 in range(0, len(det_rev)):  # det_rev[[][0]] = []:
                        if det_rev[num111][5] == 0:
                            img_set = []
                            x1 = det_rev[num111]
                            if abs(x1[1] - x1[3]) > abs(x1[0] - x1[2]):
                                x1[2] = x1[0] + abs(x1[1] - x1[3])
                            else:
                                x1[3] = x1[1] + abs(x1[0] - x1[2])
                            cropped_image = img_orr[int(x1[1]):int(x1[3]), int(x1[0]):int(x1[2])]
                            cv2.imwrite("cropped_image.jpg", cropped_image)

                            in_img = cv2.resize(cropped_image, (100, 100))
                            transf = transforms.ToTensor()
                            in_img = transf(in_img)  # tensor数据格式是torch(C,H,W)
                            img_set.append(in_img)

                            # print(angle_set_in,angle_set_in.size())
                            img_set_in = torch.stack(img_set, dim=0)
                            output = model_ang(img_set_in)  # 调用模型预测
                            output_array = np.array(output)
                            obj_angle = output_array[0][0]
                            obj_angle_list.append(obj_angle)
                            print('Arm angle is:', obj_angle)
                            # print(cnn_train.weights)

                    # if view_img:
                    #     if platform.system() == 'Linux' and p not in windows:
                    #         windows.append(p)
                    #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    #     # cv2.imshow('Detection_System', im0)
                    #     cv2.waitKey(1)  # 1 millisecond

                        img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                        qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimage)
                        self.label_worker.setPixmap(pixmap)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer[i].write(im0)

                key = cv2.waitKey(1)
                if key ==27:
                    break

                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

            # Print results
            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
            return save_path

        def parse_opt():
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str,
                                default='best2.pt',
                                help='model path or triton URL')
            parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
            # parser.add_argument('--source', type=str, default= 'D:/Yolov5/yolov5-master/data/JPEGimages', help='file/dir/URL/glob, 0 for webcam')
            parser.add_argument('--data', type=str, default='data/disk.yaml',
                                help='(optional) dataset.yaml path')
            parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
            parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.05, help='NMS IoU threshold')
            parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='show results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
            parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--visualize', action='store_true', help='visualize features')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
            parser.add_argument('--name', default='exp', help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
            parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
            parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
            parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
            parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
            parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
            parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
            opt = parser.parse_args()
            opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
            print_args(vars(opt))
            return opt

        def detect_main(option):
            check_requirements(exclude=('tensorboard', 'thop'))
            controller = Controller()
            t1 = Thread(target=controller.get_shared_data, args=(acf_list, obj_angle_list, det_list))
            t1.start()
            det_res = run(**vars(option))
            t1.join()
            return det_res
        # def detect_main(opt):
        #     check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
        #     return run(**vars(opt))

        opt: Namespace = parse_opt()
        detect_main(opt)

    def Worker_error2(self):
        if self.viewing_worker:
            current_time = datetime.now()
            # 将当前时间转换为字符串格式
            current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            error_type = "ActuatorArmStepError"
            # insert_sql = f"INSERT INTO {Records}(errortime,workername,errortype) VALUES ('%s','%s',%s)" % (current_time_str, self.worker_name, error_type)
            insert_sql = "INSERT INTO Records(errortime,workername,errortype) VALUES (%s,%s,%s)"
            cur.execute(insert_sql,(current_time_str,self.viewing_worker,error_type))
            db.commit()
            self.show_error_box(error_type)
        else:
            QMessageBox.information(self, 'Error', 'Please select a worker to view')

    # 设置5秒后关闭弹窗
    def show_error_box(self, error_type):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Error Type")
        msg_box.setText(f"The error type is: {error_type}")
        msg_box.setIcon(QMessageBox.Information)

        timer = QTimer(self)
        timer.timeout.connect(msg_box.close)
        timer.start(5000)  # 5000 milliseconds = 5 seconds

        msg_box.exec_()

    # 工人注册
    def Worker_register(self):
        worker_account = self.lineEdit_worker_account.text()
        worker_password = self.lineEdit_worker_password.text()
        # 向Worker数据表中插入语句
        insert_sql = f"INSERT INTO {Worker}(workername,password) VALUES ('%s','%s')" % (worker_account, worker_password)
        # 读取Worker数据表中的username和password字段值
        read_sql = f'''select * from {Worker} where workername = "{worker_account}"'''
        user_data = cur.execute(read_sql)
        # 判断注册账号和密码
        if user_data.real:
            QMessageBox.critical(self, "Error", "The registered account already exists! Please check")
        else:
            cur.execute(insert_sql)
            db.commit()
            QMessageBox.information(self, "Welcome", "Registration successful!")

    # 工人开始工作，并记录在数据库表中 0代表工作已结束 1代表已开始
    def StartWork(self):
        worker_account = self.lineEdit_worker_account.text()
        worker_password = self.lineEdit_worker_password.text()
        # 执行SQL语句，从Worker数据表中查询workername和password字段值
        cur.execute(f"SELECT workername,password FROM {Worker}")
        # 将数据库查询的结果保存在result中
        result = cur.fetchall()
        name_list = [it[0] for it in result]  # 从数据库查询的result中遍历查询元组中第一个元素name
        # 判断用户名或密码不能为空
        if not (worker_account and worker_password):
            QMessageBox.critical(self, "Error", "Worker account or password cannot be empty!")
            # 判断用户名和密码是否匹配
        elif worker_account in name_list:
            if worker_password == result[name_list.index(worker_account)][1]:
                current_time = datetime.now()
                # 将当前时间转换为字符串格式
                current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                # 更
                # 新特定条件下的数据
                sql = "UPDATE worker SET starttime = %s, work = 1 WHERE workername = %s"
                val = (current_time_str, worker_account)
                cur.execute(sql, val)
                # 提交更改
                db.commit()
                print(cur.rowcount, "Record update successful")
                QMessageBox.information(self, "Welcome", "Started working")
            else:
                QMessageBox.critical(self, "Error", "Password input error！")
        # 账号不在数据库中，则弹出是否注册的框
        else:
            QMessageBox.critical(self, "Error", "The account does not exist, please register!")

    # 工人结束工作 更新数据库表为0
    def EndWork(self):
        worker_account = self.lineEdit_worker_account.text()
        worker_password = self.lineEdit_worker_password.text()
        # 执行SQL语句，从Worker数据表中查询workername和password字段值
        cur.execute(f"SELECT workername,password FROM {Worker}")
        # 将数据库查询的结果保存在result中
        result = cur.fetchall()
        name_list = [it[0] for it in result]  # 从数据库查询的result中遍历查询元组中第一个元素name
        # 判断用户名或密码不能为空
        if not (worker_account and worker_password):
            QMessageBox.critical(self, "Error", "Worker account or password cannot be empty！")
            # 判断用户名和密码是否匹配
        elif worker_account in name_list:
            if worker_password == result[name_list.index(worker_account)][1]:
                current_time = datetime.now()
                # 将当前时间转换为字符串格式
                current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                # 更新特定条件下的数据
                sql = "UPDATE worker SET starttime = %s, work = 0 WHERE workername = %s"
                val = (current_time_str, worker_account)
                cur.execute(sql, val)
                # 提交更改
                db.commit()
                # print(cur.rowcount, "Record update successful")
                QMessageBox.information(self, "Completed", "Completed work")
            else:
                QMessageBox.critical(self, "Error", "Password input error！")
        # 账号不在数据库中，则弹出是否注册的框
        else:
            QMessageBox.critical(self, "Error", "The account does not exist, please register!")

# def local_print():
#     while 1:
#         if(len(acf_list)>0):
#             print("from local" ,acf_list.pop(-1))
#         sleep(1)

# def t1_task():
#     controller = Controller()
#     controller.get_shared_data(args=(acf_list, obj_angle_list, det_list))

def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    loginUi = winlogin()  # 将窗口换个名字
    loginUi.show()  # 将窗口显示出来
    sys.exit(app.exec_())

if __name__ == "__main__":

    main()
