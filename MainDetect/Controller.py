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
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots#, LoadStreams
from dataloaders2 import LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import multiprocessing
import time
from time import sleep, ctime
import UI
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from Mysql_Setting import db, connection
from datetime import datetime
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer

class Controller(object):
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
        acf = 0
        obj_angle = 0
        det = 0
        flag = 0
        while True:
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
                time_cal = time.time()
                self.states_controller(acf, obj_angle, det, time_now)

            time.sleep(0.5)

    def states_controller(self, acf, obj_angle, det, actual_time):
        with open("output.txt", "a+", encoding="utf-8") as file:
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

            page_4 = QtWidgets.QWidget()
            page_4.setObjectName("page_4")

            label_error = QtWidgets.QLabel(page_4)
            label_error.setMinimumSize(QtCore.QSize(300, 0))
            label_error.setText("")
            label_error.setObjectName("label_error")

            for target in det:
                if target[5] == 0:
                    target_0 = target
                    self.flag_0 = 1
                    print("0 Success")

                if target[5] == 1:
                    target_1 = target
                    self.flag_1 = 1
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
                file.write(actual_time + " Assembly start\n")
                self.stage_flag += 1

            if self.stage_flag == 1:
                if target_1[0] >= target_5[0] and target_1[1] >= target_5[1] and target_1[2] <= target_5[2] and target_1[3] <= target_5[3]:
                    print("ActuatorBase assembled")
                    file.write(actual_time + " ActuatorBase assembled\n")
                    self.stage_flag += 1

            if target_0[0] >= target_5[0] and target_0[1] >= target_5[1] and target_0[2] <= target_5[2] and target_0[3] <= target_5[3] and (target_5[1]-target_3[1]) >= 100 and self.stage_flag == 2:
                print("Arm & Electro assembled, wait to be screwed\n")
                file.write(actual_time + " Arm & Electro assembled, wait to be screwed\n")
                self.stage_flag += 1

            if self.stage_flag == 3:
                if acf[2] >= 0.4:
                    print("Electro is under screwing\n")
                    file.write(actual_time + " Electro is under screwing\n")
                    self.stage_flag += 1

            if target_2[0] >= target_5[0] and target_2[1] >= target_5[1] and target_2[2] <= target_5[2] and target_2[3] <= target_5[3] and self.stage_flag == 4:
                print("Actuator cover assembled, wait to be screwed")
                file.write(actual_time + " Actuator cover assembled, wait to be screwed\n")
                self.stage_flag += 1

            if self.stage_flag == 5:
                if acf[2] >= 0.4:
                    print("Actuator cover is under screwing\n")
                    file.write(actual_time + " Actuator cover is under screwing\n")
                    self.stage_flag += 1

            if -2 <= obj_angle <= 12 and target_7[0] >= target_5[0] and target_7[1] >= target_5[1] and target_7[2] <= target_5[2] and target_7[3] <= target_5[3] and self.stage_flag == 6:
                print("Platter assembled\n")
                file.write(actual_time + " Platter assembled\n")
                self.stage_flag += 1

            if target_9[0] >= target_7[0] and target_9[1] >= target_7[1] and target_9[2] <= target_7[2] and target_9[3] <= target_7[3] and self.stage_flag == 7:
                print("Spindle assembled, wait to be screwed\n")
                file.write(actual_time + " Spindle assembled, wait to be screwed\n")
                target_5_temp = target_5
                self.stage_flag += 1

            if self.stage_flag == 8:
                if acf[2] >= 0.4:
                    print("Spindle is under screwing\n")
                    file.write(actual_time + " Spindle is under screwing\n")
                    self.stage_flag += 1

            if self.flag_4 == 1 and target_4[0] - 10 <= target_5_temp[0] <= target_4[0] + 10 and self.stage_flag == 9:
                print("Case cover assembled, wait to be screwed\n")
                file.write(actual_time + " Case cover assembled, wait to be screwed\n")
                self.stage_flag += 1

            if self.stage_flag == 10:
                if acf[2] >= 0.5:
                    print("Case cover is under screwing\n")
                    file.write(actual_time + " Case cover is under screwing\n")
                    self.stage_flag += 1

            if target_6[2] <= target_5[2] and self.stage_flag == 11:
                print("Logi board assembled, wait to be screwed\n")
                file.write(actual_time + " Logi board assembled, wait to be screwed\n")
                self.stage_flag += 1

            if self.stage_flag == 12:
                if acf[2] >= 0.4:
                    print("Logi board is under screwing\n")
                    file.write(actual_time + " Logi board is under screwing\n")
                    self.stage_flag += 1

            if self.flag_4 == 1 and self.stage_flag == 13:
                end_time = calcu_time
                execution_time = end_time - start_time
                print("Assembly Verified. Task total time(s): ", execution_time)
                file.write(actual_time + " Assembly Verified. Task total time(s): " + execution_time)
                self.stage_flag += 1







