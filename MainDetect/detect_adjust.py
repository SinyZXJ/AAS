# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
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
import pyzed.sl as sl

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import sys
import UI
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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


cur = db.cursor()
Worker = 'Worker'
Records = 'Records'
class winlogin(UI.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(winlogin, self).__init__()
        self.setupUi(self)
        '''界面居中显示'''
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))
        self.initial()

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
        # self.Worker_error()

        @smart_inference_mode()
        def run(
                weights=ROOT / 'yolov5s.pt',  # model path or triton URL
                source=ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
                data=ROOT / 'data/coco.yaml',  # dataset.yaml path
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
            webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
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
            change = 0
            Platter_count = 0
            ActuatorBase_count = 0
            Spindle_count = 0
            sum_count = 0
            ActuatorArm_count = 0
            ActuatorCover_count = 0
            CaseCover_count = 0
            error_type = 'error'
            xx = 0
            aa = 0
            bb = 0
            cc = 0

            for path, im, im0s, vid_cap, s, *_ in dataset:
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
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        sum_count = Platter_count + ActuatorBase_count + Spindle_count + ActuatorArm_count + ActuatorCover_count + CaseCover_count
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)



                            if sum_count == 0:
                                self.label_error.setText('No Error')
                                if int(cls) == 7:
                                    Platter_count += 1
                                if int(cls) == 1:
                                    ActuatorBase_count += 1
                                if int(cls) == 9:
                                    Spindle_count += 1
                                if int(cls) == 0:
                                    ActuatorArm_count += 1
                                if int(cls) == 2:
                                    ActuatorCover_count += 1
                                if int(cls) == 4:
                                    CaseCover_count += 1
                                sum_count = Platter_count + ActuatorBase_count + Spindle_count + ActuatorArm_count + ActuatorCover_count + CaseCover_count

                            if sum_count == 1 and ActuatorBase_count == 1:
                                change = 1
                                self.label_error.setText('ActuatorBase Correct')
                                if int(cls) == 7:
                                    Platter_count += 1
                                if int(cls) == 9:
                                    Spindle_count += 1
                                if int(cls) == 0:
                                    ActuatorArm_count += 1
                                if int(cls) == 2:
                                    ActuatorCover_count += 1
                                if int(cls) == 4:
                                    CaseCover_count += 1
                                sum_count = Platter_count + ActuatorBase_count + Spindle_count + ActuatorArm_count + ActuatorCover_count + CaseCover_count


                            if sum_count == 1 and ActuatorBase_count != 1:
                                self.label_error.setText('ActuatorBase Step Error')
                                error_type = 'ActuatorBase Step Error'

                            if sum_count == 2 and ActuatorBase_count == 1 and ActuatorArm_count == 1:
                                change = 2
                                self.label_error.setText('ActuatorArm Correct')
                                if int(cls) == 7:
                                    Platter_count += 1
                                if int(cls) == 9:
                                    Spindle_count += 1
                                if int(cls) == 2:
                                    ActuatorCover_count += 1
                                if int(cls) == 4:
                                    CaseCover_count += 1
                                ActuatorBase_count == 1
                                xx = 1
                                sum_count = Platter_count + ActuatorBase_count + Spindle_count + ActuatorArm_count + ActuatorCover_count + CaseCover_count

                            if sum_count == 2 and ActuatorArm_count != 1:
                               self.label_error.setText('ActuatorArm Step Error')
                               error_type = 'ActuatorArm Step Error'


                            if sum_count == 3 and ActuatorArm_count == 1 and ActuatorCover_count == 1 and xx == 1:
                                change = 3
                                self.label_error.setText('ActuatorCover Correct')
                                if int(cls) == 9:
                                    Spindle_count += 1
                                if int(cls) == 7:
                                    Platter_count += 1
                                if int(cls) == 4:
                                    CaseCover_count += 1
                                aa = 1
                                sum_count = Platter_count + ActuatorBase_count + Spindle_count + ActuatorArm_count + ActuatorCover_count + CaseCover_count

                            if sum_count == 3 and ActuatorCover_count != 1 and xx == 1:
                                self.label_error.setText('ActuatorCover Step Error')
                                error_type = 'ActuatorCover Step Error'

                            if sum_count == 4 and Platter_count == 1 and ActuatorCover_count == 1 and ActuatorArm_count == 1 and aa == 1:
                                change = 4
                                self.label_error.setText('Platter Correct')
                                if int(cls) == 9:
                                    Spindle_count += 1
                                if int(cls) == 4:
                                    CaseCover_count += 1
                                ActuatorBase_count == 1
                                bb = 1
                                sum_count = Platter_count + ActuatorBase_count + Spindle_count + ActuatorArm_count + ActuatorCover_count + CaseCover_count

                            if sum_count == 4 and Platter_count != 1 and aa == 1:
                                self.label_error.setText('Platter Step Error')
                                error_type = 'Platter Step Error'


                            if sum_count == 5 and Platter_count == 1 and Spindle_count == 1 and ActuatorArm_count == 1 and ActuatorCover_count == 1 and bb == 1:
                                change = 5
                                self.label_error.setText('Spindle Correct')
                                if int(cls) == 4:
                                    CaseCover_count += 1
                                ActuatorBase_count == 1
                                cc = 1
                                sum_count = Platter_count + ActuatorBase_count + Spindle_count + ActuatorArm_count + ActuatorCover_count + CaseCover_count

                            if sum_count == 5 and Spindle_count != 1 and bb == 1:
                                self.label_error.setText('Spindle Step Error')
                                error_type = 'Spindle Step Error'

                            if sum_count == 6 and Platter_count == 1 and Spindle_count == 1 and ActuatorArm_count == 1 and ActuatorCover_count == 1 and CaseCover_count == 1 and cc == 1:
                                change = 6
                                ActuatorBase_count == 1
                                self.label_error.setText('CaseCover Correct')
                                error_type = 'No Error'

                            if sum_count == 6 and CaseCover_count != 1 and cc == 1:
                                self.label_error.setText('CaseCover Step Error')
                                error_type = 'CaseCover Step Error'


                            if change == 1:
                                ActuatorBase_count = 1
                            if change == 2:
                                ActuatorBase_count = 1
                                ActuatorArm_count = 1

                            if change == 3:
                                ActuatorCover_count = 1
                                ActuatorArm_count = 1
                                ActuatorBase_count = 1
                            if change == 4:
                                Platter_count = 1
                                ActuatorCover_count = 1
                                ActuatorBase_count = 1
                                ActuatorArm_count = 1
                            if change == 5:
                                Platter_count = 1
                                Spindle_count = 1
                                ActuatorBase_count = 1
                                ActuatorArm_count = 1
                                ActuatorCover_count = 1
                            if change == 6:
                                Platter_count = 1
                                Spindle_count = 1
                                ActuatorBase_count = 1
                                ActuatorArm_count = 1
                                ActuatorCover_count = 1
                                CaseCover_count = 1

                        # cv2.imshow(str(p), im0)
                        # 显示到 label_worker 上
                            print(sum_count)

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
                                save_path = str(
                                    Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer[i].write(im0)

                key = cv2.waitKey(1)
                if key == 27:
                    break
                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

            # Print results
            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(
                f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
            return save_path

        def parse_opt():

            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str,
                                default=ROOT / 'D:/Yolov5/yolov5-master/runs/train/exp129/weights/best.pt',
                                help='model path or triton URL')
            parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
            # parser.add_argument('--source', type=str, default= 'D:/Yolov5/yolov5-master/data/JPEGimages', help='file/dir/URL/glob, 0 for webcam')
            parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml',
                                help='(optional) dataset.yaml path')
            parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                                help='inference size h,w')
            parser.add_argument('--conf-thres', type=float, default=0.85, help='confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
            parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='show results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
            parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int,
                                help='filter by class: --classes 0, or --classes 0 2 3')
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

        def detect_main(opt):
            check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
            return run(**vars(opt))

        opt: Namespace = parse_opt()
        detect_main(opt)

        # QMessageBox.information(self, "工人信息", f"查看 {worker_name} 的信息")

     #工人错误
    def Worker_error2(self):
        if self.viewing_worker:
            current_time = datetime.now()
            # 将当前时间转换为字符串格式
            current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            error_type = "ActuatorArmStepError"
            # insert_sql = f"INSERT INTO {Records}(errortime,workername,errortype) VALUES ('%s','%s',%s)" % (
            # current_time_str,now_worker_name,error_type)
            # insert_sql = "INSERT INTO Records(errortime,workername,errortype) VALUES (%s,%s,%s)"
            # cur.execute(insert_sql,(current_time_str,self.viewing_worker,error_type))
            # db.commit()
            # self.show_error_box(error_type)
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
                # 更新特定条件下的数据
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


def main():
    '''防止界面变形'''
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    loginUi = winlogin()  # 将窗口换个名字
    loginUi.show()  # 将窗口显示出来
    sys.exit(app.exec_())  # app.exet_()是指程序一直循环运行直到主窗口被关闭终止进程（如果没有这句话，程序运行时会一闪而过）

if __name__ == '__main__':

    main()


# cap = cv2.VideoCapture(0)# 或者使用摄像头：
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(static_image_mode=False,
#                           max_num_hands=2,
#                           min_detection_confidence=0.5,
#                           min_tracking_confidence=0.5)
# mpDraw = mp.solutions.drawing_utils
# pTime = 0
# cTime = 0
#
# while True:
#     print("111")
#     success, img = cap.read()  # 读取一帧
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#     #detections = object_detection_model(frame)# 运行对象检测模型并获取检测结果
#
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 print(id, lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 # if id ==0:
#                 cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#     cv2.imshow("Real-time Object Detection", img)
#
#     if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# #
