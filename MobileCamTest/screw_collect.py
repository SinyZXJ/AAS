import cv2
import os
import numpy as np

savepath = "G:\ULTRALYTICS\MobileCamTest\Screw_data\images\train"

cv2.namedWindow('camera', 1)

video = "http://admin:admin@172.25.11.89:8081/video"
cap = cv2.VideoCapture(video)

while True:
    success, img = cap.read()
    cv2.imshow("camera", img)
    # 按键处理
    key = cv2.waitKey(10)
    if key == 27:
        # esc
        break
    if cv2.waitKey(5) & 0xFF == ord('s'):
        # 空格
        file_name = 'frames.jpg'
        cv2.imwrite(file_name, img)

# 释放摄像头
cap.release()
# 关闭窗口
cv2.destroyWindow("camera")