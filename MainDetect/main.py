import cv2
import pyzed.sl as sl
import os
import numpy as np
import mediapipe as mp
import math

#指定mediapipe的基本参数
mp_drawing = mp.solutions.drawing_utils
mp_dstyles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

#指定相机的基本参数
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
runtime_parameters = sl.RuntimeParameters()

image = sl.Mat()
mlist = []

#打开ZED 2i相机
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

with mp_holistic.Holistic(
    min_detection_confidence=0.1,
    min_tracking_confidence=0.5) as holistic:
    while err == sl.ERROR_CODE.SUCCESS:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:#两层判断，确保正确捕捉图像
            zed.retrieve_image(image, sl.VIEW.LEFT)
            img = image.get_data()

            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img)

            #画图
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_dstyles.get_default_pose_landmarks_style())

            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            w = image.get_width()
            h = image.get_height()
            # 右手21个节点坐标
            if results.right_hand_landmarks:

                # for index, landmarks in enumerate(results.right_hand_landmarks.landmark):
                #     mlist.append([int(landmarks.x * w) ,int(landmarks.y * h),landmarks.z])#反归一化，单位为厘米
                #     narray = np.copy(mlist)
                #     print(index, narray[index][0])

                #OK手势检测
                point4 = np.array([results.right_hand_landmarks.landmark[4].x, results.right_hand_landmarks.landmark[4].y, results.right_hand_landmarks.landmark[4].z])
                point8 = np.array([results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y, results.right_hand_landmarks.landmark[8].z])
                dist = np.sqrt(np.sum(np.square(point4-point8)))
                print(dist)
                if dist < 0.015 :
                    cv2.putText(img,"OKAY",[1200,400],cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)
                    print("ok")

            cv2.imshow('MediaPipe Holistic', cv2.flip(img, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break
zed.close()



