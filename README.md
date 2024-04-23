# AAS (Auto-inspection Assembly System)
A program with gesture recognition and object detection, helping people assemble in a faster and more reliable way.

Development Conditions:

- Windows 11 / Ubuntu 20.04
- Python 3.9.18 in PyCharm
- ZED SDK 4.0
- Mediapipe with CUDA 11.1 & CUDNN 8.6.0.163/8005
- PyTorch 1.9.1 + cu111
- Tensorflow 2.15.0

For **Users**:

1. run the detect_yt.py or detect.exe (not fully develped yet)
2. follow the instructions on the screen and make sure your camera has been connected to computer correctly.

For **Developers**:

1. Download Anaconda and PyCharm (or any other IDE )
2. Check your CUDA available edition and the corresponding CUDNN edition in your NVIDIA Control Panel, install both and test if they're installed correctly.
3. Download ZED SDK on the official website of Stereolab, follow its install instructions.
4. Run get_python_api.py in folder: ZED SDK, you may encounter problems like "URL Error" which means you have not fully installed dependencies. If your python edition is same as mine, you could find the 2 other corresponding dependency files in the folder "Dependency", install them in the same environment instead and then you could use your ZED camera freely.
5. Define the actions you need in "gesture_collect_tf.py" and run it, then you could enjoy the auto dataset building.
6. After collecting, start training by "gesture_train_tf.py", and then test whether it fits your imagination by calling "gesture_detect_tf.py"
7. Complete your object dataset with YOLO and put the weight file under the same folder as above.
8. Define transition conditions in "Controller.py“ and run "detect_yt.py", and you have built your own Auto-inspection Assembly System. Congratulations!

The project is presented as my Final Year Project in NUS Research Institute,if you're interested in more details, please contact me at sinyzxj@outlook.com or visit my homepage: sinyzxj.github.io

:-)

