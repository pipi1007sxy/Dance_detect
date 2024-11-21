# 引入相关库

import argparse
import cv2
import numpy as np
import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
import time
import mediapipe as mp
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QRect
import os
import sys
from pathlib import Path
import cv2

import time
import cv2

import os
import random
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
import argparse

import math
from typing import List, Mapping, Optional, Tuple, Union

import cv2
import dataclasses
import matplotlib.pyplot as plt

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 提供CMD接口
parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
parser.add_argument('--checkpoint-path', type=str, help='path to the checkpoint')
parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
parser.add_argument('--track', type=int, default=1, help='track pose id in video')
parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
args = parser.parse_args()

# 读取图片
class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img

# 读取视频
class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

# 轻量化预测
# def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
#                pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
#     height, width, _ = img.shape
#     scale = net_input_height_size / height
#
#     scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#     scaled_img = normalize(scaled_img, img_mean, img_scale)
#     min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
#     padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
#
#     tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
#     if not cpu:
#         tensor_img = tensor_img.cuda()
#
#     stages_output = net(tensor_img)
#
#     stage2_heatmaps = stages_output[-2]
#     heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
#     heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
#
#     stage2_pafs = stages_output[-1]
#     pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
#     pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
#
#     return heatmaps, pafs, scale, pad


class PoseEstimation:
    def __init__(self, checkpoint_path, height_size=256, cpu=False, track=1, smooth=1):
        self.net = PoseEstimationWithMobileNet()
        self.net = load_state(self.net, checkpoint_path)
        self.height_size = height_size
        self.cpu = cpu
        self.track = track
        self.smooth = smooth

    def infer_fast(self, img, net_input_height_size, stride, upsample_ratio, cpu,
                   pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu:
            tensor_img = tensor_img.cuda()

        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def run_demo(self, image_provider):
        self.net = self.net.eval()
        if not self.cpu:
            self.net = self.net.cuda()

        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        previous_poses = []
        delay = 1
        for img in image_provider:
            orig_img = img.copy()
            heatmaps, pafs, scale, pad = self.infer_fast(img, self.height_size, stride, upsample_ratio, self.cpu)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)

            if self.track:
                track_poses(previous_poses, current_poses, smooth=self.smooth)
                previous_poses = current_poses
            for pose in current_poses:
                pose.draw(img)
            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                if self.track:
                    cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
            key = cv2.waitKey(delay)
            if key == 27:  # esc
                return
            elif key == 112:  # 'p'
                if delay == 1:
                    delay = 0
                else:
                    delay = 1

# 运行的Demo
def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = PoseEstimation.infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1
# 加载OPENPOSE模型
# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# 定义置信度阈值
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

# 定义相关颜色
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

# 归一化坐标
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """将标准化值对转换为像素坐标."""

  # 检查浮点值是否介于0和1之间.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: 绘制坐标，即使它在图像边界之外.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px
# 绘制点的粗细半径
class DrawingSpec:
  # 用于绘制注释的颜色。默认为白色.
  color: Tuple[int, int, int] = WHITE_COLOR
  # 用于绘制注释的厚度。默认为2像素.
  thickness: int = 2
  # 圆半径。默认为2像素.
  circle_radius: int = 2

# 绘制检测点在图上
def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec,
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec(),
    color = [0,255,0]
    ):

  if not landmark_list:
    return
  if image.shape[2] != _RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  if connections:
    num_landmarks = len(landmark_list.landmark)
    # 如果起点和终点标志都可见，则绘制连接.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], drawing_spec.color,
                 drawing_spec.thickness)
  # 完成连接线后绘制界标点，即
  # 美观性更好.
  if landmark_drawing_spec:
    for idx, landmark_px in idx_to_coordinates.items():
      drawing_spec = landmark_drawing_spec[idx] if isinstance(
          landmark_drawing_spec, Mapping) else landmark_drawing_spec
      # 白色圆圈边框
      circle_border_radius = max(drawing_spec.circle_radius + 1,
                                 int(drawing_spec.circle_radius * 1.2))
      cv2.circle(image, landmark_px, circle_border_radius, color,
                 drawing_spec.thickness)
      # 将颜色填充到圆中
      cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                 color, drawing_spec.thickness)

def det_yolov5v6(info1,info2):
    start_time = time.time()
    listx = []
    list1 = []
    try:
        if args.video == '' and args.images == '':
            raise ValueError('Either --video or --image has to be provided')

        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)

        frame_provider = ImageReader(args.images)
        if args.video != '':
            frame_provider = VideoReader(args.video)
        else:
            args.track = 0

        run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
    except:
        pass


    load_good = 0
    check1 = 0
    check2 = 0
    try:
        check1 = 1
        cap = cv2.VideoCapture(info1)
    except:
        ui.printf('请加载待处理视频')
    try:
        check2 = 1
        cap2 = cv2.VideoCapture(info2)
    except:
        ui.printf('请加载标准视频')
    if check2 and check1:
        load_good = 1
    else:
        ui.printf('请检查标准视频和待处理视频是否全部加载')
    if load_good:

        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        pose2 = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        # 定义全局相似度
        all_Similarity = 0
        # 定义帧数
        count = 0
        second = 0
        count2 = 0
        while True:
            success, image = cap.read()
            success2, image2 = cap2.read()

            if image is None:
                break


            if image2 is None:
                image2 = image2_copy
            else:
                image2_copy = image2.copy()
            count += 1
            count2 += 1
            # 定义单张相似度
            one_Similarity = 0
            # not stander.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image2.flags.writeable = False
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            results2 = pose2.process(image2)
            draw_landmarks(
                image2,
                results2.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                color=[0, 255, 0])
            try:
                x1 = results.pose_landmarks.landmark[0].x
                x2 = results2.pose_landmarks.landmark[0].x
                delta_x = x1 - x2
                y1 = results.pose_landmarks.landmark[0].y
                y2 = results2.pose_landmarks.landmark[0].y
                delta_y = y1 - y2
                for ii in range(len(results.pose_landmarks.landmark)):
                    results2.pose_landmarks.landmark[ii].x = results2.pose_landmarks.landmark[ii].x + delta_x
                    results2.pose_landmarks.landmark[ii].y = results2.pose_landmarks.landmark[ii].y + delta_y
            except:
                pass

            try:
                for ii in range(len(results.pose_landmarks.landmark)):
                    x1 = results.pose_landmarks.landmark[ii].x
                    x2 = results2.pose_landmarks.landmark[ii].x
                    delta_x = abs(x1 - x2)
                    y1 = results.pose_landmarks.landmark[ii].y
                    y2 = results2.pose_landmarks.landmark[ii].y
                    delta_y = abs(y1 - y2)
                    Similarity_x = 1 - delta_x
                    Similarity_y = 1 - delta_y
                    one_Similarity = one_Similarity + (Similarity_x + Similarity_y)/2

            except:
                pass
            one_Similarity = one_Similarity/ 33
            if one_Similarity == 0:
                one_Similarity = 1
            ui.printf(str(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))) + ' 舞蹈相似度为： ' + str(
                one_Similarity)[:5])
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)


            # print(results)
            #print(results.pose_landmarks.landmark[19].x)
            # print(results.pose_landmarks.landmark[19].y)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image2.flags.writeable = True

            # show = np.ones((image.shape[0],image.shape[1],3),dtype=np.uint8)*255


            draw_landmarks(
                image,
                results2.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                color=[0,255,0])

            draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                color=[0, 0, 255])


            ui.showimg2(image2)
            QApplication.processEvents()
            ui.showimg(image)
            QApplication.processEvents()
            all_Similarity += one_Similarity

            if count2 == 25:
                count2 = 0
                second += 1
                listx.append(second)
                list1.append(one_Similarity)

                plt.plot(listx, list1, color="b", linestyle="-", linewidth=1)
                plt.xlabel('Time(s)')
                plt.ylabel('Action similarity')
                plt.title('Real time line chart of action similarity')
                plt.legend()
                plt.pause(0.0001)  # 暂停0.01秒
                plt.ioff()  # 关闭画图的窗口


        cap.release()
        cap2.release()

        ui.printf('整体舞蹈相似度为： ' + str(
            all_Similarity/count)[:5])





class Thread_1(QThread):  # 线程1
    def __init__(self,info1,info2):
        super().__init__()
        self.info1=info1
        self.info2=info2
        self.run2(self.info1,self.info2)

    def run2(self, info1,info2):
        result = []
        result = det_yolov5v6(info1,info2)

# class Thread_1(QThread):
#     def __init__(self, filepath, filepath2):
#         super().__init__()
#         self.filepath = filepath
#         self.filepath2 = filepath2
#
#     def run(self):
#         # Placeholder for video processing logic
#         pass

# class Ui_MainWindow(object):
#     def setupUi(self, MainWindow):
#         MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(1280, 900)
#         MainWindow.setStyleSheet("background-color: #1E1E1E;")
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")
#
#         self.label = QtWidgets.QLabel(self.centralwidget)
#         self.label.setGeometry(QtCore.QRect(168, 60, 900, 71))
#         self.label.setAlignment(QtCore.Qt.AlignCenter)
#         self.label.setObjectName("label")
#         self.label.setStyleSheet("font-size:50px;font-weight:bold;font-family:SimHei;background:rgba(255,255,255,0.6);")
#
#         self.label_2 = QtWidgets.QLabel(self.centralwidget)
#         self.label_2.setGeometry(QtCore.QRect(40, 200, 550, 501))
#         self.label_2.setStyleSheet("background:rgba(255,255,255,0.6);")
#         self.label_2.setAlignment(QtCore.Qt.AlignCenter)
#         self.label_2.setObjectName("label_2")
#
#         self.label_3 = QtWidgets.QLabel(self.centralwidget)
#         self.label_3.setGeometry(QtCore.QRect(620, 200, 550, 501))
#         self.label_3.setStyleSheet("background:rgba(255,255,255,0.6);")
#         self.label_3.setAlignment(QtCore.Qt.AlignCenter)
#         self.label_3.setObjectName("label_3")
#
#         self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
#         self.textBrowser.setGeometry(QtCore.QRect(73, 746, 851, 174))
#         self.textBrowser.setStyleSheet("background:rgba(255,255,255,0.6);")
#         self.textBrowser.setObjectName("textBrowser")
#
#         button_style = """
#         QPushButton {
#             background-color: #4CAF50;
#             border: none;
#             color: white;
#             padding: 15px 32px;
#             text-align: center;
#             text-decoration: none;
#             font-size: 16px;
#             margin: 4px 2px;
#             border-radius: 10px;
#         }
#         QPushButton:hover {
#             background-color: #45a049;
#         }
#         """
#
#         self.pushButton = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton.setGeometry(QtCore.QRect(1020, 750, 150, 40))
#         self.pushButton.setStyleSheet(button_style)
#         self.pushButton.setObjectName("pushButton")
#
#         self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton_2.setGeometry(QtCore.QRect(1020, 800, 150, 40))
#         self.pushButton_2.setStyleSheet(button_style)
#         self.pushButton_2.setObjectName("pushButton_2")
#
#         self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton_3.setGeometry(QtCore.QRect(1020, 850, 150, 40))
#         self.pushButton_3.setStyleSheet(button_style)
#         self.pushButton_3.setObjectName("pushButton_3")
#
#         self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton_4.setGeometry(QtCore.QRect(1020, 900, 150, 40))
#         self.pushButton_4.setStyleSheet(button_style)
#         self.pushButton_4.setObjectName("pushButton_4")
#
#         MainWindow.setCentralWidget(self.centralwidget)
#
#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)
#
#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "舞蹈动作规范度评判系统"))
#         self.label.setText(_translate("MainWindow", "舞蹈动作规范度评判系统"))
#         self.label_2.setText(_translate("MainWindow", "显示标准舞蹈检测结果"))
#         self.label_3.setText(_translate("MainWindow", "显示待检测舞蹈对比结果"))
#         self.pushButton.setText(_translate("MainWindow", "选择待识别视频"))
#         self.pushButton_2.setText(_translate("MainWindow", "选择标准视频"))
#         self.pushButton_3.setText(_translate("MainWindow", "开始评判"))
#         self.pushButton_4.setText(_translate("MainWindow", "退出系统"))
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#         self.filepath = ''
#         self.filepath2 = ''
#         self.thread_1 = None
#
#         # Connect buttons to their respective functions
#         self.ui.pushButton.clicked.connect(self.openfile)
#         self.ui.pushButton_2.clicked.connect(self.openfile2)
#         self.ui.pushButton_3.clicked.connect(self.click_1)
#         self.ui.pushButton_4.clicked.connect(self.handleCalc3)
#
#     def openfile(self):
#         fname = QFileDialog()
#         fname.setAcceptMode(QFileDialog.AcceptOpen)
#         fname, _ = fname.getOpenFileName()
#         if fname == '':
#             return
#         self.filepath = os.path.normpath(fname)
#         self.ui.printf("当前选定的文件路径为：%s" % self.filepath)
#         self.ui.printf('请注意，路径中不包含中文，否则将被卡住')
#
#     def openfile2(self):
#         fname = QFileDialog()
#         fname.setAcceptMode(QFileDialog.AcceptOpen)
#         fname, _ = fname.getOpenFileName()
#         if fname == '':
#             return
#         self.filepath2 = os.path.normpath(fname)
#         self.ui.printf("当前选定的文件路径为：%s" % self.filepath2)
#         self.ui.printf('请注意，路径中不包含中文，否则将被卡住')
#
#     def handleCalc3(self):
#         os._exit(0)
#
#     def click_1(self):
#         try:
#             if self.thread_1:
#                 self.thread_1.quit()
#         except:
#             pass
#         try:
#             self.thread_1 = Thread_1(self.filepath, self.filepath2)
#             self.thread_1.start()
#         except:
#             self.ui.printf('请检查标准视频和待处理视频是否全部加载')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(980, 1280)
        MainWindow.setStyleSheet("background-image: url(\"./template/2.png\")")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        #  设置标题样式
        self.label.setGeometry(QtCore.QRect(60, 60, 900, 71))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("")
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        #  设置标准视频样式
        self.label.setStyleSheet("font-size:50px;font-weight:bold;font-family:SimHei;background:rgba(255,255,255,0.6);")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 175, 600, 301))
        self.label_2.setStyleSheet("background:rgba(255,255,255,0.6);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        #  设置待测视频样式
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(100, 500, 600, 301))
        self.label_3.setStyleSheet("background:rgba(255,255,255,0.6);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        #  设置显示结果区域样式
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(100, 900, 600, 301))
        self.textBrowser.setStyleSheet("background:rgba(255,255,255,0.6);")
        self.textBrowser.setObjectName("textBrowser")
        #  设置上传待测视频按钮
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(750, 700, 150, 40))
        self.pushButton.setStyleSheet("background:rgba(255,255,255,1);border-radius:10px;padding:2px 4px;")
        self.pushButton.setObjectName("pushButton")
        #  设置上传标准视频按钮
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(750, 300, 150, 40))
        self.pushButton_2.setStyleSheet("background:rgba(255,255,255,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")
        #  设置检测视频按钮
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(750, 980, 200, 60))
        self.pushButton_3.setStyleSheet("background:#CDB5CD;border-radius:10px;padding:2px 4px;font-size:25px")
        self.pushButton_3.setObjectName("pushButton_3")
        #  设置退出系统按钮
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(750, 1100, 200, 60))
        self.pushButton_4.setStyleSheet("background:#CDB5CD;border-radius:10px;padding:2px 4px;font-size:25px")
        self.pushButton_4.setObjectName("pushButton_4")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setStyleSheet("""  
            QMainWindow {  
                background-image: url("./template/2.png");  
                background-repeat: no-repeat;  
                background-position: center;  
            }  

            QLabel, QPushButton, QTextBrowser {  
                font-family: 'SimHei';  
            }  

            QLabel {  
                border: 1px solid #CCCCCC;  
                border-radius: 5px;  
            }  

            QTextBrowser {  
                border: 1px solid #CCCCCC;  
                border-radius: 5px;  
                padding: 5px;  
            }  

            QPushButton {  
                border: 2px solid #555;  
                border-radius: 10px;  
                background-color: #EEE;  
                padding: 5px 10px;  
                font-size: 16px;  
                color: #333;  
                transition: background-color 0.3s;  
            }  

            QPushButton:hover {  
                background-color: #DDD;  
            }  

            QPushButton:pressed {  
                background-color: #BBB;  
            }  
        """)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "舞蹈动作规范度评判系统"))
        self.label.setText(_translate("MainWindow", "舞蹈动作规范度评判系统"))
        self.label_2.setText(_translate("MainWindow", "显示标准舞蹈检测结果"))
        self.label_3.setText(_translate("MainWindow", "显示待检测舞蹈对比结果"))
        self.pushButton.setText(_translate("MainWindow", "选择待识别视频"))
        self.pushButton_2.setText(_translate("MainWindow", "选择标准视频"))
        self.pushButton_3.setText(_translate("MainWindow", "开始评判"))
        self.pushButton_4.setText(_translate("MainWindow", "退出系统"))

        # 点击文本框绑定槽事件
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.openfile2)
        self.pushButton_3.clicked.connect(self.click_1)
        self.pushButton_4.clicked.connect(self.handleCalc3)

    def openfile(self):
        global sname,filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("当前选定的文件路径为：%s" % filepath)
        ui.printf('请注意，路径中不包含中文，否则将被卡住')
        self.pushButton.setStyleSheet("""  
            QPushButton {  
                background-color: #4CAF50;  
                border-color: #45a049;  
                color: white;  
            }  

            QPushButton:hover {  
                background-color: #45a049;  
            }  

            QPushButton:pressed {  
                background-color: #388e3c;  
            }  
        """)


    def openfile2(self):
        global sname,filepath2
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath2 = os.path.normpath(fname)
        sname = filepath2.split(os.sep)
        ui.printf("当前选定的文件路径为：%s" % filepath2)
        ui.printf('请注意，路径中不包含中文，否则将被卡住')
        self.pushButton.setStyleSheet("""  
            QPushButton {  
                background-color: #4CAF50;  
                border-color: #45a049;  
                color: white;  
            }  

            QPushButton:hover {  
                background-color: #45a049;  
            }  

            QPushButton:pressed {  
                background-color: #388e3c;  
            }  
        """)

    def handleCalc3(self):
        os._exit(0)

    def printf(self,text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 500
        else:
            ratio = n_height / 500
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QPixmap.fromImage(new_img))

    def showimg2(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 500
        else:
            ratio = n_height / 500
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_2.setPixmap(QPixmap.fromImage(new_img))


    def click_1(self):
        global filepath,filepath2
        try:
            self.thread_1.quit()
        except:
            pass
        try:
            self.thread_1 = Thread_1(filepath, filepath2)  # 创建线程
            self.thread_1.wait()
            self.thread_1.start()  # 开始线程
        except:
            ui.printf('请检查标准视频和待处理视频是否全部加载')

class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('欢迎登录')
        self.resize(600, 500)
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet("background-image: url('./template/3.png');")

        self.frame = QFrame(self)
        self.frame.setStyleSheet("background:rgba(255,255,255,0);")
        self.frame.move((self.width() - 400) // 2, 150)
        self.mainLayout = QVBoxLayout(self.frame)

        # 设置字体样式
        font = QFont()
        font.setPointSize(12)  # 设置字体大小
        font.setBold(True)  # 设置字体加粗
        # 设置账号的输入框
        self.nameEd1 = QLineEdit(self.frame)
        self.nameEd1.setFont(font)
        self.nameEd1.setFixedSize(400, 105)
        self.nameEd1.setPlaceholderText("账号")
        op1 = QGraphicsOpacityEffect()
        op1.setOpacity(0.5)
        self.nameEd1.setGraphicsEffect(op1)
        self.nameEd1.setStyleSheet('''QLineEdit{border-radius:5px;}''')
        # 设置密码的输入框
        self.nameEd3 = QLineEdit(self.frame)
        self.nameEd3.setFont(font)
        self.nameEd3.setPlaceholderText("密码")
        op5 = QGraphicsOpacityEffect()
        op5.setOpacity(0.5)
        self.nameEd3.setGraphicsEffect(op5)
        self.nameEd3.setStyleSheet('''QLineEdit{border-radius:5px;}''')
        # 设置登录按钮
        self.btnOK = QPushButton('登录', self.frame)
        self.btnOK.setFont(font)
        self.btnOK.setFixedSize(400, 60)
        op3 = QGraphicsOpacityEffect()
        op3.setOpacity(1)
        self.btnOK.setGraphicsEffect(op3)
        self.btnOK.setStyleSheet(
            '''QPushButton{background-color:#DDA0DD;border-radius:5px;color:#FFFFFF;}    
            QPushButton:hover{background-color:#FFB6C1;}'''
        )

        self.btnOK.clicked.connect(self.button_enter_verify)

        self.mainLayout.addWidget(self.nameEd1)
        self.mainLayout.addWidget(self.nameEd3)
        self.mainLayout.addWidget(self.btnOK)
        self.mainLayout.setSpacing(20)

    def button_register_verify(self):
        global path1
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        user = self.nameEd1.text()
        pas = self.nameEd3.text()
        with open(path1 + '/' + user + '.txt', "w") as f:
            f.write(pas)
        self.nameEd1.setText("注册成功")


    def button_enter_verify(self):
        # 校验账号是否正确
        global administrator, userstext, passtext
        userstext = []
        passtext = []
        administrator = 0
        pw = 0
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        users = os.listdir(path1)

        for i in users:
            with open(path1 + '/' + i, "r") as f:
                userstext.append(i[:-4])
                passtext.append(f.readline())

        for i in users:
            if i[:-4] == self.nameEd1.text():
                with open(path1 + '/' + i, "r") as f:
                    if f.readline() == self.nameEd3.text():
                        if i[:2] == 'GM':
                            administrator = 1
                            self.accept()
                        else:
                            passtext.append(f.readline())
                            self.accept()
                    else:
                        self.nameEd3.setText("密码错误")
                        pw = 1
        if pw == 0:
            self.nameEd1.setText("账号错误")

if __name__ == "__main__":
    # 创建应用
    window_application = QApplication(sys.argv)
    # 设置登录窗口
    login_ui = LoginDialog()
    # 校验是否验证通过
    if login_ui.exec_() == QDialog.Accepted:
        # 初始化主功能窗口
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        if administrator == 1:
            ui.printf('欢迎管理员')
            for i in range(0, len(userstext)):
                ui.printf('账户' + str(i) + ':' + str(userstext[i]))
                ui.printf('密码' + str(i) + ':' + str(passtext[i]))
        else:
            ui.printf('欢迎用户')
        # 设置应用退出
        sys.exit(window_application.exec_())
