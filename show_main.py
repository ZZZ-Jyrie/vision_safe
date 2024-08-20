import argparse
import cv2
import keyboard
import matplotlib
import numpy as np
import torch

from utils.depth_anything_v2.dpt import DepthAnythingV2
from utils.lanedetection import LaneDetection
from utils.segmentation import Segmenter
from utils.detection import Detector
from utils.gazecapture import Gazecapture
from utils.common import timeit
from utils.face_detect import FaseDetector


@timeit
def get_color_depth(raw_frame):
    # 获取原始帧的尺寸
    original_size = (raw_frame.shape[1], raw_frame.shape[0])  # (width, height)

    # 调整输入帧的大小为640x480
    resized_frame = cv2.resize(raw_frame, (640, 480))

    # 使用调整后的帧进行深度推断
    depth = depth_anything.infer_image(resized_frame, 518)

    # 归一化深度图像到0-255范围
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    # 将深度图像转换为伪彩色图像
    colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

    # 将彩色深度图像和深度图像调整回原始尺寸
    colored_depth = cv2.resize(colored_depth, original_size)
    depth = cv2.resize(depth, original_size)

    return colored_depth, depth


def show_frame(win_name, frame, w, h, x, y):
    '''
    显示窗口
    w: 宽度
    h: 高度
    x: 窗口左上角的x坐标
    y: 窗口左上角的y坐标
    '''
    global win_num, win_num_now

    if win_num_now <= win_num:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, w, h)
        cv2.moveWindow(win_name, x, y)
        win_num_now += 1
    cv2.imshow(win_name, frame)
    cv2.waitKey(1)


if __name__ == '__main__':

    win_num = 5
    win_num_now = 0

    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])

    args = parser.parse_args()

    DEVICE = 'cuda'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # 深度估计模块初始化
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(
        torch.load(f'models/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # 眼球追踪模块初始化
    gazecapture = Gazecapture()

    # 车道线检测初始化
    lanedetection = LaneDetection()

    # 实例分割模块初始化
    segmenter = Segmenter(model_path=r'./models/yolov8x-seg.pt')

    # 目标检测+定位模块初始化
    detector = Detector(model_path=r'./models/yolov8n.pt')

    # 面部检测模块初始化
    fasedetector = FaseDetector()

    # 相机模块初始化
    raw_cap_front = cv2.VideoCapture('onroad.mp4')  # 打开视频:前视
    raw_cap_rear = cv2.VideoCapture(0)  # 打开视频:后视

    width = 1920  # 宽度
    height = 1080  # 高度
    raw_cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    raw_cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # raw_cap_rear.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # raw_cap_rear.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    # roi
    ret, frame = raw_cap_front.read()
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = roi


    # 设置帧率
    fps = 30  # 每秒帧数
    raw_cap_front.set(cv2.CAP_PROP_FPS, fps)
    raw_cap_rear.set(cv2.CAP_PROP_FPS, fps)

    # raw_video = cv2.VideoCapture(0)  # 打开视频

    if not raw_cap_front.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        # 读取相机画面
        ret_f, raw_frame_front = raw_cap_front.read()  # 使用read()方法获取最新的帧
        raw_frame_front = raw_frame_front[y:y + h, x:x + w]
        raw_frame_front = cv2.resize(raw_frame_front, (1920, 1080))

        ret_r, raw_frame_rear = raw_cap_rear.read()  # 使用read()方法获取最新的帧
        # raw_frame_front = cv2.flip(raw_frame_front, 0)

        print(raw_frame_front.shape)

        # if not ret_f :
        #     print("无法读取摄像头画面")
        #     break

        # 深度估计计算及显示（窗口2）
        depth_frame_color, depth_frame_int8 = get_color_depth(raw_frame_front)
        show_frame('win2: depth', depth_frame_color, 480, 340, 960, 0)
        # print(depth_frame_int8.type)

        # 主视图处理及显示（窗口1）
        detect_frame = detector.detect(raw_frame_front, depth_frame_int8)
        frame_rear, frame_front = gazecapture.gaze_esti(raw_frame_rear, detect_frame)
        show_frame('win1: det', frame_front, 960, 680, 0, 0)
        show_frame('win5: gaze', frame_rear, 480, 340, 480, 680)

        # 车道线检测显示（窗口3）
        road_frame = lanedetection.land_detect(raw_frame_front)
        show_frame('win3: road', road_frame, 480, 340, 960, 340)

        # 实例分割显示（窗口4）
        seg_frame = segmenter.seg(raw_frame_front)
        show_frame('win4: seg', seg_frame, 480, 340, 0, 680)

        # 实例分割显示（窗口6）
        face_frame = fasedetector.detect_face(raw_frame_rear)
        show_frame('win6: face', face_frame, 480, 340, 960, 680)



        if keyboard.is_pressed('q'):
            print('用户结束操作')
            break

    raw_cap_front.release()
    cv2.destroyAllWindows()
