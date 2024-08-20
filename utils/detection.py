from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import cv2
from utils.common import timeit

class Detector:
    def __init__(self, model_path):
        # model_path = "../models/yolov8n.pt"  # 测试使用
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.show_config = {
            'font_scale': 0.5,
            'font_thickness': 2,
            'font_color': (0, 139, 9),
            'poly_line_color': (50, 100, 255),
            'poly_line_width': 5,
            'box_width': 2
        }

    @timeit
    def detect(self, raw_frame, depth_frame):
        results = self.model.track(raw_frame, persist=True, conf=0.5, iou=0.5)
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        frame = results[0].plot(line_width=self.show_config['box_width'])

        annotated_frame = frame

        # Plot the tracks
        if results[0].boxes.id is not None:

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                # uu = x + w / 2
                # vv = y + h / 2
                # print(depth_frame)
                z = depth_frame[int(y), int(x)]
                # z = 100
                X, Y, Z = x * z / 1000, y * z / 1000, z/10
                annotated_frame = self.display_xyz_on_frame(annotated_frame, x, y+10, X, Y, Z)

                track = self.track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False,
                              color=self.show_config['poly_line_color'], thickness=self.show_config['poly_line_width'])

        return annotated_frame

    def display_xyz_on_frame(self, frame, u, v, x, y, z):
        # 创建一个副本以免修改原始图像
        frame_with_xyz = frame.copy()

        # 在图像上标出目标中心的位置
        center_point = (int(u), int(v))
        # cv2.circle(frame_with_xyz, center_point, 5, (0, 0, 255), -1)  # 用红色圆圈标记中心点

        # 在目标上方绘制 `(x, y, z)` 信息
        print(x)
        text = f"x: {x:.1f}, y: {y:.1f}, z: {z:.1f}"
        cv2.putText(frame_with_xyz, text, (int(u - 50), int(v - 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    self.show_config['font_scale'],
                    self.show_config['font_color'], self.show_config['font_thickness'])

        return frame_with_xyz


if __name__ == '__main__':
    detector = Detector('../models/yolov8n.pt')
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cam.read()
        if ret:
            frame_d = detector(frame, 1)
            cv2.imshow('cs', frame_d)
            cv2.waitKey(1)
