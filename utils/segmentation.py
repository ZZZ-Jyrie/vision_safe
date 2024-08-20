from ultralytics import YOLO
import numpy as np
import cv2
from ultralytics.utils.plotting import colors
from utils.common import timeit


class Segmenter:
    def __init__(self, model_path):
        '''
        Args:
            model_path: 实例分割模型路径
        '''
        self.model = YOLO(model_path)

    @timeit
    def seg(self, raw_frame):
        # 使用模型进行实例分割
        results = self.model(raw_frame)
        frame = results[0].plot()

        return frame



if __name__ == '__main__':
    segmenter = Segmenter('yolov8s-seg.pt')  # 替换为你的模型路径
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if ret:
            frame_d = segmenter.seg(frame)
            cv2.imshow('Segmented Frame', frame_d)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()
