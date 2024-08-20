import cv2
import torch
from utils import models_gaze
import torchvision.transforms as transforms
import numpy as np
from utils.common import timeit


class Gazecapture:
    def __init__(self):
        # cap = cv2.VideoCapture(0)

        loc = "cuda:0"
        self.device = torch.device(loc if torch.cuda.is_available() else "cpu")

        pretrain = torch.load('./models/xgaze/epoch=39.ckpt', map_location=loc)
        statedict = pretrain if "state_dict" not in pretrain else pretrain["state_dict"]

        self.net = models_gaze.GazeRes('res18')
        self.net.to(self.device)
        self.net.load_state_dict(statedict, strict=False)
        self.transform = self.get_transform()
        self.net.eval()

    def get_transform(self, grayscale=False, convert=True, crop=False):
        transform_list = []
        transform_list += [transforms.ToPILImage()]
        print(grayscale)
        if grayscale:
            transform_list.append(transforms.Grayscale(1))
        if crop:
            transform_list += [transforms.CenterCrop(192)]
            transform_list += [transforms.Resize(224)]
        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        return transforms.Compose(transform_list)

    @timeit
    def gaze_esti(self, frame1, frame2):
        '''

        Args:
            frame1: 后（人脸）
            frame2: 前

        Returns:

        '''
        # Preprocess the frame from camera 1
        ycrcb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        fimg = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        fimg = self.transform(fimg)
        expanded_tensor = fimg.unsqueeze(0)
        final_tensor = torch.cat([expanded_tensor] * 20, dim=0)

        frame_cuda = final_tensor.to(self.device)
        img = {"face": frame_cuda}

        # Get gaze vector from camera 1
        gazes, _ = self.net(img)
        pred_gaze = gazes[0]
        pred_gaze_np = pred_gaze.cpu().data.numpy()

        # Draw gaze direction in camera 1 view
        frame1 = self.draw_gaze(frame1, pred_gaze_np)

        # Calculate gaze point in camera 2 view
        gaze_point_camera2 = self.calculate_gaze_point(pred_gaze_np, frame2.shape)

        # Draw the gaze point on the frame from camera 2
        cv2.circle(frame2, gaze_point_camera2, 25, (0, 255, 128), -1)

        return frame1, frame2

    def draw_gaze(self, image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
        image_out = image_in.copy()
        (h, w) = image_in.shape[:2]
        length = np.min([h, w]) / 2.0
        pos = (int(w / 2.0), int(h / 2.0))
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
        dy = -length * np.sin(pitchyaw[0])
        cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                        tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                        thickness, cv2.LINE_AA, tipLength=0.2)
        return image_out

    def calculate_gaze_point(self, gaze_vector, frame_shape):
        h, w = frame_shape[:2]
        h -= 80
        w -= 80

        focal_length = w  # Assuming focal length is the width of the image

        # Calculate the intersection point on the image plane
        x = int(focal_length * np.tan(gaze_vector[1] * 3) + w / 2)
        y = int(-focal_length * np.tan(gaze_vector[0] * 8) + h / 2)  # Inverted y-coordinate

        # Ensure the coordinates are within the image bounds
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        return x, y


if __name__ == '__main__':
    # 初始化 Gazecapture 实例
    gaze_capture = Gazecapture()
    cap_R = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_F = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        ret1, frame1 = cap_R.read()
        ret2, frame2 = cap_F.read()

        if ret1 and ret2:
            processed_frame1, processed_frame2 = gaze_capture(frame1, frame2)
            cv2.imshow('Camera 1: Gaze Direction', processed_frame1)
            cv2.imshow('Camera 2: Gaze Point', processed_frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap_R.release()
    cap_F.release()

    cv2.destroyAllWindows()
