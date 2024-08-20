import cv2
import torch
import models
import torchvision.transforms as transforms
import numpy as np

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
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

def calculate_gaze_point(gaze_vector, frame_shape):
    h, w = frame_shape[:2]
    focal_length = w  # Assuming focal length is the width of the image

    # Calculate the intersection point on the image plane
    x = int(focal_length * np.tan(gaze_vector[1]) + w / 2)
    y = int(-focal_length * np.tan(gaze_vector[0]) + h / 2)  # Inverted y-coordinate

    # Ensure the coordinates are within the image bounds
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    return x, y

def get_transform(grayscale=False, convert=True, crop=False):
    transform_list = [transforms.ToPILImage()]
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

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

loc = "cuda:0"
device = torch.device(loc if torch.cuda.is_available() else "cpu")

pretrain = torch.load('checkpoints/xgaze/epoch=39.ckpt', map_location=loc)
statedict = pretrain if "state_dict" not in pretrain else pretrain["state_dict"]

net = models.GazeRes('res18')
net.to(device)
net.load_state_dict(statedict, strict=False)
transform = get_transform()
net.eval()

with torch.no_grad():
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            # Preprocess the frame from camera 1
            ycrcb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            fimg = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

            fimg = transform(fimg)
            expanded_tensor = fimg.unsqueeze(0)
            final_tensor = torch.cat([expanded_tensor] * 20, dim=0)

            frame_cuda = final_tensor.to(device)
            img = {"face": frame_cuda}

            # Get gaze vector from camera 1
            gazes, _ = net(img)
            pred_gaze = gazes[0]
            pred_gaze_np = pred_gaze.cpu().data.numpy()

            # Draw gaze direction in camera 1 view
            frame_with_gaze1 = draw_gaze(frame1, pred_gaze_np)

            # Calculate gaze point in camera 2 view
            gaze_point_camera2 = calculate_gaze_point(pred_gaze_np, frame2.shape)

            # Draw the gaze point on the frame from camera 2
            cv2.circle(frame2, gaze_point_camera2, 10, (0, 255, 0), -1)

            # Display both frames
            cv2.imshow('Camera 1: Gaze Direction', frame_with_gaze1)
            cv2.imshow('Camera 2: Gaze Point', frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
