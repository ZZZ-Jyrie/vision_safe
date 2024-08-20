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
def get_transform(grayscale=False, convert=True, crop = False):
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


cap = cv2.VideoCapture(0)

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
        ret, frame = cap.read()
        if ret:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            fimg = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

            fimg = transform(fimg)
            # frame_tensor = torch.tensor(fimg)

            # 在第0维度上添加一个维度，变为[1, 3, 224, 224]
            expanded_tensor = fimg.unsqueeze(0)

            # 使用torch.cat()方法将张量在第0维度上复制20份，得到形状为[20, 3, 224, 224]的张量
            final_tensor = torch.cat([expanded_tensor] * 20, dim=0)


            frame_cuda = final_tensor.to(device)

            img = {"face": frame_cuda}

            gazes, _ = net(img)

            pred_gaze = gazes[0]
            pred_gaze_np = pred_gaze.cpu().data.numpy()

            frame_with_gaze = draw_gaze(frame, pred_gaze_np)
            print(pred_gaze_np)

            cv2.imshow('Gaze Direction', frame_with_gaze)
            cv2.waitKey(1)