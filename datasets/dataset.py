import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataSet(Dataset):
    def __init__(self, frame_idx=None, vid_dir=None, transform=None, img_size=None, train=True):
        self.vid_dir = vid_dir
        self.transform = transform
        self.img_size = img_size
        self.frame_path = []

        # get image path
        if vid_dir:
            all_imgs = os.listdir(vid_dir)
            all_imgs.sort()

            for img_id in all_imgs:
                self.frame_path.append(img_id)

        # get frame indedx: the id for first frame is 0 and the id for last is 1
        if frame_idx:
            self.frame_idx = [
                (x - min(frame_idx))/(max(frame_idx) - min(frame_idx)) for x in frame_idx]
        else:
            num_frame = len(self.frame_path)
            self.frame_idx = [float(x) / (num_frame - 1)
                              for x in range(num_frame)]

        # length judge
        if vid_dir:
            assert len(self.frame_path) == len(
                self.frame_idx), 'The number of frames should equal to that of frame indexes'

    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, idx):

        # get frame idx
        frame_idx = torch.tensor(self.frame_idx[idx])

        # get gt image
        if self.frame_path:
            valid_idx = int(idx)
            img_id = self.frame_path[valid_idx]
            img_name = os.path.join(self.vid_dir, img_id)
            image = Image.open(img_name).convert("RGB")

            if self.img_size and image.size != self.img_size:
                image = image.resize(self.img_size)

            tensor_image = self.transform(image)
            if tensor_image.size(1) > tensor_image.size(2):
                tensor_image = tensor_image.permute(0, 2, 1)
            data_dict = {
                "img_id": frame_idx,
                "img_gt": tensor_image,
            }
        else:
            data_dict = {
                "img_id": frame_idx
            }

        return data_dict
