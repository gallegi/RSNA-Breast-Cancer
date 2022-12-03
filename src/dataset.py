import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2

class BCDataset(Dataset):
    def __init__(self, df, im_dir, transforms):
        self.df = df.reset_index(drop=True)
        self.im_dir = im_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        im_name = str(row['patient_id']) + '_' + str(row['image_id']) + '.png'
        
        im_path = os.path.join(self.im_dir, im_name)

        im = cv2.imread(im_path)[:,:,::-1]
        im_ts = self.transforms(image=im)['image']

        if 'cancer' in self.df.columns:
            label = torch.tensor(row['cancer']).float()
        else:
            label = -1

        return im_ts, label
