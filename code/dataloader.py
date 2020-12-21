import os
import random
import glob
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

random.seed(100)


def populate_chunk_list(images_path):
    image_list = []
    file_list = os.listdir(images_path)
    for f in file_list:
        if os.path.isdir(images_path + f):
            image_list.append(glob.glob(images_path + f + "/image/*.jpg"))
    return image_list

def single_chunk(images_path):
    image_list = glob.glob(images_path + "*.jpg")
    return [image_list]

class sequential_loader(data.Dataset):

    def __init__(self, images_path, seq_num, data_type, img_size=256):
        if data_type == 'train':
            self.chunk_list = populate_chunk_list(images_path)
        else:
            self.chunk_list = single_chunk(images_path)

        if type(img_size) == list:
            self.size = img_size
        elif type(img_size) == int:
            self.size = (img_size, img_size)
        else:
            raise TypeError("Img size unsupported!")
        self.seq_num = seq_num

        self.seq_list = []
        for chunk in self.chunk_list:
            chunk.sort()
            for i in range(0, len(chunk)-self.seq_num, self.seq_num//2 if data_type == 'train' else self.seq_num):
                self.seq_list.append(chunk[i:i+self.seq_num])
        if data_type == 'train':
            random.shuffle(self.seq_list)

        print("Total %s chunks: %d" % (data_type, len(self.chunk_list)))
        print("Total %s sequences: %d" % (data_type, len(self.seq_list)))

    def __getitem__(self, index):
        data_path = self.seq_list[index]

        imgs = []
        for img in data_path:
            data = Image.open(img)
            data = data.resize(self.size, Image.ANTIALIAS)
            data = (np.asarray(data) / 255.0)
            imgs.append(torch.from_numpy(data).float())

        imgs = torch.stack(imgs)
        return imgs.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.seq_list)
