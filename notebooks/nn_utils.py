import cv2
import os

import torch
from torch import nn

# Basic blocks and utilities
def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        else:
            raise NotImplementedError(f'Undefined type: {t}')
    return nn.Sequential(*L)


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# DnCNN Model
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Activation function should be either ReLU or LeakyReLU'
        bias = True

        m_head = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x-n
        

def preprocess_and_save_images(input_directory, output_directory, chunk_size=(224, 224)):
    """
    First need to download data from:
    https://www.kaggle.com/datasets/sovitrath/uav-small-object-detection-dataset?resource=download
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    images = [img for img in os.listdir(input_directory) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    chunk_width, chunk_height = chunk_size
    
    for img_name in images:
        img_path = os.path.join(input_directory, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        num_chunks_width = width // chunk_width
        num_chunks_height = height // chunk_height

        for i in range(num_chunks_height):
            for j in range(num_chunks_width):
                cropped_img = image[i*chunk_height:(i+1)*chunk_height, j*chunk_width:(j+1)*chunk_width]
                cropped_img_name = f"{img_name.split('.')[0]}_chunk_{i}_{j}.jpg"
                cv2.imwrite(os.path.join(output_directory, cropped_img_name), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
