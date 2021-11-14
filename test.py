import torch
import cv2
from torch.autograd import Variable
from train import generator


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out



G = generator()
z_dimension = 100
img=cv2.imread('real_images.png')
num_img = img.size(0)
print('here')
G.load_state_dict(torch.load('generator.pth'))

z = Variable(torch.randn(num_img, z_dimension)).cuda()

new_img=G(z)

fake_images = to_img(new_img.cpu().data)

cv2.imshow('',fake_images)