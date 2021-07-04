# -*- coding: utf-8 -*-
"""
 Created on 2021/7/3 10:22
 Filename   : pytorch_common_code.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012/PythonSamples
 Description: Pytorch 常用命令
"""

# =======================================================
import torch
import torchvision
import PIL
import numpy as np


def common_code():
    """ Pytorch 常用命令 """
    """ cuda """
    print(f'cuda is available: {torch.cuda.is_available()}')

    """ 张量基本信息 """
    tensor = torch.randn(3, 4, 3)
    print('torch.randn(3, 4, 5)')
    print(f'数据类型：{tensor.type()}')
    print(f'张量大小：{tensor.size()}')
    print(f'维度数量：{tensor.dim()}')

    """ 张量命名 """
    imgs = torch.randn(1, 2, 2, 3, names=('N', 'C', 'H', 'W'))
    print(f"imgs.sum('C'): {imgs.sum('C')}")
    print(f"imgs.select('C', index=0): {imgs.select('C', index=0)}")

    """ torch.tensor 与 np.ndarray 转换 """
    ndarray = tensor.cpu().numpy()
    tensor = torch.from_numpy(ndarray).float()

    """ torch.tensor 与 PIL.Image 转换 """
    image = torchvision.transforms.functional.to_pil_image(tensor)
    tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open('images/world.jpg'))

    """ np.ndarray 与 PIL.Image 转换 """
    """ 注意第三个维度 """
    image = PIL.Image.fromarray(ndarray.astype(np.uint8))
    ndarray = np.asarray(PIL.Image.open('images/world.jpg'))

    """ 张量拼接 """
    tensor1 = torch.randn(1, 2)
    tensor2 = torch.randn(1, 2)
    """ 指定维度拼接 """
    tensor_cat = torch.cat([tensor1, tensor2], dim=0)
    """ 新增维度 """
    tensor_stack = torch.stack([tensor1, tensor2], dim=0)
    print(tensor_cat)
    print(tensor_stack)


if __name__ == '__main__':
    common_code()
