import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 12

    # ---- 原始低光图 ----
    data_lowlight_pil = Image.open(image_path).convert('RGB')
    data_lowlight_np = np.asarray(data_lowlight_pil) / 255.0  # [H,W,3], float

    data_lowlight = torch.from_numpy(data_lowlight_np).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor

    # 同时裁剪 numpy
    data_lowlight_np = data_lowlight_np[0:h, 0:w, :]

    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1).cuda().unsqueeze(0)

    # ---- 加载模型 ----
    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))

    # ---- 推理 ----
    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)
    end_time = time.time() - start

    # ---- 转 numpy ----
    enhanced_np = enhanced_image.detach().cpu().numpy()[0]  # [3,H,W]
    enhanced_np = np.transpose(enhanced_np, (1, 2, 0))      # [H,W,3]

    # ---- 计算 PSNR ----
    psnr_val = psnr(data_lowlight_np, enhanced_np, data_range=1.0)
    print("Time: {:.4f}s, PSNR: {:.4f}".format(end_time, psnr_val))

    # ---- 保存增强图 ----
    image_output_path = image_path.replace('test_data', 'result_Zero_DCE++')
    save_dir = image_output_path.replace('/' + image_output_path.split("/")[-1], '')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torchvision.utils.save_image(enhanced_image, image_output_path)

    return end_time, psnr_val


if __name__ == '__main__':

    with torch.no_grad():
        filePath = 'data/test_data/'
        file_list = os.listdir(filePath)

        sum_time = 0
        sum_psnr = 0
        img_count = 0

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                print(image)
                t, p = lowlight(image)
                sum_time += t
                sum_psnr += p
                img_count += 1

        print("Total time:", sum_time)
        if img_count > 0:
            print("Average PSNR:", sum_psnr / img_count)
