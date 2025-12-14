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
from skimage.metrics import peak_signal_noise_ratio as psnr

def lowlight(image_path, output_dir, model_path='snapshots_Zero_DCE++/Epoch99.pth'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 12

    data_lowlight_pil = Image.open(image_path).convert('RGB')
    data_lowlight_np = np.asarray(data_lowlight_pil) / 255.0  # [H,W,3], float

    data_lowlight = torch.from_numpy(data_lowlight_np).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor

    data_lowlight_np = data_lowlight_np[0:h, 0:w, :]

    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1).cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load(model_path))

    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)
    end_time = time.time() - start

    enhanced_np = enhanced_image.detach().cpu().numpy()[0]  # [3,H,W]
    enhanced_np = np.transpose(enhanced_np, (1, 2, 0))      # [H,W,3]

    psnr_val = psnr(data_lowlight_np, enhanced_np, data_range=1.0)
    print("Time: {:.4f}s, PSNR: {:.4f}".format(end_time, psnr_val))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_name = os.path.basename(image_path)
    image_output_path = os.path.join(output_dir, image_name)

    torchvision.utils.save_image(enhanced_image, image_output_path)

    return end_time, psnr_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-DCE++ Low Light Image Enhancement')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='input directory')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='output directory')
    parser.add_argument('--model_path', '-m', type=str, default='snapshots_Zero_DCE++/Epoch99.pth',
                        help='model path (default: snapshots_Zero_DCE++/Epoch99.pth)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    with torch.no_grad():
        input_dir = args.input_dir
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        test_list = []
        for ext in image_extensions:
            test_list.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if len(test_list) == 0:
            print(f"Warning: No image files found in directory {input_dir}")
            sys.exit(1)
        
        print(f"Found {len(test_list)} images")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {args.output_dir}")
        print("-" * 50)

        sum_time = 0
        sum_psnr = 0
        img_count = 0

        for image in test_list:
            print(f"Processing: {image}")
            t, p = lowlight(image, args.output_dir, args.model_path)
            sum_time += t
            sum_psnr += p
            img_count += 1

        print("-" * 50)
        print(f"Total processed: {img_count} images")
        print(f"Total time: {sum_time:.4f}s")
        if img_count > 0:
            print(f"Average time: {sum_time/img_count:.4f}s/image")
            print(f"Average PSNR: {sum_psnr / img_count:.4f}")
