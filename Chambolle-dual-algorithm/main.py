import os
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import logging
from tqdm import tqdm

def image_pairs_generator(original_dir, noisy_dir):
    for split in ['train', 'valid']:
        original_folder = os.path.join(original_dir, split)
        noisy_folder = os.path.join(noisy_dir, split)

        if not os.path.exists(original_folder) or not os.path.exists(noisy_folder):
            print(f"Warning: {split} folder missing in original/noisy directories.")
            continue

        for filename in os.listdir(noisy_folder):
            original_path = os.path.join(original_folder, filename)
            noisy_path = os.path.join(noisy_folder, filename)

            if not os.path.exists(original_path):
                print(f"Warning: {original_path} does not exist, skipping...")
                continue

            original_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
            noisy_img = cv2.imread(noisy_path, cv2.IMREAD_COLOR)

            if original_img is None or noisy_img is None:
                print(f"Error: Failed to read {filename}, skipping...")
                continue

            yield original_img, noisy_img, filename

def compute_snr(original, denoised):
    noise = original - denoised
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)

    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

def main(original_dir, noisy_dir):
    logging.basicConfig(filename="log_new.txt", level=logging.INFO)
    for original_img, noisy_img, filename in tqdm(image_pairs_generator(original_dir, noisy_dir), total=800):
        # skimage 的去噪处理（总变分去噪）
        denoised_img = denoise_tv_chambolle(noisy_img, weight=0.1)
        denoised_img = (denoised_img * 255).astype(np.uint8)  # 还原 uint8 格式

        # 计算 SNR, PSNR, SSIM
        snr_before = compute_snr(original_img, noisy_img) 
        snr_after = compute_snr(original_img, denoised_img)
        psnr_before = psnr(original_img, noisy_img)
        psnr_after = psnr(original_img, denoised_img)
        logging.info(f"Image: {filename}")
        logging.info(f"  SNR Before: {snr_before:.2f}, After: {snr_after:.2f}")
        logging.info(f"  PSNR Before: {psnr_before:.2f}, After: {psnr_after:.2f}")
        logging.info("-" * 50)
if __name__ == "__main__":
    main("dataset", "noise_dataset_new") 