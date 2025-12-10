import os
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import csv
from matplotlib import pyplot as plt

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
    csv_dir= os.path.join(noisy_dir, "train_noise_type_new.csv")
    with open(csv_dir, "r") as f:
        reader = csv.reader(f)
        store_train_noise_type = list(reader)
    store_train_noise_type = dict(store_train_noise_type)
    # select 5 examples for 3 types of noise
    noise_dict_cnt = {"gaussian": 0, "salt_and_pepper": 0, "poisson": 0}
    if not os.path.exists("denoise_example"):
        os.mkdir("denoise_example")
    for original_img, noisy_img, filename in image_pairs_generator(original_dir, noisy_dir):
        noise_type = store_train_noise_type[filename]
        if noise_dict_cnt[noise_type] == 5:
            continue
        if noise_dict_cnt["gaussian"] == 5 and noise_dict_cnt["salt_and_pepper"] == 5 and noise_dict_cnt["poisson"] == 5:
            break

        denoised_img = denoise_tv_chambolle(noisy_img, weight=0.1)
        denoised_img = (denoised_img * 255).astype(np.uint8)

        if noise_dict_cnt[noise_type] < 5:
            noise_dict_cnt[noise_type] += 1
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
            plt.title("Noisy Image")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB))
            plt.title("Denoised Image")
            plt.axis("off")
            plt.suptitle(f"Noise Type: {noise_type}")
            plt.savefig(os.path.join("denoise_example", f"{noise_type}_{noise_dict_cnt[noise_type]}.png"), dpi=300)

    

if __name__ == "__main__":
    main("dataset", "noise_dataset_new") 