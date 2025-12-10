import os
import numpy as np
import cv2
import random
import csv
from PIL import Image
from tqdm import tqdm

def add_gaussian_noise(image, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise) 
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.005, pepper_prob=0.005):
    noisy_image = image.copy()
    total_pixels = image.size


    num_salt = int(total_pixels * salt_prob)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

def add_poisson_noise(image):

    noisy_image = np.random.poisson(image.astype(np.float32) * 1.0) 
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def preprocess(dir, output_dir):
    store_train_noise_type = []
    store_valid_noise_type = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ['train', 'valid']:
        input_folder = os.path.join(dir, split)
        output_folder = os.path.join(output_dir, split)

        if not os.path.exists(input_folder):
            print(f"Warning: {input_folder} does not exist, skipping...")
            continue
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in tqdm(os.listdir(input_folder)):
            if filename.endswith(".png"):
                img_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)


                image = cv2.imread(img_path, cv2.IMREAD_COLOR)


                noise_type = random.choice(["gaussian", "salt_and_pepper", "poisson"])

                if noise_type == "gaussian":
                    noisy_image = add_gaussian_noise(image)
                elif noise_type == "salt_and_pepper":
                    noisy_image = add_salt_and_pepper_noise(image)
                elif noise_type == "poisson":
                    noisy_image = add_poisson_noise(image)

                cv2.imwrite(output_path, noisy_image)

                print(f"Processed {filename} with {noise_type} noise.")
                if split == 'train':
                    store_train_noise_type.append([filename, noise_type])
                else:
                    store_valid_noise_type.append([filename, noise_type])

    with open(os.path.join(output_dir, "train_noise_type_new.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(store_train_noise_type)
    with open(os.path.join(output_dir, "valid_noise_type_new.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(store_valid_noise_type)



if __name__ == "__main__":
    preprocess("dataset", "noise_dataset_new")
