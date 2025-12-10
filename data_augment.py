import cv2
import numpy as np
import random
import os
import argparse


def random_darken(image, min_factor=0.2, max_factor=0.9):
    """
    Random darkening: multiply the image by a random factor
    within [min_factor, max_factor].
    """
    factor = random.uniform(min_factor, max_factor)
    darkened = image.astype(np.float32) * factor
    darkened = np.clip(darkened, 0, 255).astype(np.uint8)
    return darkened


def motion_blur(image, max_kernel_size=25):
    """
    Random-direction motion blur:
    - Random blur length
    - Random blur angle in [0, 180) degrees
    """
    # Random kernel size
    k_size = random.randint(5, max_kernel_size)
    if k_size % 2 == 0:
        k_size += 1

    # Random angle (radians)
    angle = random.uniform(0, 180)
    angle_rad = np.deg2rad(angle)

    # Create empty kernel
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    center = k_size // 2

    # Direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Draw a line in the kernel
    for i in range(k_size):
        offset = i - center
        x = int(center + offset * dx)
        y = int(center + offset * dy)
        if 0 <= x < k_size and 0 <= y < k_size:
            kernel[y, x] = 1.0

    # Normalize
    kernel /= kernel.sum()

    # Apply convolution
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def add_gaussian_noise(image, min_sigma=5, max_sigma=25):
    """
    Add Gaussian noise with sigma randomly chosen
    from [min_sigma, max_sigma].
    """
    sigma = random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def process_image(img_path, scale=0.5):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping unreadable file: {img_path}")
        return None

    # Resize (downscale)
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_img = img.copy()
    # Apply random darkening
    blurred_img = random_darken(img)

    # Apply random-direction motion blur
    augmented_img = motion_blur(blurred_img)

    # Add Gaussian noise
    augmented_img = add_gaussian_noise(augmented_img)

    return resized_img, augmented_img


def main():
    parser = argparse.ArgumentParser(description="Batch image processing tool")
    parser.add_argument("--input", "-i", help="Input image directory")
    parser.add_argument("--output", "-o", help="Output image directory")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "blurred"), exist_ok=True)

    # Supported image extensions
    exts = [".jpg", ".jpeg", ".png"]

    # Collect input image files
    files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in exts
    ]
    files.sort()  # Ensure stable ordering: 1.jpg, 2.jpg, 3.jpg...

    if not files:
        print("No images found in input directory.")
        return

    # Batch processing
    for idx, filename in enumerate(files, start=1):
        in_path = os.path.join(input_dir, filename)
        resized_out_path = os.path.join(output_dir, "original", f"{idx}.jpg")
        augmented_out_path = os.path.join(output_dir, "blurred", f"{idx}.jpg")

        print(f"Processing {in_path} -> {resized_out_path} and {augmented_out_path}")

        resized_img, augmented_img = process_image(in_path)
        cv2.imwrite(resized_out_path, resized_img)
        cv2.imwrite(augmented_out_path, augmented_img)

    print("All images processed successfully!")


if __name__ == "__main__":
    main()