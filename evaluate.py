import argparse
import os
import numpy as np
from PIL import Image
from glob import glob
from natsort import natsorted
import csv
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_snr(image1, image2):
    """
    Calculate Signal-to-Noise Ratio (SNR) between two images.
    
    Args:
        image1: numpy array of first image (signal/reference)
        image2: numpy array of second image (noisy/processed)
    
    Returns:
        SNR value in dB
    """
    # Convert to float and normalize to [0, 1] if needed
    if image1.dtype != np.float64:
        image1 = image1.astype(np.float64)
    if image2.dtype != np.float64:
        image2 = image2.astype(np.float64)
    
    # Normalize to [0, 1] if images are in [0, 255] range
    if image1.max() > 1.0:
        image1 = image1 / 255.0
    if image2.max() > 1.0:
        image2 = image2 / 255.0
    
    # Calculate noise (difference between images)
    noise = image1 - image2
    
    # Calculate signal power (mean of squared signal values)
    signal_power = np.mean(image1 ** 2)
    
    # Calculate noise power (mean of squared noise values)
    noise_power = np.mean(noise ** 2)
    
    # Calculate SNR in dB
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')
    
    return snr


def compute_psnr(image1, image2):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images."""
    return peak_signal_noise_ratio(image1, image2, data_range=1.0)


def compute_ssim(image1, image2):
    """Calculate Structural Similarity Index (SSIM) between two images."""
    return structural_similarity(image1, image2, channel_axis=2, data_range=1.0)


def load_image(image_path):
    """Load image and convert to numpy array."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def get_image_files(directory):
    """Get all image files from directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, ext)))
    return natsorted(files)


def match_image_files(files1, files2):
    """
    Match image files from two directories by filename (without extension).
    Returns list of tuples: (file1_path, file2_path, base_name)
    """
    # Create dictionaries mapping base names to full paths
    dict1 = {}
    for f in files1:
        base_name = os.path.splitext(os.path.basename(f))[0]
        dict1[base_name] = f
    
    dict2 = {}
    for f in files2:
        base_name = os.path.splitext(os.path.basename(f))[0]
        dict2[base_name] = f
    
    # Find matching pairs
    matched_pairs = []
    for base_name in dict1:
        if base_name in dict2:
            matched_pairs.append((dict1[base_name], dict2[base_name], base_name))
        else:
            print(f"Warning: No matching file for {base_name} in label directory")
    
    return matched_pairs


def main():
    parser = argparse.ArgumentParser(description='Calculate SNR, PSNR and SSIM between image pairs in two directories')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Input directory 1 (noisy/processed images)')
    parser.add_argument('--label_dir', '-l', type=str, required=True,
                        help='Label directory 2 (signal/reference images)')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image files from both directories
    print(f"Loading images from {args.input_dir}...")
    files1 = get_image_files(args.input_dir)
    print(f"Found {len(files1)} images in input directory")
    
    print(f"Loading images from {args.label_dir}...")
    files2 = get_image_files(args.label_dir)
    print(f"Found {len(files2)} images in label directory")
    
    if len(files1) == 0:
        print(f"Error: No image files found in {args.input_dir}")
        return
    
    if len(files2) == 0:
        print(f"Error: No image files found in {args.label_dir}")
        return
    
    # Match image files
    matched_pairs = match_image_files(files1, files2)
    
    if len(matched_pairs) == 0:
        print("Error: No matching image pairs found between the two directories")
        return
    
    print(f"Found {len(matched_pairs)} matching image pairs")
    print("-" * 60)
    
    # Calculate metrics for each pair
    results = []
    snr_values = []
    psnr_values = []
    ssim_values = []
    
    for file1, file2, base_name in matched_pairs:
        try:
            # Load images
            img1 = load_image(file1)
            img2 = load_image(file2)
            
            # Check if images have the same shape
            if img1.shape != img2.shape:
                print(f"Warning: {base_name} - Images have different shapes: {img1.shape} vs {img2.shape}")
                # Resize img2 to match img1
                from PIL import Image
                img2_pil = Image.fromarray(img2)
                img1_pil = Image.fromarray(img1)
                img2_pil = img2_pil.resize(img1_pil.size, Image.Resampling.LANCZOS)
                img2 = np.array(img2_pil)
            
            # Calculate metrics
            snr = compute_snr(img1, img2)
            psnr = compute_psnr(img1, img2)
            ssim = compute_ssim(img1, img2)
            snr_values.append(snr)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            results.append({
                'filename': base_name,
                'input_file': os.path.basename(file1),
                'label_file': os.path.basename(file2),
                'SNR': snr,
                'PSNR': psnr,
                'SSIM': ssim
            })
            
            print(f"{base_name}: SNR = {snr:.4f} dB | PSNR = {psnr:.4f} dB | SSIM = {ssim:.4f}")
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            continue
    
    # Calculate statistics
    if len(snr_values) > 0:
        mean_snr = np.mean(snr_values)
        std_snr = np.std(snr_values)
        min_snr = np.min(snr_values)
        max_snr = np.max(snr_values)
        mean_psnr = np.mean(psnr_values)
        std_psnr = np.std(psnr_values)
        min_psnr = np.min(psnr_values)
        max_psnr = np.max(psnr_values)
        mean_ssim = np.mean(ssim_values)
        std_ssim = np.std(ssim_values)
        min_ssim = np.min(ssim_values)
        max_ssim = np.max(ssim_values)
        
        print("-" * 60)
        print(f"Statistics:")
        print(f"  Total images: {len(snr_values)}")
        print(f"  Mean SNR: {mean_snr:.4f} dB")
        print(f"  Std SNR: {std_snr:.4f} dB")
        print(f"  Min SNR: {min_snr:.4f} dB")
        print(f"  Max SNR: {max_snr:.4f} dB")
        print("-" * 60)
        print("\n")
        print(f"  Mean PSNR: {mean_psnr:.4f} dB")
        print(f"  Std PSNR: {std_psnr:.4f} dB")
        print(f"  Min PSNR: {min_psnr:.4f} dB")
        print(f"  Max PSNR: {max_psnr:.4f} dB")
        print("-" * 60)
        print("\n")
        print(f"  Mean SSIM: {mean_ssim:.4f}")
        print(f"  Std SSIM: {std_ssim:.4f}")
        print(f"  Min SSIM: {min_ssim:.4f}")
        print(f"  Max SSIM: {max_ssim:.4f}")
        print("-" * 60)
        print("\n")
        
        # Save results to CSV
        csv_path = os.path.join(args.output_dir, 'snr_results.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'input_file', 'label_file', 'SNR', 'PSNR', 'SSIM']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to {csv_path}")
        
        # Save summary to text file
        summary_path = os.path.join(args.output_dir, 'snr_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Image Quality Evaluation Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Input directory: {args.input_dir}\n")
            f.write(f"Label directory: {args.label_dir}\n\n")
            f.write(f"Total images: {len(snr_values)}\n")
            f.write(f"Mean SNR: {mean_snr:.4f} dB\n")
            f.write(f"Std SNR: {std_snr:.4f} dB\n")
            f.write(f"Min SNR: {min_snr:.4f} dB\n")
            f.write(f"Max SNR: {max_snr:.4f} dB\n")
            f.write("-" * 60)
            f.write("\n")
            f.write(f"Mean PSNR: {mean_psnr:.4f} dB\n")
            f.write(f"Std PSNR: {std_psnr:.4f} dB\n")
            f.write(f"Min PSNR: {min_psnr:.4f} dB\n")
            f.write(f"Max PSNR: {max_psnr:.4f} dB\n")
            f.write("-" * 60)
            f.write("\n")
            f.write(f"Mean SSIM: {mean_ssim:.4f}\n")
            f.write(f"Std SSIM: {std_ssim:.4f}\n")
            f.write(f"Min SSIM: {min_ssim:.4f}\n")
            f.write(f"Max SSIM: {max_ssim:.4f}\n")
            f.write("-" * 60)
            f.write("\n")
            f.write("Individual Results:\n")
            for result in results:
                f.write(
                    f"{result['filename']}: "
                    f"SNR={result['SNR']:.4f} dB, "
                    f"PSNR={result['PSNR']:.4f} dB, "
                    f"SSIM={result['SSIM']:.4f}\n"
                )
        
        print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
