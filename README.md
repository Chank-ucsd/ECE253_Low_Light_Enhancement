# ECE253 Low Light Image Enhancement

This project implements and compares various methods for low-light image enhancement, motion deblurring, and denoising. The project uses **Restormer** for motion deblurring and denoising, and **Zero-DCE++** for low-light enhancement.

## Overview

This repository contains implementations and comparisons of multiple image enhancement methods:

- **Restormer**: Efficient Transformer for motion deblurring and real image denoising
- **Zero-DCE++**: Zero-reference deep curve estimation for low-light enhancement
- **Histogram Equalization**: Traditional enhancement method
- **Richardson-Lucy Deconvolution**: Iterative deblurring algorithm
- **Chambolle Dual Algorithm**: Total variation denoising

## Project Structure

```
ECE253_Low_Light_Enhancement/
├── Restormer/                    # Restormer implementation
│   ├── Motion_Deblurring/        # Motion deblurring module
│   ├── Denoising/                # Image denoising module
│   └── basicsr/                  # BasicSR utilities
├── Zero-DCE_extension/           # Zero-DCE++ implementation
│   └── Zero-DCE++/
│       ├── lowlight_test.py      # Testing script
│       ├── lowlight_train.py     # Training script
│       └── snapshots_Zero_DCE++/ # Pretrained models
├── Histogram-Equalization/      # Histogram equalization method
├── Chambolle-dual-algorithm/     # Chambolle denoising
├── richardson-lucy-python/       # Richardson-Lucy deconvolution
├── evaluate.py                   # Evaluation script (SNR, PSNR, SSIM)
├── compare_methods.py            # Method comparison and visualization
├── plot_two_bars.py              # Bar chart plotting utility
├── data_augment.py               # Data augmentation script
└── dataset/                      # Dataset directory
```

## Usage

### 1. Low-Light Enhancement with Zero-DCE++

Test on custom dataset:
```bash
cd Zero-DCE_extension/Zero-DCE++
python lowlight_test.py -i <input_dir> -o <output_dir> -m <model_path>
```

Example:
```bash
python lowlight_test.py \
    -i /path/to/lowlight/images \
    -o /path/to/output \
    -m snapshots_Zero_DCE++/Epoch99.pth
```

### 2. Motion Deblurring with Restormer

Test on custom dataset:
```bash
cd Restormer/Motion_Deblurring
python test.py -i <input_dir> -o <output_dir> -w <weights_path>
```

Example:
```bash
python test.py \
    -i /path/to/blurred/images \
    -o /path/to/output \
    -w pretrained_models/motion_deblurring.pth
```

### 3. Real Image Denoising with Restormer

Test on custom dataset:
```bash
cd Restormer/Denoising
python test_real_denoising_sidd.py -i <input_dir> -o <output_dir> -w <weights_path>
```

Example:
```bash
python test_real_denoising_sidd.py \
    -i /path/to/noisy/images \
    -o /path/to/output \
    -w pretrained_models/real_denoising.pth
```

### 4. Evaluate Image Quality

Calculate SNR, PSNR, and SSIM between two image directories:
```bash
python evaluate.py -i <processed_dir> -l <reference_dir> -o <output_dir>
```

Example:
```bash
python evaluate.py \
    -i dataset/enhanced_images \
    -l dataset/original_images \
    -o dataset/evaluation_results
```

This will generate:
- `snr_results.csv`: Detailed results for each image
- `snr_summary.txt`: Summary statistics

### 5. Compare Methods

Compare multiple methods and generate visualizations:
```bash
python compare_methods.py
```

This script:
- Parses evaluation summaries from different methods
- Generates comparison charts (bar charts, radar charts, line plots, heatmaps)
- Saves results to `method_comparison.csv`

### 6. Data Augmentation

Generate augmented dataset with motion blur, darkening, and noise:
```bash
python data_augment.py -i <input_dir> -o <output_dir> [--split]
```

Options:
- `-i, --input`: Input image directory
- `-o, --output`: Output directory
- `-s, --split`: (Optional) Split each image into 2x2 patches

Example:
```bash
# Process whole images
python data_augment.py -i dataset/original -o dataset/augmented

# Split into 4 patches per image
python data_augment.py -i dataset/original -o dataset/augmented --split
```

## Pretrained Models

The repository includes pretrained models:
- **Zero-DCE++**: `Zero-DCE_extension/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth`
- **Restormer Motion Deblurring**: Check `Restormer/Motion_Deblurring/pretrained_models/`
- **Restormer Denoising**: Check `Restormer/Denoising/pretrained_models/`

## Evaluation Metrics

The project uses three main metrics for evaluation:

1. **SNR (Signal-to-Noise Ratio)**: Measures signal power relative to noise
2. **PSNR (Peak Signal-to-Noise Ratio)**: Measures peak signal power relative to noise
3. **SSIM (Structural Similarity Index)**: Measures structural similarity between images

## Visualization Tools

### Plot Two Bars

Compare two methods with a bar chart:
```python
from plot_two_bars import plot_two_bars

plot_two_bars(
    category1_name="Method A",
    value1=25.5,
    category2_name="Method B",
    value2=30.2,
    ylabel="PSNR (dB)",
    title="Method Comparison",
    save_name="comparison.png"
)
```

## Dataset Structure

The dataset directory should be organized as follows:
```
dataset/
├── original/          # Original reference images
├── blurred/           # Blurred/processed images
├── enhanced/          # Enhanced results
└── evaluation/        # Evaluation results
```

## Notes

- All test scripts support custom dataset directories
- Images are automatically resized/padded to meet model requirements
- Evaluation scripts automatically match images by filename (without extension)
- The project supports multiple image formats: JPG, JPEG, PNG, BMP

## Citation

If you use this code, please cite the original papers:

**Restormer:**
```
@article{zamir2022restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Zamir, Syed Waqas and Arora, Aditya and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Yang, Ming-Hsuan},
  journal={CVPR},
  year={2022}
}
```

**Zero-DCE++:**
```
@article{li2021zero,
  title={Zero-DCE++: Zero-reference Deep Curve Estimation for Low-light Image Enhancement},
  author={Li, Chunming and Guo, Chunjie and Chen, Chang Loy},
  journal={NeurIPS},
  year={2021}
}
```

## License

Please refer to the LICENSE files in respective subdirectories for license information.

## Contact

For questions or issues, please open an issue on the repository.
