from skimage import io, img_as_float
import numpy as np
from PIL import Image

def fill_extend(img, out): # fill the border of the image with reflect
    # img.shape [rows, cols, channels]
    # out.shape [rows+2, cols+2, channels]
    out[1:-1, 1:-1] = img
    out[0, 1:-1] = img[1, :]
    out[1:-1, 0] = img[:, 1]
    out[-1, 1:-1] = img[-2, :]
    out[1:-1, -1] = img[:, -2]
    return out