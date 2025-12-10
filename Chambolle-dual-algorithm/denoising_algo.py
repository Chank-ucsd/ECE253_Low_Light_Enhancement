import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from utils import read_image, fill_extend


def gradient(img):
    grad_X = np.roll(img, -1, axis=1) - img
    grad_Y = np.roll(img, -1, axis=0) - img
    return np.stack((grad_X, grad_Y), axis=-1)

def div(p):
    div_X = p[:, :, 0] - np.roll(p[:, :, 0], 1, axis=1)
    div_Y = p[:, :, 1] - np.roll(p[:, :, 1], 1, axis=0)
    return div_X + div_Y

def shrink(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)



def denoise_tv_chambolle_pock(img, lambd=0.1, max_iter=200, tau=0.25, sigma=0.25, theta=1.0):
    u = img.astype(np.float64)
    p = np.zeros((*u.shape, 2), dtype=np.float64)
    u_bar = u.copy()

    for _ in range(max_iter):
        grad_u = gradient(u_bar)
        p_new = p + sigma * grad_u
        norm_p_new = np.maximum(1, np.sqrt(p_new[:, :, 0]**2 + p_new[:, :, 1]**2))
        p = p_new / norm_p_new[..., np.newaxis]

        u_new = (u + tau * div(p) + tau * img) / (1 + tau)

        u_bar = u + theta * (u_new - u)
        u = u_new

    return u

def denoise_tv_chambolle_dual(img, lambd=0.1, max_iter=200):
    u = img.astype(np.float64)
    p = np.zeros((*u.shape, 2), dtype=np.float64)
    
    sigma = 1 / 8.0

    for _ in range(max_iter):
        grad_u = gradient(u)
        p_new = p + sigma * grad_u
        norm_p_new = np.maximum(1, np.linalg.norm(p_new, axis=-1, keepdims=True))
        p = p_new / norm_p_new

        div_p = div(p)
        u = img - lambd * div_p
        u = np.clip(u, 0, 1)

    return u


def denoise_tv_bregman(img, weight=5.0, max_iter=100, eps=0.001):
    #img.shape [rows, cols, channels]
    img_shape = img.shape
    print(f"Image shape: {img_shape}")
    extended_shape = (img_shape[0] + 2, img_shape[1] + 2, img_shape[2])
    out = np.zeros(extended_shape, dtype=np.float64)

    dx = out.copy()
    dy = out.copy()
    bx = out.copy()
    by = out.copy()

    lam = 2 * weight
    rmse = float('inf')
    norm = (weight + 4 * lam)
    out = fill_extend(img, out)

    i = 0
    regularization = np.multiply(img, weight)
    while i < max_iter and rmse > eps:
        uprev = out[1:-1, 1:-1, :]

        ux = out[1:-1, 2:, :] - uprev
        uy = out[2:, 1:-1, :] - uprev

        unew = np.divide(
            (np.multiply((out[2:, 1:-1, :]
                    + out[0:-2, 1:-1, :]
                    + out[1:-1, 2:, :]
                    + out[1:-1, 0:-2, :]

                    + dx[1:-1, 0:-2, :]
                    - dx[1:-1, 1:-1, :]
                    + dy[0:-2, 1:-1, :]
                    - dy[1:-1, 1:-1, :]

                    - bx[1:-1, 0:-2, :]
                    + bx[1:-1, 1:-1, :]
                    - by[0:-2, 1:-1, :]
                    + by[1:-1, 1:-1, :]), lam) + regularization),
            norm)
        
        out[1:-1, 1:-1, :] = unew.copy()
        # print(unew, uprev)
        # print(unew.shape, uprev.shape)
        rmse = np.linalg.norm((unew - uprev).ravel(), ord=2)

        bxx = bx[1:-1, 1:-1, :].copy()
        byy = by[1:-1, 1:-1, :].copy()

        tx = ux + bxx
        ty = uy + byy

        s = np.sqrt(np.multiply(tx, tx) + np.multiply(ty, ty))
        dxx = np.divide(np.add(np.zeros(s.shape, dtype=np.float32), np.multiply(lam, np.multiply(s, tx))), np.add(np.multiply(s, lam), 1))
        dyy = np.divide(np.add(np.zeros(s.shape, dtype=np.float32), np.multiply(lam, np.multiply(s, ty))), np.add(np.multiply(s, lam), 1))

        dx[1:-1, 1:-1, :] = dxx.copy()
        dy[1:-1, 1:-1, :] = dyy.copy()

        bx[1:-1, 1:-1, :] += ux - dxx
        by[1:-1, 1:-1, :] += uy - dyy

        i += 1

    return out[1:-1, 1:-1, :]





if __name__ == "__main__":
    image_path = "image1.jpeg" 
    method = 'bregman'
    is_gray = False
    image = read_image(image_path, is_gray)


    noisy_image = image + 0.1 * np.random.normal(size=image.shape)
    noisy_image = np.clip(noisy_image, 0, 1)


    if method == 'bregman':
        denoised_image = denoise_tv_bregman(noisy_image, weight=0.2)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(noisy_image, cmap='gray')
    ax[1].set_title("Noisy Image")
    ax[1].axis("off")

    ax[2].imshow(denoised_image, cmap='gray')
    ax[2].set_title("Denoised Image (TV)")
    ax[2].axis("off")

    plt.show()