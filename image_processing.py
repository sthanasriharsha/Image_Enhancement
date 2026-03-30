import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — required on servers
import matplotlib.pyplot as plt
from collections import Counter
import cv2

IMG_PATH = "static/img/img_now.jpg"
NORMAL_PATH = "static/img/img_normal.jpg"


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def open_rgb(path=IMG_PATH):
    """Always return an RGB uint8 numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_arr(arr, path=IMG_PATH):
    """Save a uint8 numpy array as JPEG."""
    Image.fromarray(arr.astype(np.uint8)).save(path)


def is_grey_scale(img_path=IMG_PATH):
    """Return True if the image is greyscale (R == G == B for all pixels)."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return np.array_equal(r, g) and np.array_equal(g, b)


# ─────────────────────────────────────────────
#  COLOUR / TONE
# ─────────────────────────────────────────────

def grayscale():
    arr = open_rgb().astype(np.uint16)
    grey = ((arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) // 3).astype(np.uint8)
    grey_rgb = np.stack([grey, grey, grey], axis=2)
    save_arr(grey_rgb)


def brightness_addition():
    arr = open_rgb().astype(np.int16)
    save_arr(np.clip(arr + 100, 0, 255))


def brightness_substraction():
    arr = open_rgb().astype(np.int16)
    save_arr(np.clip(arr - 100, 0, 255))


def brightness_multiplication():
    arr = open_rgb().astype(np.float32)
    save_arr(np.clip(arr * 1.25, 0, 255))


def brightness_division():
    arr = open_rgb().astype(np.float32)
    save_arr(np.clip(arr / 1.25, 0, 255))


# ─────────────────────────────────────────────
#  LOW-LIGHT ENHANCEMENT  (new / improved)
# ─────────────────────────────────────────────

def gamma_correction(gamma=0.5):
    """Brighten a dark image using gamma correction (gamma < 1 = brighter)."""
    arr = open_rgb().astype(np.float32) / 255.0
    corrected = np.power(arr, gamma)
    save_arr((corrected * 255).astype(np.uint8))


def clahe_enhancement():
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation) on L channel
    in LAB colour space — best all-round low-light enhancer.
    """
    img_bgr = cv2.imread(IMG_PATH)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    cv2.imwrite(IMG_PATH, result)


def retinex_enhancement():
    """
    Single-Scale Retinex — illumination normalisation for low-light images.
    """
    arr = open_rgb().astype(np.float32) + 1.0   # avoid log(0)
    sigma = 30
    result = np.zeros_like(arr)
    for i in range(3):
        blurred = cv2.GaussianBlur(arr[:, :, i], (0, 0), sigma)
        result[:, :, i] = np.log(arr[:, :, i]) - np.log(blurred + 1.0)
    # Normalise each channel to 0-255
    for i in range(3):
        ch = result[:, :, i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6) * 255
        result[:, :, i] = ch
    save_arr(result.astype(np.uint8))


def denoise():
    """Fast Non-Local Means denoising."""
    img_bgr = cv2.imread(IMG_PATH)
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
    cv2.imwrite(IMG_PATH, denoised)


# ─────────────────────────────────────────────
#  SPATIAL TRANSFORMS
# ─────────────────────────────────────────────

def zoomin():
    """2× zoom in using cv2 resize (fast)."""
    img = cv2.imread(IMG_PATH)
    h, w = img.shape[:2]
    zoomed = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(IMG_PATH, zoomed)


def zoomout():
    """2× zoom out using cv2 resize (fast)."""
    img = cv2.imread(IMG_PATH)
    h, w = img.shape[:2]
    zoomed = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(IMG_PATH, zoomed)


def move_left():
    arr = open_rgb()
    shifted = np.roll(arr, -50, axis=1)
    shifted[:, -50:, :] = 0
    save_arr(shifted)


def move_right():
    arr = open_rgb()
    shifted = np.roll(arr, 50, axis=1)
    shifted[:, :50, :] = 0
    save_arr(shifted)


def move_up():
    arr = open_rgb()
    shifted = np.roll(arr, -50, axis=0)
    shifted[-50:, :, :] = 0
    save_arr(shifted)


def move_down():
    arr = open_rgb()
    shifted = np.roll(arr, 50, axis=0)
    shifted[:50, :, :] = 0
    save_arr(shifted)


# ─────────────────────────────────────────────
#  FILTERS  (all now use cv2 — fast & correct)
# ─────────────────────────────────────────────

def edge_detection():
    img = cv2.imread(IMG_PATH)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)
    result = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(IMG_PATH, result)


def blur():
    img = cv2.imread(IMG_PATH)
    result = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imwrite(IMG_PATH, result)


def sharpening():
    img = cv2.imread(IMG_PATH)
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    result = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(IMG_PATH, result)


# ─────────────────────────────────────────────
#  HISTOGRAM
# ─────────────────────────────────────────────

def histogram_rgb():
    arr = open_rgb()
    plt.figure(figsize=(6, 3))
    if is_grey_scale():
        g = arr[:, :, 0].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black', width=1)
        plt.tight_layout()
        plt.savefig('static/img/grey_histogram.jpg', dpi=150)
        plt.clf()
        plt.close()
    else:
        for ch_idx, (name, color) in enumerate([('red', 'red'), ('green', 'green'), ('blue', 'blue')]):
            data = Counter(arr[:, :, ch_idx].flatten())
            plt.bar(list(data.keys()), data.values(), color=color, width=1)
            plt.tight_layout()
            plt.savefig(f'static/img/{name}_histogram.jpg', dpi=150)
            plt.clf()
            plt.close()


def histogram_equalizer():
    """Equalise using CLAHE on L channel (better than plain global equalisation)."""
    clahe_enhancement()


# ─────────────────────────────────────────────
#  THRESHOLDING
# ─────────────────────────────────────────────

def threshold(lower_thres, upper_thres):
    arr = open_rgb()
    condition = (arr >= lower_thres) & (arr <= upper_thres)
    arr[condition] = 255
    save_arr(arr)
