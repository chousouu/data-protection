import numpy as np
import json
from pathlib import Path
import cv2

from scipy.special import kl_div
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio
import skimage.measure


def image_to_gray(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def histogram(original: np.ndarray, encrypted: np.ndarray): #TODO: scipy.special has bad implementation. change
    return kl_div(original, encrypted)


def correlation(original: np.ndarray, encrypted: np.ndarray):
    pearson = pearsonr(original.flatten(), encrypted.flatten())
    return pearson.statistic


def NPCR(original: np.ndarray, encrypted: np.ndarray):
    """
    original.shape = (H, W, C), np.uint8
    """
    difference_mask = (original == encrypted).all(axis=2)
    return np.mean(difference_mask, axis=(0, 1)) * 100


def UACI(original: np.ndarray, encrypted: np.ndarray, 
         bit_per_color: int = 8):
    original = image_to_gray(original)
    encrypted = image_to_gray(encrypted)

    intensity_diff = np.abs(original - encrypted) / (2 ** bit_per_color - 1)
    return np.mean(intensity_diff) * 100


def cipher_size(original: np.ndarray, encrypted: np.ndarray):
    return encrypted.size / 3 # since R,G,B


def PSNR(original: np.ndarray, encrypted: np.ndarray):
    return peak_signal_noise_ratio(original, encrypted, data_range=255)


def MSE(original: np.ndarray, encrypted: np.ndarray):
    mse = (original - encrypt) ** 2
    return np.mean(mse)


def entropy(original: np.ndarray, encrypted: np.ndarray):
    return skimage.measure.shannon_entropy(encrypted)


class Logger():
    def __init__(self, encryption_name: str, image_type: str = 'int8'):
        if image_type != 'int8':
            raise ValueError('Working with only uint8 type')
        
        self.method = encryption_name
        self.data_range = 255
        self.bit_color = 8

        self.metric_list = ['Hist', 'Correlation', 
                            'NPCR', 'UACI', 
                            'Cipher_size', 'PSNR',
                            'MSE', 'Entropy']
        self._funcs_list = [histogram, correlation, 
                            NPCR, UACI, 
                            cipher_size, PSNR,
                            MSE, entropy]
        self._metrics_func = {key: val for key, val in zip(self.metric_list, self._funcs_list)}

        self.metrics = {metric: list() for metric in self.metric_list}
    

    def clear_metrics(self):
        self.metrics = {key: list() for key in self.metrics.keys()}


    def compute_metrics(self, 
                        original_img: np.ndarray, encryped_img: np.ndarray):
        for key in self.metrics.keys():
            func = self._metrics_func[key]
            self.metrics[key].append(func(original_img, encryped_img))
        return self.metrics
    

    def log_metrics(self):
        print(f"---- Enctrypthon method: {self.method} ----")
        for key, val_arr in self.metrics.items():
            print(f"{key}: {np.array(val_arr).mean()}")


if __name__ == '__main__':
    orig = np.random.randn(64, 64, 3).astype(np.uint8)
    encrypt = np.random.randn(64, 64, 3).astype(np.uint8)

    logger = Logger('bbb')
    logger.compute_metrics(orig, encrypt)
    logger.log_metrics()