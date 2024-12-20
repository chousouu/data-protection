from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

import random
import secrets

from PIL import Image
import numpy as np
import io
from matplotlib import pyplot as plt

image_path = 'image.png'

key = secrets.token_bytes(16)
iv = secrets.token_bytes(16)

def encrypt_aes(image: np.ndarray, key: bytes, iv: bytes):
    image_data = image.tobytes()
    print(len(image_data))
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    # image_data = pad(image_data, AES.block_size)
        
    encrypted = iv + cipher.encrypt(image_data)
    print(len(encrypted))
    return encrypted

def decrypt_aes(encrypted, key, shape):
    iv = encrypted[:AES.block_size]
    encrypted = encrypted[AES.block_size:]
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    decrypted = cipher.decrypt(encrypted)
    # decrypted = unpad(decrypted, AES.block_size)
    print(len(decrypted))
    arr = np.frombuffer(decrypted, dtype=np.uint8).reshape(shape)
    image = Image.fromarray(arr)
    return image

if __name__ == "__main__":

    img = Image.open(image_path).resize((256, 256))
    img_arr = np.array(img, dtype=np.uint8)
    img_shape = img_arr.shape

    encrypted = encrypt_aes(img_arr, key, iv)
    img_dec = decrypt_aes(encrypted, key, img_shape)

    img_enc = Image.fromarray(np.frombuffer(encrypted[AES.block_size:], dtype=np.uint8).reshape(img_shape))

    plt.imshow(img_dec)