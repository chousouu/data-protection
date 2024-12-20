from Crypto.Cipher import DES3
from Crypto.Util.Padding import pad, unpad
from PIL import Image
import numpy as np
import os
import struct

image_path = 'image.png'

key = os.urandom(24)
iv = os.urandom(8)

def encrypt_image(input_image_path, key, iv):
    image = Image.open(input_image_path)
    image_data = np.array(image)

    byte_data = image_data.tobytes()

    cipher = DES3.new(key, DES3.MODE_CBC, iv)

    padded_data = pad(byte_data, DES3.block_size)
    encrypted_data = cipher.encrypt(padded_data)
    
    img_enc = Image.fromarray(np.frombuffer(encrypted_data[:277**2], dtype=np.uint8).reshape((277, 277)))
    img_enc.save('encrypted_image.png')

    output_data = struct.pack('ii', image.size[0], image.size[1]) + cipher.iv + encrypted_data
    return output_data

def decrypt_image(encrypted_data, key):
    width, height = struct.unpack('ii', encrypted_data[:8])
    iv = encrypted_data[8:16]
    encrypted_content = encrypted_data[16:]

    cipher = DES3.new(key, DES3.MODE_CBC, iv)

    decrypted_data = unpad(cipher.decrypt(encrypted_content), DES3.block_size)

    image_data = np.frombuffer(decrypted_data, dtype=np.uint8)
    image_data = image_data.reshape((height, width, 3))

    decrypted_image = Image.fromarray(image_data)
    return decrypted_image

def calculate_metrics(original_image_path, decrypted_image):
    original_image = Image.open(original_image_path)
    original_data = np.array(original_image)
    decrypted_data = np.array(decrypted_image)
    
    total_pixels = original_data.size
    changed_pixels = np.sum(original_data != decrypted_data)
    
    npcr = (changed_pixels / total_pixels) * 100 

    uaci = np.mean(np.abs(original_data.astype(np.int32) - decrypted_data.astype(np.int32))) / 255 * 100

    mse = np.mean((original_data - decrypted_data) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255**2 / mse)

    print(f"PSNR: {psnr:.2f} dB")
    print(f"NPCR: {npcr:.4f}")
    print(f"UACI: {uaci:.2f}%")

if __name__ == "__main__":
    encrypted_data = encrypt_image(image_path, key, iv)
    decrypted_image = decrypt_image(encrypted_data, key)

    decrypted_image.save('decrypted_image.png')

    calculate_metrics(image_path, decrypted_image)