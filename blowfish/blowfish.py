from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from PIL import Image
import os
import numpy as np

key = os.urandom(16)
iv = os.urandom(8)

def encrypt_image(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    cipher = Cipher(algorithms.Blowfish(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(algorithms.Blowfish.block_size).padder()
    padded_data = padder.update(image_data) + padder.finalize()
    print(type(padded_data))
    
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    return encrypted_data

def decrypt_image(encrypted_data):
    cipher = Cipher(algorithms.Blowfish(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    unpadder = padding.PKCS7(algorithms.Blowfish.block_size).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    
    return unpadded_data

#def calculate_metrics(original_image_path, decrypted_image_data):
#    # Загружаем оригинальное изображение для сравнения
#    original_image = Image.open(original_image_path)
#    original_data = np.array(original_image)

#    # Сохраняем расшифрованные данные в временный файл для дальнейшей обработки
#    temp_decrypted_image_path = 'temp_decrypted_image.png'
#    with open(temp_decrypted_image_path, 'wb') as temp_file:
#        temp_file.write(decrypted_image_data)

#    decrypted_image = Image.open(temp_decrypted_image_path)
#    decrypted_data = np.array(decrypted_image)

#    # 1. PSNR (Peak Signal-to-Noise Ratio)
#    mse = np.mean((original_data - decrypted_data) ** 2)
#    if mse == 0:
#        psnr = float('inf')  # Если MSE равно нулю, PSNR бесконечно
#    else:
#        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

#    # 2. NPCR (Number of Pixels Change Rate)
#    original_flat = original_data.flatten()
#    decrypted_flat = decrypted_data.flatten()
#    npcr = np.sum(original_flat != decrypted_flat) / original_flat.size

#    # 3. UACI (Unified Average Changing Intensity)
#    uaci = np.mean(np.abs(original_flat - decrypted_flat)) / 255.0 * 100

#    print(f"PSNR: {psnr:.2f} dB")
#    print(f"NPCR: {npcr:.4f}")
#    print(f"UACI: {uaci:.2f}%")

image_path = 'image.png'

encrypted_image_data = encrypt_image(image_path)

with open('encrypted_image.enc', 'wb') as encrypted_file:
    encrypted_file.write(iv + encrypted_image_data)

print("Изображение зашифровано и сохранено как encrypted_image.enc")

with open('encrypted_image.enc', 'rb') as encrypted_file:
    iv = encrypted_file.read(8)
    encrypted_image_data = encrypted_file.read()

decrypted_image_data = decrypt_image(encrypted_image_data)

with open('decrypted_image.png', 'wb') as decrypted_file:
    decrypted_file.write(decrypted_image_data)

print("Изображение расшифровано и сохранено как decrypted_image.png")

# Рассчитываем метрики
#calculate_metrics(image_path, decrypted_image_data)