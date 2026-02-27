import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#figure12
# تابع برای بارگذاری و تبدیل تصویر به ماتریس خاکستری
def load_image(image_path, size=None):
    img = Image.open(image_path).convert('L')  # تبدیل به خاکستری
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS) 
    return np.array(img)

# تابع برای اضافه کردن نویز به تصویر
#با استفاده از تابع، نویز گاوسی به تصویر صاف اضافه می‌شود تا تصویر پرنویز تولید شود
def add_noise(image, noise_level=0.1):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

# بارگذاری تصویر صاف
smooth_image_path = "C:/Users/Asus/Desktop/images.jpg"
size = (256, 256)  # اندازه فرضی
smooth_img = load_image(smooth_image_path, size)

# نرمال‌سازی تصویر به بازه [0, 1]
smooth_img = smooth_img / 255.0

# تولید تصویر پرنویز با اضافه کردن نویز
noisy_img = add_noise(smooth_img, noise_level=0.1)

# محاسبه SVD برای هر دو تصویر
_, S_smooth, _ = np.linalg.svd(smooth_img, full_matrices=True)
_, S_noisy, _ = np.linalg.svd(noisy_img, full_matrices=True)

# رسم نمودار اصلی
fig, ax = plt.subplots(figsize=(8, 6))
#از plt.semilogy برای رسم مقادیر تکین در مقیاس لگاریتمی استفاده شده است
ax.semilogy(S_smooth, label='Image', color='red', linestyle='-')
ax.semilogy(S_noisy, label='Noise', color='blue', linestyle='--')
ax.set_xlabel('Index')
ax.set_ylabel('SVs')
ax.set_title('Figure 12: Rate of SVs Decaying')
ax.legend()
ax.grid(True, which="both", ls="--")

# اضافه کردن کادر کوچک (Inset) برای نمایش جزئیات اولیه
inset_ax = ax.inset_axes([0.1, 0.5, 0.3, 0.3])  # موقعیت و اندازه کادر کوچک
inset_ax.semilogy(S_smooth[:100], color='red', linestyle='-')
inset_ax.semilogy(S_noisy[:100], color='blue', linestyle='--')
inset_ax.set_xlim(0, 70)  # محدود کردن به 100 مقدار اول
inset_ax.grid(True, which="both", ls="--")

plt.show()

# ذخیره نمودار
plt.savefig('figure_12.png')

################################

#figure10
# تابع برای بارگذاری و تبدیل تصویر به ماتریس خاکستری
def load_image(image_path, size=None):
    img = Image.open(image_path).convert('L') 
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)  
    return np.array(img)

# بارگذاری تصاویر
original_image_path = "C:/Users/Asus/Desktop/Screenshot_a.png" 
faked_image_path = "C:/Users/Asus/Desktop/Screenshot_b.png" 

# تنظیم اندازه تصاویر (باید هم‌اندازه باشند)
size = (256, 256)  # اندازه فرضی (باید با اندازه تصاویر شما تنظیم شود)
original_img = load_image(original_image_path, size)
faked_img = load_image(faked_image_path, size)

# نرمال‌سازی تصاویر به بازه [0, 1]
original_img = original_img / 255.0
faked_img = faked_img / 255.0

# محاسبه SVD برای هر دو تصویر
_, S_original, _ = np.linalg.svd(original_img, full_matrices=True)
_, S_faked, _ = np.linalg.svd(faked_img, full_matrices=True)

# رسم نمودار مقادیر تکین در مقیاس لگاریتمی
plt.figure(figsize=(8, 6))
plt.semilogy(S_original, label='Original', linestyle='-', color='black')
plt.semilogy(S_faked, label='Faked', linestyle=':', color='red')
plt.xlabel('Index')
plt.ylabel('Singular Values (Log Scale)')
plt.title('(c) SVs of both of (a) and (b)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# ذخیره نمودار
plt.savefig('figure_10c.png')