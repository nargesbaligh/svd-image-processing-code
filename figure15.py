import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# تنظیمات اولیه
block_size = 32  # اندازه بلوک 32x32 برای تطابق با تصویر

# خواندن تصویر اصلی
image_path = "C:/Users/Asus/Desktop/barchasb.png"  # مسیر تصویر خودتون رو وارد کنید
original_image = Image.open(image_path).convert('L')  # تبدیل به خاکستری
original_image = np.array(original_image)  # تبدیل به آرایه numpy

# تغییر اندازه تصویر به 256x256 (در صورت نیاز)
if original_image.shape != (256, 256):
    original_image = np.array(Image.fromarray(original_image).resize((256, 256)))

# محاسبه ظرفیت بلوک‌ها
image_size = original_image.shape
capacity_map = np.zeros((image_size[0] // block_size, image_size[1] // block_size))
for i in range(0, image_size[0], block_size):
    for j in range(0, image_size[1], block_size):
        block = original_image[i:i+block_size, j:j+block_size]
        # ظرفیت به صورت واریانس بلوک (بلوک‌های زبر واریانس بیشتری دارند)
        capacity_map[i//block_size, j//block_size] = np.var(block)

# تنظیمات برای رسم
fig = plt.figure(figsize=(10, 5))

# (a) تصویر اصلی
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
plt.title('(a) Original')
plt.axis('off')

# (b) نقشه ظرفیت
plt.subplot(1, 2, 2)
plt.imshow(capacity_map, cmap='gray', interpolation='nearest')
plt.title('(b) Capacity')
plt.axis('off')

plt.tight_layout()
plt.suptitle('Figure 15. Block-based Capacity Calculation', y=1.05)
plt.show()