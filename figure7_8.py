import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#figure 7
#بارگذاری تصویر و تبدیل آن به حالت خاکستری
image_path = "C:/Users/Asus/Desktop/Figure7.jpg"  
image = Image.open(image_path)
image = image.resize((200, 200)) 
image = image.convert('L') 
image_array = np.array(image)

# انجام تجزیه SVD
U, S, V = np.linalg.svd(image_array, full_matrices=True)

# بازسازی تصویر با استفاده از k=60 مقدار تکینِ اول (فشرده‌سازی 47٪)
k1 = 60  
S_k1 = np.zeros_like(image_array, dtype=float)
S_k1[:k1, :k1] = np.diag(S[:k1]) 
compressed_image_k60 = np.dot(U[:, :k1], np.dot(S_k1[:k1, :k1], V[:k1, :]))

# بازسازی تصویر با استفاده از k=20 مقدار تکینِ اول (فشرده‌سازی 16٪)
k2 = 20  
S_k2 = np.zeros_like(image_array, dtype=float)
S_k2[:k2, :k2] = np.diag(S[:k2]) 
compressed_image_k20 = np.dot(U[:, :k2], np.dot(S_k2[:k2, :k2], V[:k2, :]))

# محاسبه نسبت‌های فشرده‌سازی برای بررسی
m, n = image_array.shape # ابعاد تصویر استخراج می شوند
# فرمول نسبت فشرده‌سازی: R = (mk + nk + k) / (mn) * 100
R_k60 = (m * k1 + n * k1 + k1) / (m * n) * 100
R_k20 = (m * k2 + n * k2 + k2) / (m * n) * 100
print(f"Compression ratio for k={k1}: {R_k60:.2f}%")
print(f"Compression ratio for k={k2}: {R_k20:.2f}%")

# تنظیم شکل برای نمایش سه تصویر
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#نمایش تصویر اصلی
axes[0].imshow(image_array, cmap='gray')
axes[0].set_title('(a) Original')
axes[0].axis('off')

# نمایش تصویر فشرده‌شده با k=60 (فشرده‌سازی 47٪)
axes[1].imshow(compressed_image_k60, cmap='gray')
axes[1].set_title('(b) Compression 47% (k=60)')
axes[1].axis('off')

# نمایش تصویر فشرده‌شده با k=20 (فشرده‌سازی 16٪)
axes[2].imshow(compressed_image_k20, cmap='gray')
axes[2].set_title('(c) Compression 16% (k=20)')
axes[2].axis('off')

# تنظیم عنوان کلی شکل
plt.suptitle('Figure 7. SVD Based Compression', fontsize=16)

# ذخیره‌ی شکل (اختیاری)
plt.savefig("figure7_svd_compression.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()
##########################################

#figure 8
# تابع برای بارگذاری و تبدیل تصویر به ماتریس خاکستری
def load_image(image_path, size=None):
    img = Image.open(image_path).convert('L')  # تبدیل به خاکستری
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)  # تغییر اندازه
    return np.array(img)

#تابع برای محاسبه NMSE
def calculate_nmse(original, modified):
    mse = np.mean((original - modified) ** 2)
    return mse / np.mean(original ** 2)

# بارگذاری تصاویر
# فرض می‌کنیم تصویر (d) همان تصویر میزبان اصلی است
host_image_path = "C:/Users/Asus/Desktop/images_keshtijpg.jpg"  # مسیر تصویر میزبان (قایق‌ها)
watermark_image_path = "C:/Users/Asus/Desktop/eyes.png"   # مسیر تصویر علامت (چشم هوروس)


#تنظیم اندازه تصاویر
size = (256, 256) 
host_img = load_image(host_image_path, size)
watermark_img = load_image(watermark_image_path, size)

# نرمال‌سازی تصاویر به بازه [0, 1]
host_img = host_img / 255.0
watermark_img = watermark_img / 255.0

# محاسبه SVD برای تصویر میزبان و علامت
U_h, S_h, V_h = np.linalg.svd(host_img, full_matrices=True)
U_w, S_w, V_w = np.linalg.svd(watermark_img, full_matrices=True)

# پارامترها
alpha = 0.2  
M = len(S_h)  

# تنظیم k برای روش مقیاس‌بندی‌شده (برای تولید نویز مشابه تصویر (c))
k_scaled = 150  # افزایش دادم تا نویز شطرنجی بیشتری ایجاد بشه
S_m1 = S_h.copy()
for i in range(M - k_scaled, M):
    if i < len(S_w):
        S_m1[i] = S_h[i] + alpha * S_w[i]  # جمع مستقیم

# بازسازی تصویر علامت‌گذاری‌شده (روش 1 - تصویر (c))
watermarked_img1 = U_h @ np.diag(S_m1) @ V_h.T
watermarked_img1 = np.clip(watermarked_img1 * 255, 0, 255).astype(np.uint8)

# تنظیم k برای روش لگاریتمی (برای تولید کیفیت بالا مشابه تصویر (d))
k_log = 10  # کاهش دادم تا کیفیت بالا بمونه و نویز کم بشه
S_m2 = S_h.copy()
for i in range(M - k_log, M):
    if i < len(S_w):
        S_m2[i] = S_h[i] + alpha * np.log1p(S_w[i] + 1e-10)  # جمع با لگاریتم

# بازسازی تصویر علامت‌گذاری‌شده (روش 2 - تصویر (d))
watermarked_img2 = U_h @ np.diag(S_m2) @ V_h.T
watermarked_img2 = np.clip(watermarked_img2 * 255, 0, 255).astype(np.uint8)

# محاسبه NMSE برای هر دو روش
nmse1 = calculate_nmse(host_img, watermarked_img1 / 255.0)
nmse2 = calculate_nmse(host_img, watermarked_img2 / 255.0)

# نمایش تصاویر
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (b) علامت (چشم هوروس)
axes[0].imshow(watermark_img, cmap='gray')
axes[0].set_title('(b) Watermark')
axes[0].axis('off')

# (c) تصویر علامت‌گذاری‌شده با جمع مقیاس‌بندی‌شده (قایق‌ها با نویز)
axes[1].imshow(watermarked_img1, cmap='gray')
axes[1].set_title(f'(c) Scaled Addition\nNMSE: {nmse1:.2e}')
axes[1].axis('off')

# (d) تصویر علامت‌گذاری‌شده با جمع لگاریتمی (قایق‌ها با کیفیت بالا)
axes[2].imshow(watermarked_img2, cmap='gray')
axes[2].set_title(f'(d) Log Scaled Addition\nNMSE: {nmse2:.2e}')
axes[2].axis('off')

plt.suptitle('Figure 8: Effect of Logarithmic Transformation on SVs Range')
plt.show()

# ذخیره تصاویر
img = (watermark_img * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(img).save('watermark_eye_of_horus.jpg')
Image.fromarray(watermarked_img1).save('watermarked_scaled.jpg')
Image.fromarray(watermarked_img2).save('watermarked_log_scaled.jpg')