import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# تصویر با استفاده از PIL بارگذاری می شود

#Figure 3# بارگذاری تصویر و تبدیل آن به حالت خاکستری
# تصویر به مقیاس خاکستری تبدیل می‌شود تا کار با آن ساده‌تر باشد (یک ماتریس دوبعدی به جای سه‌بعدی RGB)
image_path = "C:/Users/Asus/Desktop/images.jpg" 
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# انجام تجزیهٔ SVD
U, S, V = np.linalg.svd(image_array, full_matrices=True)

# تنظیم شکل برای نمایش سه تصویر
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# نمایش تصویر اصلی
axes[0].imshow(image_array, cmap='gray')
axes[0].set_title('(a) Original Image')
axes[0].axis('off')

# نمایش ماتریس U (بردارهای تکینِ چپ)
axes[1].imshow(U, cmap='gray', aspect='auto')
axes[1].set_title('(b) Left SCs; U')
axes[1].axis('off')

# نمایش ماتریس V (بردارهای تکینِ راست)
axes[2].imshow(V, cmap='gray', aspect='auto')
axes[2].set_title('(c) Right SCs; V')
axes[2].axis('off')
# تنظیم عنوان کلی شکل
plt.suptitle('Figure 3. 2D Representation of SCs', fontsize=16)

plt.tight_layout()
plt.show()

####################################################################################
# بارگذاری تصویر و تبدیل آن به حالت خاکستری
image_path = "C:/Users/Asus/Desktop/Figure5.png" 
image = Image.open(image_path)
image = image.resize((200, 200)) 
image = image.convert('L')  
image_array = np.array(image)

# انجام تجزیهٔ SVD
U, S, V = np.linalg.svd(image_array, full_matrices=True)

# بازسازی تصویر با استفاده از k=30 مقدار تکینِ اول (زیر‌فضای دادهٔ تصویر / تصویر نویززدایی‌شده)
k = 30 
S_k = np.zeros_like(image_array, dtype=float)
S_k[:k, :k] = np.diag(S[:k])  
image_data_subspace = np.dot(U[:, :k], np.dot(S_k[:k, :k], V[:k, :]))

# محاسبهٔ زیر‌فضای نویز (نویز باقی‌مانده)
# زیرفضای نویز با کم کردن تصویر بازسازی‌شده (بدون نویز) از تصویر اصلی محاسبه می‌شود
noise_subspace = image_array - image_data_subspace

# برای نمایش بهتر، زیرفضای نویز نرمال‌سازی می‌شود تا مقادیر آن بین 0 و 1 قرار گیرند:
# نرمال‌سازی باعث می‌شود که زیرفضای نویز به‌خوبی قابل‌مشاهده باشد. بدون نرمال‌سازی، ممکن است این تصویر خیلی کم‌رنگ یا نامشخص به نظر برسد
# نرمال‌سازی زیر‌فضای نویز برای نمایش بهتر
noise_subspace = (noise_subspace - noise_subspace.min()) / (noise_subspace.max() - noise_subspace.min())

# تنظیم شکل برای نمایش سه تصویر
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# نمایش تصویر اصلیِ نویزی (MRI)
axes[0].imshow(image_array, cmap='gray')
axes[0].set_title('(a) Original Noisy MRI Image')
axes[0].axis('off')

# نمایش زیر‌فضای دادهٔ تصویر (تصویر نویززدایی‌شده)
axes[1].imshow(image_data_subspace, cmap='gray')
axes[1].set_title('(b) Image Data Subspace')
axes[1].axis('off')

# نمایش زیر‌فضای نویز
axes[2].imshow(noise_subspace, cmap='gray')
axes[2].set_title('(c) Noise Subspace')
axes[2].axis('off')

# تنظیم عنوان کلی شکل
plt.suptitle('Figure 5. SVD Denoising', fontsize=16)

# ذخیره‌ی شکل (اختیاری)
plt.savefig("figure5_svd_denoising.png", dpi=300, bbox_inches='tight')

# نمایش شکل
plt.tight_layout()
plt.show()



