import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#figure 1
#تصویر به مقیاس خاکستری تبدیل می‌شود (convert('L')) تا فقط یک کانال رنگی داشته باشیم و فرآیند SVD ساده‌تر شود.
#تصویر با استفاده از کتابخانه PIL بارگذاری می‌شود.
# بارگذاری تصویر و تبدیل آن به حالت خاکستری
image_path = "C:/Users/Asus/Desktop/Figure1.jpg"  
image = Image.open(image_path)
image = image.resize((200, 200)) 
image = image.convert('L')  
image_array = np.array(image)

# انجام تجزیه SVD
U, S, V = np.linalg.svd(image_array, full_matrices=True)

# بازسازی تصویر با استفاده از k=20 مقدار تکینِ اول (SVD کوتاه‌شده / Truncated SVD)
k = 20
S_k = np.zeros_like(image_array, dtype=float)
S_k[:k, :k] = np.diag(S[:k]) 
truncated_svd_image = np.dot(U[:, :k], np.dot(S_k[:k, :k], V[:k, :]))

# تنظیم شکل برای نمایش دو تصویر
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#  نمایش تصویر اصلی
axes[0].imshow(image_array, cmap='gray')
axes[0].set_title('(a) Original')
#محورها غیرفعال می‌شوند تا فقط تصویر نمایش داده شود
axes[0].axis('off')

# نمایش تصویر بازسازی‌شده با SVD کوتاه‌شده
axes[1].imshow(truncated_svd_image, cmap='gray')
axes[1].set_title('(b) Truncated SVD')
axes[1].axis('off')

# تنظیم عنوان کلی شکل
plt.suptitle('Figure 1. Truncated SVD', fontsize=16)

#ذخیره کردن شکل
plt.savefig("figure1_truncated_svd.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

#####################################################################
#Figure 2
# بارگذاری تصویر و تبدیل آن به حالت خاکستری
image_path = "C:/Users/Asus/Desktop/picture2.jpg"
image = Image.open(image_path)
##تصویر به این اندازه پیکسل تغییر اندازه داده می‌شود تا محاسبات سریع‌تر و نمایش واضح‌تر باشد
image = image.resize((200, 200))
image = image.convert('L') 
image_array = np.array(image)

## انجام تجزیهٔ SVD
U, S, V = np.linalg.svd(image_array, full_matrices=True)

# بازسازی تصویر با استفاده از k=۳۰ مقدار تکینِ اول (زیر‌فضای دادهٔ تصویر)
k = 30  
S_k = np.zeros_like(image_array, dtype=float)
S_k[:k, :k] = np.diag(S[:k]) 
#تصویر بازسازی‌شده
image_data_subspace = np.dot(U[:, :k], np.dot(S_k[:k, :k], V[:k, :]))

# محاسبهٔ زیر‌فضای نویز (باقیمانده)
#زیرفضای نویز به صورت تفاوت بین تصویر اصلی و تصویر بازسازی‌شده محاسبه می‌شود
noise_subspace = image_array - image_data_subspace

# تنظیم شکل برای نمایش سه تصویر
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# نمایش تصویر اصلی
axes[0].imshow(image_array, cmap='gray')
axes[0].set_title('(a) Original Image')
axes[0].axis('off')

# نمایش زیر‌فضای دادهٔ تصویر (بازسازی‌شده با k=۳۰)
axes[1].imshow(image_data_subspace, cmap='gray')
axes[1].set_title('(b) Image Data Subspace')
axes[1].axis('off')

# نمایش زیر‌فضای نویز
axes[2].imshow(noise_subspace, cmap='gray')
axes[2].set_title('(c) Noise Subspace')
axes[2].axis('off')

plt.suptitle('Figure 2. SVD Subspaces', fontsize=16)

plt.tight_layout()
plt.show()

