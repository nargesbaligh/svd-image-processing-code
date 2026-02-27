import numpy as np
import matplotlib.pyplot as plt

# figure13
# تنظیمات برای شبیه‌سازی داده‌ها
np.random.seed(42)  # برای تکرارپذیری
n = 300  # تعداد مقادیر تکین

# شبیه‌سازی مقادیر تکین برای تصویر اصلی (Image 13)
# از 10^0 تا 10^-10 کاهش می‌یابد
svs_original = np.logspace(0, -10, n)  # کاهش لگاریتمی از 10^0 تا 10^-10
svs_original += np.random.normal(0, 0.2 * svs_original, n)  # نویز بیشتر

# شبیه‌سازی مقادیر تکین برای تصویر صاف‌شده (Smoothed Version)
svs_smoothed = np.logspace(0, -10, n) * 0.95  # مقادیر کمی کوچک‌تر از تصویر اصلی
# اضافه کردن افت شدید در انتهای خط آبی تا 10^-20
for i in range(250, n):  # از شاخص 250 به بعد افت شدید
    svs_smoothed[i] *= (n - i) / (n - 250) * 1e-10  # کاهش شدید به سمت 10^-20
svs_smoothed += np.random.normal(0, 0.25 * svs_smoothed, n)  # نویز بیشتر برای خط آبی

# رسم نمودار
plt.figure(figsize=(8, 6))
plt.semilogy(range(n), svs_original, 'r-', label='Image 13')
plt.semilogy(range(n), svs_smoothed, 'b--', label='Smoothed Version')
plt.grid(True, which="both", ls="--")
plt.xlabel('Index')
plt.ylabel('SVs')
plt.title('Figure 13. LPF Effect on SVs of an Image and its Smoothed Version')
plt.legend()

# تنظیم محدوده محور y از 10^-20 تا 10^10
plt.ylim(1e-20, 1e10)

plt.show()
###############################################
#figure14
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from PIL import Image

# تنظیمات اولیه
image_size = (256, 256)

# خواندن تصویر اصلی (MRI مغز)
image_path_brain = "C:/Users/Asus/Desktop/mri.png"  # مسیر تصویر خودتون رو وارد کنید
brain_image = Image.open(image_path_brain).convert('L')  # تبدیل به خاکستری
brain_image = np.array(brain_image)  # تبدیل به آرایه numpy

# تغییر اندازه تصویر به 256x256 (در صورت نیاز)
if brain_image.shape != (256, 256):
    brain_image = np.array(Image.fromarray(brain_image).resize((256, 256)))

# نرمال‌سازی تصویر به محدوده 0-1
brain_image = brain_image / 255.0

# شبیه‌سازی تصاویر خطوط افقی و عمودی
horizontal_lines = np.zeros(image_size)
horizontal_lines[::10, :] = 1  # خطوط افقی با فاصله 10 پیکسل

vertical_lines = np.zeros(image_size)
vertical_lines[:, ::10] = 1  # خطوط عمودی با فاصله 10 پیکسل

# نرمال‌سازی تصاویر شبیه‌سازی‌شده
horizontal_lines = horizontal_lines / horizontal_lines.max()
vertical_lines = vertical_lines / vertical_lines.max()

# محاسبه SVD برای هر تصویر
def compute_svd(image):
    U, S, Vt = svd(image, full_matrices=False)
    return U, S, Vt

# SVD برای هر تصویر
U_brain, S_brain, Vt_brain = compute_svd(brain_image)
U_horizontal, S_horizontal, Vt_horizontal = compute_svd(horizontal_lines)
U_vertical, S_vertical, Vt_vertical = compute_svd(vertical_lines)

# تنظیمات برای رسم
fig = plt.figure(figsize=(15, 10))

# (a) تصویر مغز
plt.subplot(3, 4, 1)
plt.imshow(brain_image, cmap='gray')
plt.title('(a) Brain')
plt.axis('off')

# (b) خطوط افقی
plt.subplot(3, 4, 2)
plt.imshow(horizontal_lines, cmap='gray')
plt.title('(b) Horizontal')
plt.axis('off')

# (c) خطوط عمودی
plt.subplot(3, 4, 3)
plt.imshow(vertical_lines, cmap='gray')
plt.title('(c) Vertical')
plt.axis('off')

# (d) نمودار مقادیر تکین
plt.subplot(3, 4, 4)
plt.semilogy(S_brain, label='Brain')
plt.semilogy(S_horizontal, label='Horizontal')
plt.semilogy(S_vertical, label='Vertical')
plt.grid(True, which="both", ls="--")
plt.title('(d) SVs')
plt.legend()

# (e-g) بردارهای Vt (بردارهای تکین راست)
plt.subplot(3, 4, 5)
plt.imshow(Vt_brain[0, :].reshape(16, 16), cmap='gray')  # بردار اول Vt برای مغز
plt.title('(e) V Brain')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(Vt_horizontal[0, :].reshape(16, 16), cmap='gray')  # بردار اول Vt برای خطوط افقی
plt.title('(f) V Horizontal')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(Vt_vertical[0, :].reshape(16, 16), cmap='gray')  # بردار اول Vt برای خطوط عمودی
plt.title('(g) V Vertical')
plt.axis('off')

# (h-j) بردارهای U (بردارهای تکین چپ)
plt.subplot(3, 4, 9)
plt.imshow(U_brain[:, 0].reshape(16, 16), cmap='gray')  # بردار اول U برای مغز
plt.title('(h) U Brain')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(U_horizontal[:, 0].reshape(16, 16), cmap='gray')  # بردار اول U برای خطوط افقی
plt.title('(i) U Horizontal')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(U_vertical[:, 0].reshape(16, 16), cmap='gray')  # بردار اول U برای خطوط عمودی
plt.title('(j) U Vertical')
plt.axis('off')

plt.tight_layout()
plt.suptitle('Figure 14. SVD Orientation', y=1.05)
plt.show()