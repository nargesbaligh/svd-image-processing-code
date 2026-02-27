import numpy as np
import matplotlib.pyplot as plt
#figure4
# تنظیمات اولیه برای تولید داده‌های نمونه
np.random.seed(42)  # برای تکرارپذیری
slices = np.arange(0, 31)  # محور افقی: 0 تا 30
correlation = np.random.uniform(-0.05, 0.05, size=31)  # مقادیر تصادفی کوچک برای همبستگی
correlation[0] = 0.85  # یک نقطه برجسته در اسلایس 0 با همبستگی بالا

# رسم نمودار پراکندگی
plt.figure(figsize=(8, 6))
# رسم خطوط برای اتصال نقاط
plt.plot(slices, correlation, color='black', linestyle='-', linewidth=1)
# رسم نقاط به شکل ستاره
plt.scatter(slices, correlation, marker='*', color='black', s=50)

# تنظیمات محورها
plt.xlabel('Slices')
plt.ylabel('Correlation')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(-0.1, 0.9)  # محدوده محور عمودی
plt.xticks(np.arange(0, 31, 5))  # تنظیمات محور افقی
plt.title('Figure 4. Correlation is carried out between different subspaces (slices)')

# نمایش نمودار
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# فرض کنید که تصویر شما قبلاً به یک ماتریس SVD تبدیل شده باشد
# در اینجا، به عنوان مثال یک ماتریس تصادفی برای SVD می‌سازیم
np.random.seed(42)
image_matrix = np.random.rand(30, 30)  # ماتریس تصادفی 30x30

# تجزیه SVD
U, S, Vt = np.linalg.svd(image_matrix)

# فرض کنید که می‌خواهیم همبستگی را بین برش‌های مختلف از U یا V محاسبه کنیم
correlation = []

# محاسبه همبستگی بین برش‌های مختلف
for i in range(30):
    for j in range(i + 1, 30):
        corr = np.corrcoef(U[:, i], U[:, j])[0, 1]  # همبستگی بین برش‌ها
        correlation.append(corr)

# رسم همبستگی
plt.figure(figsize=(8, 6))
plt.plot(range(len(correlation)), correlation, marker='*')
plt.xlabel('Slices')
plt.ylabel('Correlation')
plt.title('Correlation is carried out between different subspaces (slices)')
plt.grid(True)
plt.show()
