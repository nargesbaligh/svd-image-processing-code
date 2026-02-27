#Fig. 9:
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#توابع کمکی
def imread_gray(path, size=None):
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float64)

def clip_uint8(x):
    return np.uint8(np.clip(x, 0, 255))

def svd2(a):
    # full_matrices= باعث می‌شود ابعاد ماتریس‌ها فشرده بماند : a (m,n) -> U(m,r), s(r,), Vt(r,n)
    U, s, Vt = np.linalg.svd(a, full_matrices=False)
    return U, s, Vt


#روش های جاسازی  (Fig 9c, 9d)
def embed_scaled_SVs(host, wm, alpha=0.2, k=64):
    
    #Fig 9(c)-روش : Sm[i] = Sh[i] + alpha * Sw[q]

    Uh, sh, Vth = svd2(host)
    Uw, sw, Vtw = svd2(wm)

    r = min(len(sh), len(sw))
    k = int(min(k, r))

    sm = sh.copy()
    tail_start = r - k  # متناظر با ناحیه (M-k < i < M) در مقاله
    sm[tail_start:r] = sh[tail_start:r] + alpha * sw[0:k]

    Y = Uh @ np.diag(sm) @ Vth
    return Y, (Uh, sh, Vth, Uw, sw, Vtw, alpha, k)

def embed_log_scaled_SVs(host, wm, alpha=0.2, k=64, eps=1e-12):
    """
    Fig 9(d)-روش (paper's idea): کاهش دامنه مقادیر تکین واترمارک با لگاریتم قبل از جمع کردن:
      Sm[i] = Sh[i] + alpha * log(Sw[q])
    """
    Uh, sh, Vth = svd2(host)
    Uw, sw, Vtw = svd2(wm)

    r = min(len(sh), len(sw))
    k = int(min(k, r))

    sm = sh.copy()
    tail_start = r - k
    sm[tail_start:r] = sh[tail_start:r] + alpha * np.log(sw[0:k] + eps)

    Y = Uh @ np.diag(sm) @ Vth
    return Y, (Uh, sh, Vth, Uw, sw, Vtw, alpha, k, eps)


# اختیاری: استخراج non-blind بر اساس ایده معادله 5  
def extract_log_watermark(host, watermarked, Uw, Vtw, alpha, k, eps=1e-12):
    """
     استخراج  Non-blind  برای حالت جاسازی لگاریتمی:
      s'_w[q] = exp((Sm[i]-Sh[i]) / alpha)
    بازسازی W' = Uw * diag(s'_w) * Vw^T  (truncated)
    """
    Uh, sh, Vth = svd2(host)
    Um, sm, Vtm = svd2(watermarked)

    r = min(len(sh), len(sm))
    k = int(min(k, r))
    tail_start = r - k

    sw_hat = np.exp((sm[tail_start:r] - sh[tail_start:r]) / alpha)
    #بازسازی واترمارک فقط با k مؤلفه
    Wrec = Uw[:, :k] @ np.diag(sw_hat) @ Vtw[:k, :]
    return Wrec


# اجرای نمونه برای ساخت پنل ۲×۲ مشابه شکل ۹
if __name__ == "__main__":
    
    host_path = r"C:\Users\Asus\Desktop\imagess\host.png"
    wm_path   = r"C:\Users\Asus\Desktop\imagess\watermark.png"  

    host = imread_gray(host_path)
    wm   = imread_gray(wm_path, size=(host.shape[1], host.shape[0])) 

    alpha = 0.2
    k = 64  #بسته به اندازه تصویر  32/64/128 را امتحان می کنیم

    Y_scaled, _ = embed_scaled_SVs(host, wm, alpha=alpha, k=k)         # Fig 9(c)
    Y_log, _    = embed_log_scaled_SVs(host, wm, alpha=alpha, k=k)     # Fig 9(d)

    # رسم
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    ax[0, 0].imshow(clip_uint8(host), cmap="gray")
    ax[0, 0].set_title("(a) Original"); ax[0, 0].axis("off")

    ax[0, 1].imshow(clip_uint8(wm), cmap="gray")
    ax[0, 1].set_title("(b) Watermark"); ax[0, 1].axis("off")

    ax[1, 0].imshow(clip_uint8(Y_scaled), cmap="gray")
    ax[1, 0].set_title("(c) Scaled addition of watermark SVs"); ax[1, 0].axis("off")

    ax[1, 1].imshow(clip_uint8(Y_log), cmap="gray")
    ax[1, 1].set_title("(d) Scaled addition of log(watermark SVs)"); ax[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    #ذخیره خروجی ها 
    Image.fromarray(clip_uint8(Y_scaled)).save("watermarked_scaled.png")
    Image.fromarray(clip_uint8(Y_log)).save("watermarked_log.png")