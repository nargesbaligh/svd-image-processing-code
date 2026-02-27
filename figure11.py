# Fig. 11:
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def imread_gray(path, size=None):
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.float64)


def clip_uint8(x):
    return np.uint8(np.clip(x, 0, 255))


def svd_compact(a):
    # a: (m,n) -> U(m,r), s(r,), Vt(r,n), r=min(m,n)
    U, s, Vt = np.linalg.svd(a, full_matrices=False)
    return U, s, Vt


def reconstruct_with_svs(base_img, donor_img, k=30):
    Ub, sb, Vtb = svd_compact(base_img)
    Ud, sd, Vtd = svd_compact(donor_img)

    r = min(len(sb), len(sd))
    k = int(min(k, r))

    # Truncate
    Ub_k = Ub[:, :k]
    Vtb_k = Vtb[:k, :]
    sd_k = sd[:k]

    Y = Ub_k @ np.diag(sd_k) @ Vtb_k
    return Y


if __name__ == "__main__":
     #مسبرها
    base_path = r"C:\Users\Asus\Desktop\imagess\lena.png"    # image whose U,V are used (geometry)
    sv_donor1_path = r"C:\Users\Asus\Desktop\imagess\sv_donor1_path.png"  # provides SVs for panel (b)
    sv_donor2_path = r"C:\Users\Asus\Desktop\imagess\sv_donor2_path.png"  # provides SVs for panel (c)

    k = 30
    # کامنت انگلیسی فهم بهتری برای این چند خط کد دارد
    # Read base first, then resize donors to match base size
    base = imread_gray(base_path)
    H, W = base.shape
    donor1 = imread_gray(sv_donor1_path, size=(W, H))
    donor2 = imread_gray(sv_donor2_path, size=(W, H))

    # (a) Using its own SVs (i.e., normal truncated SVD)
    rec_a = reconstruct_with_svs(base, base, k=k)

    # (b) Use SVs from donor1, but keep base geometry
    rec_b = reconstruct_with_svs(base, donor1, k=k)

    # (c) Use SVs from donor2, but keep base geometry
    rec_c = reconstruct_with_svs(base, donor2, k=k)

    # رسم 
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(clip_uint8(rec_a), cmap="gray")
    ax[0].set_title("(a) SVs from itself"); ax[0].axis("off")

    ax[1].imshow(clip_uint8(rec_b), cmap="gray")
    ax[1].set_title("(b) SVs from donor1"); ax[1].axis("off")

    ax[2].imshow(clip_uint8(rec_c), cmap="gray")
    ax[2].set_title("(c) SVs from donor2"); ax[2].axis("off")

    plt.tight_layout()
    plt.show()

    # ذخیره سازی
    Image.fromarray(clip_uint8(rec_a)).save("fig11_a_selfSV.png")
    Image.fromarray(clip_uint8(rec_b)).save("fig11_b_donor1SV.png")
    Image.fromarray(clip_uint8(rec_c)).save("fig11_c_donor2SV.png")