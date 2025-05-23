import numpy as np
import random

# Create data augmentation and save it to folder, the same training data 
# will be run multiple time to create different data augmentation 
# which is used as a training data for fine tuning 

# ─── CONFIG ─────────────────────────────────────────────────────────────
N_AUG_PER_SAMPLE = 3
MIX_ALPHA        = 0.4
CUTOUT_HOLES     = 1
CUTOUT_LENGTH    = 4   # along the 16-dim axis

# each has 25% to occur (mixup, cutmix, cutout, none)
p_mixup   = 0.25
p_cutmix  = 0.50  # cumulative
p_cutout  = 0.75  # cumulative
# [0.75,1.0) -> none

# ─── HELPERS ─────────────────────────────────────────────────────────────
def mixup(x1, y1, x2, y2, alpha):
    lam = np.random.beta(alpha, alpha) if alpha>0 else 1.0
    return lam*x1 + (1-lam)*x2, lam*y1 + (1-lam)*y2

def rand_bbox(L, lam):
    cut_len = int(L * np.sqrt(1. - lam))
    start = random.randint(0, L-cut_len)
    return start, start+cut_len

def cutmix(x1, y1, x2, y2, alpha):
    lam = np.random.beta(alpha, alpha) if alpha>0 else 1.0
    L = x1.shape[-1]
    x1_new = x1.copy()
    i1, i2 = rand_bbox(L, lam)
    x1_new[:, i1:i2] = x2[:, i1:i2]
    # adjust lam to exact ratio
    lam_adj = 1 - (i2 - i1)/L
    return x1_new, lam_adj*y1 + (1-lam_adj)*y2

def cutout(x, n_holes, length):
    C, L = x.shape
    x_new = x.copy()
    for _ in range(n_holes):
        i = random.randint(0, L-length)
        x_new[:, i:i+length] = 0
    return x_new

# ─── AUGMENTATION LOOP ───────────────────────────────────────────────────
def augment_and_save(X, y, out_path="augmented_features.npz"):
    N, C, L = X.shape
    X_aug, y_aug = [], []

    for i in range(N):
        xi, yi = X[i], y[i]
        # always include original
        # X_aug.append(xi)
        # y_aug.append(yi)

        r = random.random()
        if r < p_mixup:
            j = random.randrange(N)
            x_new, y_new = mixup(xi, yi, X[j], y[j], MIX_ALPHA)

        elif r < p_cutmix:
            j = random.randrange(N)
            x_new, y_new = cutmix(xi, yi, X[j], y[j], MIX_ALPHA)

        elif r < p_cutout:
            x_new = cutout(xi, CUTOUT_HOLES, CUTOUT_LENGTH)
            y_new = yi

        else:
            x_new, y_new = xi, yi

        X_aug.append(x_new)
        y_aug.append(y_new)

    # to arrays
    X_aug = np.stack(X_aug, axis=0)  # shape (N*(1+N_AUG), C, L)
    y_aug = np.array(y_aug, dtype=float)

    print(X_aug.shape)
    print(y_aug.shape)


    # save to disk
    np.savez_compressed(out_path, x=X_aug, y=y_aug)
    print(f"Saved {X_aug.shape[0]} samples -> '{out_path}'")

# ─── EXAMPLE USAGE ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # suppose X and y are already loaded:
    #    X: np.ndarray of shape (num_samples, 256, 16)
    #    y: np.ndarray of shape (num_samples,)
    #
    # e.g.:

    data = np.load("extracted_features.npz")
    # X = np.load("features.npy")  
    # y = np.load("labels.npy")
    X = data['x']
    y = data['y']
    print(X.shape)
    print(y.shape)
    augment_and_save(X, y, out_path="train_augmented.npz")