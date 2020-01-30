import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np


def show(img):
    band = np.shape(np.shape(img))[0]
    if band == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def wavelet(img):
    LL, (LH, HL, HH) = pywt.dwt2(img, 'bior1.3')
    return [LL, LH, HL, HH]


def inverse_wavelet(LL, LH, HL, HH):
    return pywt.idwt2((LL, (LH, HL, HH)), 'bior1.3')




# %%
image = cv2.imread('data/Halftone/2_1.bmp')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show(image)
# %%
a = wavelet(image)
# %%

# %%

