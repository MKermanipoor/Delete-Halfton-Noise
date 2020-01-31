import cv2
import numpy as np
import matplotlib.pyplot as plt


def cast_int(t):
    return [int(i) for i in t]


img1 = cv2.imread('data/Halftone/1_4.bmp')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('data/Attack 1/1_4.bmp')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

orb = cv2.ORB_create()

# %%
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# %%
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# %%
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:3], None, flags=2)
plt.imshow(img3)
plt.show()



# %%
pts1 = []
pts2 = []
for i in range(len(matches)):
    # if len(pts1) >= 3:
    #     break
    #
    # if len(pts1) == 0:
    #     pts1.append(cast_int(kp1[matches[i].queryIdx].pt))
    #     pts2.append(cast_int(kp2[matches[i].trainIdx].pt))
    #     continue
    #
    # t = True
    # for point in pts1:


    pts1.append(cast_int(kp1[matches[i].queryIdx].pt))
    pts2.append(cast_int(kp2[matches[i].trainIdx].pt))

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
#%%
#M = cv2.getAffineTransform(pts2, pts1)
M, mask = cv2.findHomography(pts2, pts1,cv2.RANSAC,5.0)
result = cv2.warpPerspective(img2, M, (512, 512))
plt.imshow(result)
plt.show()
