import cv2
import numpy as np
import matplotlib.pyplot as plt


def cast_int(t):
    return np.array([int(i) for i in t])


img1 = cv2.imread('data/Halftone/2_2.bmp')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('data/Attack 1/2_2.bmp')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# %%
# orb = cv2.ORB_create(50, )
orb = cv2.xfeatures2d.SIFT_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# %%
# bf = cv2.BFMatcher()
bf = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=5))
matches = bf.knnMatch(des1, des2, k=2)
# matches = bf.match(des1, des2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
good = sorted(good, key=lambda x: x.distance)
# %%
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good[:10], None, flags=2)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
plt.imshow(img3)
plt.show()




#%%
pts1 = []
pts2 = []
for m in good:
# for m in matches:
    pts1.append(cast_int(kp1[m.queryIdx].pt))
    pts2.append(cast_int(kp2[m.trainIdx].pt))

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
# %%
# M = cv2.getAffineTransform(pts2, pts1)
M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
# %%
# result = cv2.warpAffine(img2, M, (512, 512))
result = cv2.warpPerspective(img2, M, (512, 512))
plt.imshow(result)
plt.show()


# %%
pts1 = []
pts2 = []
# for m in good:
for m in matches:
    if len(pts1) >= 3:
        break

    if len(pts1) == 0:
        pts1.append(cast_int(kp1[m.queryIdx].pt))
        pts2.append(cast_int(kp2[m.trainIdx].pt))
        continue

    t = True
    for p in pts1:
        if np.linalg.norm(cast_int(kp1[m.queryIdx].pt) - p) < 30:
            t = False
            break

    if t:
        pts1.append(cast_int(kp1[m.queryIdx].pt))
        pts2.append(cast_int(kp2[m.trainIdx].pt))

features = img1
for p in pts1:
    features = cv2.circle(features, (p[0], p[1]), 10, (255, 0, 0))

plt.imshow(features)
plt.show()