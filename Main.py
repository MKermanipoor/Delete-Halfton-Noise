import cv2
import numpy as np
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt


def cast_int(t):
    return np.array([int(i) for i in t])


def get_matrix(img1, img2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:3], None, flags=2)
    # plt.imshow(img3)
    # plt.show()

    pts1 = []
    pts2 = []
    for i in range(len(matches)):
        pts1.append(cast_int(kp1[matches[i].queryIdx].pt))
        pts2.append(cast_int(kp2[matches[i].trainIdx].pt))

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
    return M, len(matches)
    # result = cv2.warpPerspective(img2, M, (512, 512))
    # plt.imshow(result)
    # plt.show()


def get_sift_matrix(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=40)
    search_params = dict(checks=75)
    bf = cv2.FlannBasedMatcher(index_params, search_params)
    matches = bf.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    goods = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            goods.append(m)
            pts1.append(cast_int(kp1[m.queryIdx].pt))
            pts2.append(cast_int(kp2[m.trainIdx].pt))

    # goods = sorted(goods, key=lambda x: x.distance)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, goods[:10], None, flags=2)
    # plt.imshow(img3)
    # plt.show()

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return M, len(goods)


s_ssim = 0
s_mse = 0
s_mp = 0

for i in range(1, 11):
    for j in range(1, 6):
        image1 = cv2.imread('data/Halftone/' + str(i) + '_' + str(j) + '.bmp')
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread('data/Attack 1/' + str(i) + '_' + str(j) + '.bmp')
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        Matrix, match_size = get_sift_matrix(image1, image2)

        image3 = cv2.imread('data/Attack 2/' + str(i) + '_' + str(j) + '.bmp')
        image3 = cv2.warpPerspective(image3, Matrix, (512, 512))
        # image3 = cv2.warpAffine(image3, Matrix, (512, 512))

        orginal = cv2.imread('data/Original/' + str(i) + '.bmp')

        (score, diff) = compare_ssim(image3, orginal, full=True, multichannel=True)
        s_ssim += score

        sub = np.subtract(orginal, image3)
        mse = np.square(sub).mean()
        s_mse += mse

        mp = len(sub[sub == 0])
        s_mp += mp

        print(str(i) + '_' + str(j) + ' : ' + str(score) + ' ' + str(mse) + ' ' + str(match_size))

        cv2.imwrite('data/Result/' + str(i) + '_' + str(j) + '.png', image3)

        # image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        # plt.imshow(image3)
        # plt.show()

print('mean SSIM : ' + str(s_ssim / 50))
print('mean MSE : ' + str(s_mse / 50))
print('mean MP : ' + str(s_mp / 50))
