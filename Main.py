import cv2
import numpy as np
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt


# usefull function for cast tuple of float to numpy array of integer
def cast_int(t):
    return np.array([int(i) for i in t])


def get_sift_matrix(img1, img2):
    # initialize sift detector
    sift = cv2.xfeatures2d.SIFT_create()

    # detect and describe halftone and attack 2 images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # match key points of 2 images
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=40)
    search_params = dict(checks=75)
    bf = cv2.FlannBasedMatcher(index_params, search_params)
    matches = bf.knnMatch(des1, des2, k=2)

    # select best points
    pts1 = []
    pts2 = []
    goods = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            goods.append(m)
            pts1.append(cast_int(kp1[m.queryIdx].pt))
            pts2.append(cast_int(kp2[m.trainIdx].pt))
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # calculate affine matrix
    M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    # if you want to see top 10 of match points uncomment these lines
    # goods = sorted(goods, key=lambda x: x.distance)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, goods[:10], None, flags=2)
    # plt.imshow(img3)
    # plt.show()
    return M, len(goods)


# sum of metrics
s_ssim = 0

# for all images
for i in range(1, 11):
    for j in range(1, 6):
        # read halftone image
        image1 = cv2.imread('data/Halftone/' + str(i) + '_' + str(j) + '.bmp')
        # convert to rgb
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        # read attack 1 image
        image2 = cv2.imread('data/Attack 1/' + str(i) + '_' + str(j) + '.bmp')
        # convert to rgb
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        # get affine matrix and match point number
        # base on halftone image and attack 1 image
        Matrix, match_size = get_sift_matrix(image1, image2)

        # read attack 2 image
        image3 = cv2.imread('data/Attack 2/' + str(i) + '_' + str(j) + '.bmp')
        # apply affine matrix
        image3 = cv2.warpPerspective(image3, Matrix, (512, 512))

        # read original image
        original = cv2.imread('data/Original/' + str(i) + '.bmp')

        # compute ssim
        (score, diff) = compare_ssim(image3, original, full=True, multichannel=True)
        s_ssim += score

        # compute mse
        sub = np.subtract(original, image3)
        mse = np.square(sub).mean()

        # print metric
        print(str(i) + '_' + str(j) + ' : ' + str(score) + ' ' + str(mse) + ' ' + str(match_size))

        cv2.imwrite('data/Result/' + str(i) + '_' + str(j) + '.png', image3)

        # if you want to see result of affine on attack 2 image uncomment this 3 lines
        # image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        # plt.imshow(image3)
        # plt.show()
# print mean of ssim
print('mean SSIM : ' + str(s_ssim / 50))
