from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import opening, disk, dilation, erosion, closing
import glob


def grays(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def white_image(im):
    return cv2.bitwise_not(np.zeros(im.shape, np.uint8))


images = glob.glob('*.jpg')

if not images:
    print("No .jpg images found in current directory. Please place some test images here.")
    exit(1)

for fname in images:
    image = cv2.imread(fname)
    if image is None:
        print(f"Could not read image {fname}")
        continue
        
    print(f"Processing image: {fname}")
    
    wh = white_image(image)
    wh1 = white_image(image)
    wh2 = white_image(image)
    wh3 = white_image(image)
    wh_gray = grays(wh)
    wh_gray1 = grays(wh)
    wh_gray3 = grays(wh)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, sigmaColor=7, sigmaSpace=5)

    kernel_sharp = np.array((
        [-2, -2, -2],
        [-2, 17, -2],
        [-2, -2, -2]), dtype='int')
    im = cv2.filter2D(blur, -1, kernel_sharp)

    edged = cv2.Canny(im, 180, 240)
    thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    tnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    cv2.drawContours(wh1, cnts, -1, 255, 1)
    cv2.drawContours(wh3, tnts, -1, 255, 1)

    o = []
    p = []
    for cnt in tnts:
        counter = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []
        if cv2.contourArea(cnt) > 5:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counter += 1
                    o.append(x)
                    p.append(y)
                    pts.append((x, y))
        if counter > 10:
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(wh_gray, [pts], True, 0)

    for cnt in cnts:
        counters = 0
        cnt = np.array(cnt)
        cnt = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
        pts = []
        print(f"Contour area: {cv2.contourArea(cnt)}")
        if cv2.contourArea(cnt) > 10:
            for i in cnt:
                x, y = i
                if edged[y, x] == 255:
                    counters += 1
                    o.append(x)
                    p.append(y)
                    pts.append((x, y))

        if counters > 10:
            pts = np.array(pts)
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(wh_gray1, [pts], True, 0)

    plt.figure()
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    res = closing(cv2.bitwise_not(wh_gray), selem=disk(1))
    wes = closing(cv2.bitwise_not(wh_gray1), selem=disk(1))

    plt.figure()
    plt.title('Threshold Contours')
    plt.imshow(wh_gray, cmap='gray')
    
    plt.figure()
    plt.title('Edge Contours')
    plt.imshow(cv2.bitwise_not(wh_gray1), cmap='gray')

    dst = cv2.bitwise_and(wh_gray, wh_gray1)
    plt.figure()
    plt.title('Combined Result')
    plt.imshow(cv2.bitwise_not(dst), cmap='gray')
    plt.show()

# plt.close()
