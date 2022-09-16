# ARTag_detection.py
"""
Author: Adarsh Malapaka (amalapak@umd.edu), 2022
Brief: Detects an AR Tag from a single video frame using Fast Fourier Transform 
       based edge detection, Corner Detection and other image processing functions.

Course: ENPM673 - Perception for Autonomous Robotics [Proj-01, Question 1 (a)]
Date: 7th March, 2022

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift

def segment_edges(img):

    img_fft = fft2(img)
    img_fft_mag = 20*np.log(1+np.abs(img_fft))
    img_fft_shift = fftshift(img_fft)
    plt.subplot(2, 2, 1), plt.imshow(img_fft_mag, cmap='gray')
    plt.title('FFT Magnitude (Log) Plot')

    img_fft_shift_mag = 20*np.log(1+np.abs(img_fft_shift))
    plt.subplot(2, 2, 2), plt.imshow(img_fft_shift_mag, cmap='gray')
    plt.title('Shifted FFT Magnitude (Log) Plot')

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.ones((rows, cols), np.uint8)
    r = 180
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    img_fft_mask = img_fft_shift * mask
    img_fft_mask_mag = 20*np.log(1+np.abs(img_fft_mask))
    plt.subplot(2, 2, 3), plt.imshow(img_fft_mask_mag, cmap='gray')
    plt.title('Masked FFT Plot')

    img_fft_mask_ishift = ifftshift(img_fft_mask)
    img_ifft = np.uint8(np.abs(ifft2(img_fft_mask_ishift)))
    plt.subplot(2, 2, 4), plt.imshow(img_ifft, cmap='gray')
    plt.title('Inverse FFT Plot')

    plt.show()

    return img_ifft


def extract_corners(img_ifft):
    
    img_detected_corners = img.copy()
    corners = cv2.goodFeaturesToTrack(img_ifft, 15, 0.01, 50, useHarrisDetector=True, k=0.04)
    corners = np.int0(corners)
    corners = corners.reshape(corners.shape[0], corners.shape[1]*corners.shape[2])

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(img_detected_corners,(x,y),8,(255,0,0),-1)

    cv2.imshow("Detected Corners using Harris Corner Detection", img_detected_corners)
    cv2.waitKey(0)

    return corners
    

def detect_ARTag(corners):
 
    x_min_i = np.argmin(corners[:,0])
    x_max_i = np.argmax(corners[:,0])
    y_min_i = np.argmin(corners[:,1])
    y_max_i = np.argmax(corners[:,1])
    
    # Deleting the white background corners using 1st Min-Max values
    corners = np.delete(corners, (x_min_i,x_max_i,y_min_i,y_max_i), 0)

    x_vals = corners[:,0]
    y_vals = corners[:,1]

    x_min = np.min(x_vals)
    y_x_min = corners[np.argmin(x_vals)][1]

    x_max = np.max(x_vals)
    y_x_max = corners[np.argmax(x_vals)][1]

    y_min = np.min(y_vals)
    x_y_min = corners[np.argmin(y_vals)][0]

    y_max = np.max(y_vals)
    x_y_max = corners[np.argmax(y_vals)][0]

    # Obtaining the tag corners using 2nd Min-Max values
    tag_corners = np.vstack([[x_min, y_x_min]])
    tag_corners = np.vstack([tag_corners,[x_max, y_x_max]])
    tag_corners = np.vstack([tag_corners,[x_y_min, y_min]])
    tag_corners = np.vstack([tag_corners,[x_y_max, y_max]])

    return tag_corners


if __name__ == "__main__":

    img = cv2.imread('tag_single_frame.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_blur = cv2.medianBlur(img_gray, 3)
    _, img_thresh = cv2.threshold(img_blur, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    img_smooth = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_smooth = cv2.morphologyEx(img_smooth, cv2.MORPH_CLOSE, kernel)

    img_ifft = segment_edges(img_smooth)
    
    corners = extract_corners(img_ifft)

    tag_corners = detect_ARTag(corners)
    
    x_min, y_x_min = tag_corners[0]
    x_max, y_x_max = tag_corners[1]
    x_y_min, y_min = tag_corners[2]
    x_y_max, y_max = tag_corners[3]

    cv2.circle(img,(x_min, y_x_min),8,(255,255,0),4)
    cv2.circle(img,(x_max, y_x_max),8,(255,255,0),4)
    cv2.circle(img,(x_y_min, y_min),8,(255,255,0),4)
    cv2.circle(img,(x_y_max, y_max),8,(255,255,0),4)

    cv2.line(img, (x_min, y_x_min), (x_y_min, y_min), (0,0,255), 5)
    cv2.line(img, (x_y_min, y_min), (x_max, y_x_max), (0,0,255), 5)
    cv2.line(img, (x_max, y_x_max), (x_y_max, y_max), (0,0,255), 5)
    cv2.line(img, (x_y_max, y_max), (x_min, y_x_min), (0,0,255), 5)

    cv2.imshow("Detected AR Tag", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()