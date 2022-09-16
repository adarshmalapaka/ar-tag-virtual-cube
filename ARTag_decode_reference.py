# ARTag_decode_reference.py
"""
Author: Adarsh Malapaka (amalapak@umd.edu), 2022
Brief: Detects the given reference AR Tag and decodes the upright orientation 
       and tag ID information.

Course: ENPM673 - Perception for Autonomous Robotics [Proj-01, Question 1 (b)]
Date: 7th March, 2022

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from numpy import linalg as la
import imutils


def segment_edges(im):
    
    img = im.copy() 
    img_fft = fft2(img)
    
    # Shifting the FFT plot to get low frequency components in the center
    img_fft_shift = fftshift(img_fft)    

    rows, cols = im.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.ones((rows, cols), np.uint8)
    r = 220    # Mask Radius
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r    # Circular mask to filter out low frequencies
    mask[mask_area] = 0

    img_fft_mask = img_fft_shift * mask

    img_fft_mask_ishift = ifftshift(img_fft_mask)
    img_ifft = np.uint8(np.abs(ifft2(img_fft_mask_ishift)))

    return img_ifft



def extract_corners(img_ifft):
    
    # Harris Corner detection
    corners = cv2.goodFeaturesToTrack(img_ifft, 15, 0.01, 50, useHarrisDetector=True, k=0.04)    # For FFT
    corners = np.int0(corners)
    corners = corners.reshape(corners.shape[0], corners.shape[1]*corners.shape[2])

    return corners
    

def corners_ARTag(corners):
    
    # Finding the white background (paper) corners
    x_min_i = np.argmin(corners[:,0])
    x_max_i = np.argmax(corners[:,0])
    y_min_i = np.argmin(corners[:,1])
    y_max_i = np.argmax(corners[:,1])
    
    # Deleting the white background (paper) corners
    corners = np.delete(corners, (x_min_i,x_max_i,y_min_i,y_max_i), 0)

    x_vals = corners[:,0]
    y_vals = corners[:,1]

    # Min & Max values of corners now give AR Tag corners
    x_min = min(x_vals)
    y_x_min = corners[np.argmin(x_vals)][1]

    x_max = max(x_vals)
    y_x_max = corners[np.argmax(x_vals)][1]

    y_min = min(y_vals)
    x_y_min = corners[np.argmin(y_vals)][0]

    y_max = max(y_vals)
    x_y_max = corners[np.argmax(y_vals)][0]

    tag_corners = np.vstack([[x_min, y_x_min]])
    tag_corners = np.vstack([tag_corners,[x_max, y_x_max]])
    tag_corners = np.vstack([tag_corners,[x_y_min, y_min]])
    tag_corners = np.vstack([tag_corners,[x_y_max, y_max]])

    return tag_corners


def slice_ARTag(img, x_min, y_min, x_max, y_max):
    slice_img = img.copy()
    slice_img = slice_img[y_min:y_max, x_min:x_max]

    return slice_img


def decode_tag(img_warp):
    
    # Finding the corners of the bounding box formed by the first occurrence of 
    # white cells in the inner 4x4 grid
    white_corners = np.where(img_warp != 0) 
    x_min = np.min(white_corners[0])
    x_max = np.max(white_corners[0])
    y_min = np.min(white_corners[1])
    y_max = np.max(white_corners[1])

    # 4x4 grid in the AR Tag
    img_tag_4x4 = slice_ARTag(img_warp, x_min, y_min, x_max, y_max)

    # Size of the smallest cell in the grid
    cell_size_w = int((x_max-x_min)/4)
    cell_size_h = int((y_max-y_min)/4)

    # 4x4 grid in the AR Tag
    img_tag_2x2 = img_tag_4x4[cell_size_h:3*cell_size_h, cell_size_w:3*cell_size_w]

    # Obtaining the upper and bottom 4 corners of the 4x4 grid
    img_tag_4x4_TL = img_tag_4x4[0:cell_size_h, 0:cell_size_w]
    img_tag_4x4_TR = img_tag_4x4[0:cell_size_h, 3*cell_size_w:4*cell_size_w]
    img_tag_4x4_BL = img_tag_4x4[3*cell_size_h:4*cell_size_h, 0:cell_size_w]
    img_tag_4x4_BR = img_tag_4x4[3*cell_size_h:4*cell_size_h, 3*cell_size_w:4*cell_size_w]

    # Finding the upright orientation of the tag
    rotate_flag = 0
    if np.mean(img_tag_4x4_BR) >= 245:
        pass
    elif np.mean(img_tag_4x4_TR) >= 245:
        img_tag_2x2 = imutils.rotate_bound(img_tag_2x2, 90)    # CW rotation of 90 deg
        rotate_flag = 90
    elif np.mean(img_tag_4x4_BL) >= 245:
        img_tag_2x2 = imutils.rotate_bound(img_tag_2x2, -90)    # CCW rotation of 90 deg
        rotate_flag = -90
    elif np.mean(img_tag_4x4_TL) >= 245:
        img_tag_2x2 = imutils.rotate_bound(img_tag_2x2, 180)
        rotate_flag = 180

    plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(img_tag_4x4, cv2.COLOR_BGR2RGB))
    plt.title('Inner 4x4 Grid from Tag')
    plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(img_tag_2x2, cv2.COLOR_GRAY2RGB))
    plt.title('Inner 2x2 Grid (Upright Orientation))')
    plt.tight_layout()
    plt.show()

    # Obtaining the inner 2x2 grids
    cell_TL = img_tag_2x2[0:cell_size_h, 0:cell_size_w]    # LSB
    cell_TR = img_tag_2x2[0:cell_size_h, cell_size_w:2*cell_size_w]
    cell_BR = img_tag_2x2[cell_size_h:2*cell_size_h, cell_size_w:2*cell_size_w]
    cell_BL = img_tag_2x2[cell_size_h:2*cell_size_h, 0:cell_size_w]    # MSB
    
    # Computing the values of the inner 2x2 grids
    cell_TL = 1 if np.mean(cell_TL) >= 245 else 0
    cell_TR = 1 if np.mean(cell_TR) >= 245 else 0
    cell_BR = 1 if np.mean(cell_BR) >= 245 else 0
    cell_BL = 1 if np.mean(cell_BL) >= 245 else 0
    
    # Converting the binary value into decimal tag ID
    tag_id = 8*cell_BL + 4*cell_BR + 2*cell_TR + cell_TL

    return tag_id, rotate_flag


if __name__ == "__main__":

    # Decoding Reference AR Tag

    img_ref = cv2.imread('tag_ref_image.png')
    
    # Converting the frame to grayscale
    img_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    
    # Filtering out noise from the frame
    img_blur = cv2.medianBlur(img_gray, 7)
    _, img_thresh = cv2.threshold(img_blur, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to further enhance the frame 
    kernel = np.ones((5, 5), np.uint8)
    img_smooth = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    img_smooth = cv2.morphologyEx(img_smooth, cv2.MORPH_CLOSE, kernel)

    # Edge detection using FFT
    img_ifft = segment_edges(img_smooth)

    # Extracting all corners from the frame
    corners = extract_corners(img_ifft)
    
    # Obtaining the corners of the AR Tag
    tag_corners = corners_ARTag(corners)
    x_min, y_x_min = tag_corners[0]
    x_max, y_x_max = tag_corners[1]
    x_y_min, y_min = tag_corners[2]
    x_y_max, y_max = tag_corners[3]

    # Cropping out the rectangle enclosing the tag
    img_tag = slice_ARTag(img_smooth, x_min, y_min, x_max, y_max)

    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title('Reference AR Tag')
    plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(img_smooth, cv2.COLOR_GRAY2RGB))
    plt.title('Pre-processed Tag')

    tag_id, rotate_flag = decode_tag(img_tag)

    print(f"Decoded Reference AR Tag ID: {tag_id}")

