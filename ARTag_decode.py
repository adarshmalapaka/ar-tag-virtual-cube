# ARTag_decode.py
"""
Author: Adarsh Malapaka (amalapak@umd.edu), 2022
Brief: Detects the AR Tag from a single frame of the given video and 
       decodes the upright orientation and tag ID information using Warping.

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
    r = 180    # Mask Radius
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
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
    
    # Deleting the white background corners
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

  
def compute_homography(x, y, xp, yp):
    
    # SVD Decomposition
    A = np.array([[x[0], y[0], 1, 0, 0, 0, -x[0]*xp[0], -y[0]*xp[0], -xp[0]],
    [0, 0, 0, x[0], y[0], 1, -x[0]*yp[0], -y[0]*yp[0], -yp[0]],
    [x[1], y[1], 1, 0, 0, 0, -x[1]*xp[1], -y[1]*xp[1], -xp[1]],
    [0, 0, 0, x[1], y[1], 1, -x[1]*yp[1], -y[1]*yp[1], -yp[1]],
    [x[2], y[2], 1, 0, 0, 0, -x[2]*xp[2], -y[2]*xp[2], -xp[2]],
    [0, 0, 0, x[2], y[2], 1, -x[2]*yp[2], -y[2]*yp[2], -yp[2]],
    [x[3], y[3], 1, 0, 0, 0, -x[3]*xp[3], -y[3]*xp[3], -xp[3]],
    [0, 0, 0, x[3], y[3], 1, -x[3]*yp[3], -y[3]*yp[3], -yp[3]]])

    _, _, Vt = la.svd(A)
    Vt = np.transpose(Vt) 

    H = Vt[:,-1]
    H = H/H[-1]
    H_matrix = np.reshape(H, (3, 3))

    return H_matrix


def slice_ARTag(img, x_min, y_min, x_max, y_max):
    slice_img = img.copy()
    slice_img = slice_img[y_min:y_max, x_min:x_max]

    return slice_img


def warp_image(src, H, m_dest, n_dest, INVERSE_WARP = True):
    img_warp = np.zeros((m_dest,n_dest), dtype=np.uint8)

    try:

        # Inverse of Homography matrix (normalized)
        H_inv = np.linalg.inv(H)
        H_inv = H_inv/H_inv[-1,-1]
        
        if INVERSE_WARP:
            # Inverse Warping
            for i in range(m_dest):
                for j in range(n_dest):
                    X_c = np.array([[i, j, 1]]).T    # Camera coordinates
                    X_w = np.matmul(H_inv,X_c)    # World coordinates
                    img_warp[i,j] = src[int(X_w[1]/X_w[2]),int(X_w[0]/X_w[2])]
        else:
            # Forward Warping
            for i in range(src.shape[1]):
                for j in range(src.shape[0]):
                    X_w = np.array([[i, j, 1]]).T
                    X_c = np.matmul(H,X_w)
                    if 0 <= X_c[0] <= m_dest and 0 <= X_c[1] <= n_dest:
                        img_warp[int(X_c[0]/X_c[2]),int(X_c[1]/X_c[2])] = src[j,i] 
    except:
        pass
    # img_warp_smooth = cv2.GaussianBlur(img_warp, (3, 3), 0)

    # Smoothening & Thresholding the warped image to remove sampling inconsistencies
    img_warp_smooth = cv2.bilateralFilter(img_warp, 2, 2.2, 2.2, cv2.BORDER_DEFAULT)
    _, img_warp_thresh = cv2.threshold(img_warp_smooth, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    # plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(img_warp, cv2.COLOR_GRAY2RGB))
    # plt.title('Forward Warping')
    # plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(img_warp_smooth, cv2.COLOR_BGR2RGB))
    # plt.title('Smoothened Forward Warping')
    # plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(img_warp_thresh, cv2.COLOR_GRAY2RGB))
    # plt.title('Thresholded Forward Warping')
    # plt.suptitle('Forward Warping Results')
    # plt.tight_layout()
    # plt.show()

    return img_warp_thresh


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

    plt.subplot(3, 2, 4), plt.imshow(cv2.cvtColor(img_tag_4x4, cv2.COLOR_GRAY2RGB))
    plt.title('Inner 4x4 Grid')
    plt.subplot(3, 2, 5), plt.imshow(cv2.cvtColor(img_tag_2x2, cv2.COLOR_GRAY2RGB))
    plt.title('Inner 2x2 Grid')
    
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

    plt.subplot(3, 2, 6), plt.imshow(cv2.cvtColor(img_tag_2x2, cv2.COLOR_GRAY2RGB))
    plt.title('Inner 2x2 Grid (Upright)')
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
    
    # Decoding AR Tag from single frame of video

    img = cv2.imread('tag_single_frame.jpg')
    
    # Converting the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Filtering out noise from the image
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

    # Homography from the image tag corners to world tag corners
    X = [x_y_min, x_max, x_y_max, x_min]
    Y = [y_min, y_x_max, y_max, y_x_min]
    Xp = [0, 0, 241, 241] 
    Yp = [0, 241, 241, 0]
    H_matrix = compute_homography(X, Y, Xp, Yp)    

    # Cropping out the rectangle enclosing the tag
    img_tag = slice_ARTag(img_gray, x_min, y_min, x_max, y_max)

    # Warping the cropped tag to upright world tag co-ordinates
    img_warp = warp_image(img_gray, H_matrix, 240, 240, True)   
    
    plt.subplot(3, 2, 1), plt.imshow(cv2.cvtColor(img_smooth, cv2.COLOR_GRAY2RGB))
    plt.title('Pre-processed Image')
    plt.subplot(3, 2, 2), plt.imshow(cv2.cvtColor(img_tag, cv2.COLOR_BGR2RGB))
    plt.title('Detected AR Tag')
    plt.subplot(3, 2, 3), plt.imshow(cv2.cvtColor(img_warp, cv2.COLOR_GRAY2RGB))
    plt.title('Warped AR Tag')
    
    # Decoding ID and rotation from the tag
    tag_id = decode_tag(img_warp)[0]
    
    print(f"Decoded video AR Tag ID: {tag_id}")

    cv2.circle(img,(x_min, y_x_min),8,(255,255,0),4)
    cv2.circle(img,(x_max, y_x_max),8,(255,255,0),4)
    cv2.circle(img,(x_y_min, y_min),8,(255,255,0),4)
    cv2.circle(img,(x_y_max, y_max),8,(255,255,0),4)

    cv2.line(img, (x_min, y_x_min), (x_y_min, y_min), (0,0,255), 5)
    cv2.line(img, (x_y_min, y_min), (x_max, y_x_max), (0,0,255), 5)
    cv2.line(img, (x_max, y_x_max), (x_y_max, y_max), (0,0,255), 5)
    cv2.line(img, (x_y_max, y_max), (x_min, y_x_min), (0,0,255), 5)

    cv2.putText(img,"Tag ID: "+str(tag_id),(int((x_min+x_max)/2-30), int((y_min+y_max)/2+200)),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
                
    cv2.imshow("Detected AR Tag", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()