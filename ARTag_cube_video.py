# ARTag_cube_video.py
"""
Author: Adarsh Malapaka (amalapak@umd.edu), 2022
Brief: Constructs a virtual cube on the detected AR tag
       in the video feed using Projection Matrix and Image Warping.

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


def corners_ARTag(corners, img):
    
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


def compute_homography(x, y, xp, yp):

    A = np.array([[x[0], y[0], 1, 0, 0, 0, -x[0]*xp[0], -y[0]*xp[0], -xp[0]],
    [0, 0, 0, x[0], y[0], 1, -x[0]*yp[0], -y[0]*yp[0], -yp[0]],
    [x[1], y[1], 1, 0, 0, 0, -x[1]*xp[1], -y[1]*xp[1], -xp[1]],
    [0, 0, 0, x[1], y[1], 1, -x[1]*yp[1], -y[1]*yp[1], -yp[1]],
    [x[2], y[2], 1, 0, 0, 0, -x[2]*xp[2], -y[2]*xp[2], -xp[2]],
    [0, 0, 0, x[2], y[2], 1, -x[2]*yp[2], -y[2]*yp[2], -yp[2]],
    [x[3], y[3], 1, 0, 0, 0, -x[3]*xp[3], -y[3]*xp[3], -xp[3]],
    [0, 0, 0, x[3], y[3], 1, -x[3]*yp[3], -y[3]*yp[3], -yp[3]]])

    # SVD Decomposition
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


def projection_matrix(K, H):  
    
    # Coponents of the Homography matrix H
    h1 = H[:,0]         
    h2 = H[:,1]
    
    # SCaling in projection matrix
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    B_tilde = lamda * np.matmul(np.linalg.inv(K),H)

    B_tilde_det = np.linalg.det(B_tilde)

    # Sign of determinant of B determines position of projection wrt camera
    B = B_tilde if B_tilde_det >= 0 else -1*B_tilde
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    # Rotation matrix components    
    r1 = b1
    r2 = b2                      
    r3 = np.cross(r1, r2)
    
    # Translation vector
    t = b3
    R = np.column_stack((r1, r2, r3))    # Rotation Matrix
    R_t = np.column_stack((R, t))    
    
    P = np.matmul(K,R_t)    # Projection Matrix

    return P


if __name__ == "__main__":

    video_writer = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('AR_Tag_Cube.avi',video_writer,30,(1920,1080))
    cap = cv2.VideoCapture('1tagvideo.mp4')
    frame_count = 0    # Frame Number count variable 

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            try:
                
                # Converting the frame to grayscale
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Filtering out noise from the frame
                img_blur = cv2.medianBlur(img_gray, 5)
                _, img_thresh = cv2.threshold(img_blur, 240, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Morphological operations to further enhance the frame 
                kernel = np.ones((15, 15), np.uint8)
                img_smooth = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
                img_smooth = cv2.morphologyEx(img_smooth, cv2.MORPH_CLOSE, kernel)            

                # Edge detection using FFT
                img_ifft = segment_edges(img_smooth)

                # Extracting all corners from the frame
                corners = extract_corners(img_ifft)

                # Obtaining the corners of the AR Tag
                tag_corners = corners_ARTag(corners, frame)
                x_min, y_x_min = tag_corners[0]
                x_max, y_x_max = tag_corners[1]
                x_y_min, y_min = tag_corners[2]
                x_y_max, y_max = tag_corners[3]
                
                # Homography from the world tag corners to image tag corners
                X = [0, 0, 241, 241] 
                Y = [0, 241, 241, 0]
                Xp = [x_y_min, x_max, x_y_max, x_min]
                Yp = [y_min, y_x_max, y_max, y_x_min]
                H_matrix = compute_homography(X, Y, Xp, Yp)

                K = np.array([[1346.100595, 0, 932.1633975],
                              [ 0, 1355.933136, 654.8986796],
                              [ 0, 0, 1]])    # Camera Intrinsic Parameters matrix
                
                P = projection_matrix(K, H_matrix) 

                # 8 Coordinates of cube in 'world' reference frame
                x1,y1,z1 = np.matmul(P,[0,0,0,1])
                x2,y2,z2 = np.matmul(P,[0,241,0,1])
                x3,y3,z3 = np.matmul(P,[241,0,0,1])
                x4,y4,z4 = np.matmul(P,[241,241,0,1])
                x5,y5,z5 = np.matmul(P,[0,0,-241,1])
                x6,y6,z6 = np.matmul(P,[0,241,-241,1])
                x7,y7,z7 = np.matmul(P,[241,0,-241,1])
                x8,y8,z8 = np.matmul(P,[241,241,-241,1])
                
                cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255,255,0), 2)
                cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (255,255,0), 2)
                cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (255,255,0), 2)
                cv2.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (255,255,0), 2)

                cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,255), 2)
                cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,255), 2)
                cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,255), 2)
                cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,255), 2)

                cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (255,0,255), 2)
                cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (255,0,255), 2)
                cv2.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (255,0,255), 2)
                cv2.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (255,0,255), 2)

                frame = cv2.putText(frame,"Frame: "+str(frame_count),(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
                
                # out.write(frame)
                frame_count = frame_count + 1
                cv2.imshow("Virtual Cube on Tag", frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                continue

        else: 
            break

    # out.release()
    cap.release()
    cv2.destroyAllWindows()



