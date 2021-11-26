import numpy as np
import cv2
import math
import torch

CUDA = False

def calculate_center_and_principal(c):
    center_x = np.round(c['m10'] / c['m00']).astype(np.int)
    center_y = np.round(c['m01'] / c['m00']).astype(np.int)
    theta = math.atan2(2*c['mu11'], (c['mu20'] - c['mu02'])) / 2.0 * 180 / math.pi
    return center_x, center_y, theta

def crop_area(binary):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.boundingRect(contours[0])
    binary = binary[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return binary

def detect_area_center_and_principal(piece: np.ndarray):
    """

    :param piece:
    :return: [(img, (center_x, center_y), mode), ...], ret
    ret: 0 for no split, 1 for vertical split ([Upper, Buttom]), 2 for horizonal split ([Left, Right]),
    """

    ret, thresh = cv2.threshold(piece, 127, 255, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros_like(thresh)
    for i in range(len(contours)):
        cnt = contours[i]
        if i <= 1:
            cv2.drawContours(img, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(img, [cnt], 0, 0, -1)
    rect = cv2.boundingRect(contours[0])
    img = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    c = cv2.moments(img, binaryImage=True)

    center_x, center_y, angle = calculate_center_and_principal(c)

    # angle = getOrientation(contours[0])
    # center_x = np.round(c['m10'] / c['m00']).astype(np.int)
    # center_y = np.round(c['m01'] / c['m00']).astype(np.int)
    return img, (center_x, center_y), angle

# def getOrientation(pts):
#     sz = len(pts)
#     data_pts = np.empty((sz, 2), dtype=np.float64)
#     for i in range(data_pts.shape[0]):
#         data_pts[i,0] = pts[i,0,0]
#         data_pts[i,1] = pts[i,0,1]
#     # Perform PCA analysis
#     mean = np.empty((0))
#     mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
#     # mean, eigenvectors, eigenvalues = cv.PCACompute(data_pts, mean, 2) #image, mean=None, maxComponents=10
#     # Store the center of the object
#     angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])/math.pi*180. # orientation in degree #PCA第一维度的角度
#     # angle = angle + 180 if angle < 0 else angle
#     return angle

def check_flip(src, dst):
    hs, ws = src.shape
    hd, wd = dst.shape
    h = min(hs, hd)
    w = min(ws, wd)
    src = src[hs//2-h//2:hs//2+h//2, ws//2-w//2:ws//2+w//2]/255.
    dst = dst[hd//2-h//2:hd//2+h//2, wd//2-w//2:wd//2+w//2]/255.
    score1 = np.sum(np.abs(src-dst))
    score2 = np.sum(np.abs(src-np.flip(np.flip(dst, 0), 1)))
    if score1 > score2:
        return True
    else:
        return False


piece, center, princ = detect_area_center_and_principal(cv2.imread("../output/workpiece1.png", 0))
piece_vis = cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
cv2.circle(piece_vis, center, 20, (0, 0, 255), -1)
cv2.line(piece_vis,center,(center[0]+int(100*math.cos(princ/180*math.pi)), center[1]+int(100*math.sin(princ/180*math.pi))),(255, 0, 0),10)
cv2.namedWindow("Origin", cv2.WINDOW_NORMAL)
cv2.imshow("Origin", piece_vis)

piece_rot, center_rot, princ_rot = detect_area_center_and_principal(cv2.imread("../output/workpiece1_roted1.png", 0))
piece_rot_vis = cv2.cvtColor(piece_rot, cv2.COLOR_GRAY2BGR)
cv2.circle(piece_rot_vis, center_rot, 20, (0, 0, 255), -1)
cv2.line(piece_rot_vis,center_rot,(center_rot[0]+int(100*math.cos(princ_rot/180*math.pi)), center_rot[1]+int(100*math.sin(princ_rot/180*math.pi))),(255, 0, 0),10)
cv2.namedWindow("Rot", cv2.WINDOW_NORMAL)
cv2.imshow("Rot", piece_rot_vis)

h, w = piece_rot.shape
d = max(h, w)
piece_rot = np.pad(piece_rot, ((d, d), (d, d)))

rot = cv2.getRotationMatrix2D((center_rot[0]+d, center_rot[1]+d), princ_rot-princ, 1.0)
piece_correct = cv2.warpAffine(piece_rot, rot, (w+2*d, h+2*d))
piece_correct = crop_area(piece_correct)
# piece_correct, center_correct, princ_correct = detect_area_center_and_principal(piece_correct)

piece_correct = np.flip(np.flip(piece_correct, 0), 1) if check_flip(piece, piece_correct) else piece_correct
# piece_correct = np.flip(piece_correct.T, 1)


cv2.namedWindow("Corrected", cv2.WINDOW_NORMAL)
cv2.imshow("Corrected", piece_correct)

piece_torch = torch.Tensor(piece_correct).unsqueeze(0).unsqueeze(0)
piece_torch = piece_torch.transpose(2, 3)
piece_torch = torch.flip(piece_torch, [3])
piece_torch = piece_torch.numpy().astype(np.uint8)

cv2.namedWindow("Torch", cv2.WINDOW_NORMAL)
cv2.imshow("Torch", piece_correct)

cv2.waitKey(0)

