import numpy as np
import cv2
from utils.gribber import Gribber, SortGribberConfig, StackGribberConfig
from utils.DB.DbResposity import *
from utils.parse_dxf import *
import torch
import torch.nn.functional as F
import math

WTHRESHOLDS = [300, 2200, 3200, 5000, 7000, 13000]  # 倒数第二个参数为是否使用两个抓手进行抓取的阈值，<=该参数使用单手抓取，>该参数使用双手抓取
HTHRESHOLDS = [75, 200, 500]
LOCTHRESH = 10
# CUDA = torch.cuda.is_available()
CUDA = False
HeightThresholdDown = 75  # 高度>=该阈值进行抓取
WidthThresholdDown = 300  # 宽度>=该阈值进行抓取
HeightThresholdUp = 1500  # 高度<=该阈值进行抓取
WidthThresholdUp = 12000  # 宽度<=该阈值进行抓取
# HORIGRABTHRESHOLD_PART = 600  # <=该阈值则只进行水平抓取
# VERTICALGRABTHRESHOLD_COORD_REV = 1580  # 若抓取点距离料板右端距离<=该阈值则只进行垂直抓取


def check_for_grab(partDB):
    for n in partDB.keys():
        partDB[n].update({
            "Grabbability": (max(partDB[n]["PartWidth"], partDB[n]["PartHeight"]) >= WidthThresholdDown and
                            min(partDB[n]["PartWidth"], partDB[n]["PartHeight"]) >= HeightThresholdDown) and
                            (max(partDB[n]["PartWidth"], partDB[n]["PartHeight"]) <= WidthThresholdUp and
                             min(partDB[n]["PartWidth"], partDB[n]["PartHeight"]) <= HeightThresholdUp) and
                            partDB[n]["RequireSort"] and (partDB[n]["PartWidth"] >= partDB[n]["PartHeight"])
        })
    return partDB


def select_kernel_by_area(area, hw=None):
    kernel_size = None
    need_break = False
    h, w = area.shape if hw is None else hw
    h, w = min(h, w), max(h, w)
    if WTHRESHOLDS[0] <= w <= WTHRESHOLDS[1]:
        if HTHRESHOLDS[0] <= h <= HTHRESHOLDS[1]:
            kernel_size = '50'
            need_break = False
        elif HTHRESHOLDS[1] < h <= HTHRESHOLDS[2]:
            kernel_size = '100'
            need_break = False
        elif HTHRESHOLDS[2] < h:
            kernel_size = 'small'
            need_break = False
        else:
            raise ValueError("Area size is not valid: ", h, w)
    elif WTHRESHOLDS[1] < w <= WTHRESHOLDS[2]:
        kernel_size = 'medium'
        need_break = False
    elif WTHRESHOLDS[2] < w <= WTHRESHOLDS[4]:
        kernel_size = 'large'
        need_break = False
    # elif WTHRESHOLDS[2] < w <= WTHRESHOLDS[3]:
    #     kernel_size = 'large'
    #     need_break = False
    # elif WTHRESHOLDS[3] < w <= WTHRESHOLDS[4]:
    #     kernel_size = 'medium'
    #     need_break = True
    elif WTHRESHOLDS[4] < w <= WTHRESHOLDS[5]:
        kernel_size = 'large'
        need_break = True
    else:
        raise ValueError("Area size is not valid: ", h, w)

    return need_break, kernel_size


def grab_by_gravity_center(piece, kernels_dict, mode, rot_base, center: tuple, id=0, split=0, sn='0', num=0, tgt_angle=None):
    kernels = kernels_dict[mode]
    if len(piece.shape) < 3:
        piece = piece.unsqueeze(0)
    if len(kernels.shape) < 3:
        kernels = kernels.unsqueeze(0)

    _, h, w = piece.shape
    if h % 2 != 0:
        piece = F.pad(piece, (0, 0, 0, 1), mode='constant', value=0)
    if w % 2 != 0:
        piece = F.pad(piece, (0, 1, 0, 0), mode='constant', value=0)

    # centerize the gravity center
    center_x, center_y = center
    if id == 1:
        if split == 1:
            center_y -= h
        elif split == 2:
            center_x -= w
    _, h, w = piece.shape
    bias_x = (w//2 - center_x)*2
    piece = F.pad(piece, (bias_x, 0, 0, 0), mode='constant', value=0) if bias_x >=0 else F.pad(piece, (0, -bias_x, 0, 0), mode='constant', value=0)
    bias_y = (h//2 - center_y)*2
    piece = F.pad(piece, (0, 0, bias_y, 0), mode='constant', value=0) if bias_y >=0 else F.pad(piece, (0, 0, 0, -bias_y), mode='constant', value=0)

    if h % 2 != 0:
        piece = F.pad(piece, (0, 0, 0, 1), mode='constant', value=0)
    if w % 2 != 0:
        piece = F.pad(piece, (0, 1, 0, 0), mode='constant', value=0)

    _, h, w = piece.shape
    ck, hk, wk = kernels.shape
    pad_y = np.ceil((hk - h)/2).astype(np.int) if hk > h else 0
    pad_x = np.ceil((wk - w)/2).astype(np.int) if wk > w else 0
    if pad_x != 0 or pad_y != 0:
        piece = F.pad(piece, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)
    _, h, w = piece.shape
    piece_cut = piece[:, (h-hk)//2:(h+hk)//2, (w-wk)//2:(w+wk)//2]

    scores = (piece_cut.expand_as(kernels) * kernels).sum(-1).sum(-1)
    max_index = scores.argmax().long().numpy() if tgt_angle is None else tgt_angle // rot_base
    kernel = kernels[max_index]
    # print(scores.shape)
    # print(max_index)
    visualize_grab_point(piece, kernel, sn, id, num)

    kernel_vis = postprocess_torch(kernel * piece_cut)  # TODO 在旋转之前检测圆然后转换圆心坐标
    rot = cv2.getRotationMatrix2D((wk//2, hk//2), -max_index * rot_base, 1.0)
    kernel_vis = cv2.warpAffine(kernel_vis, rot, (wk, hk))
    # cv2.imshow("kernel_vis", kernel_vis)
    # cv2.waitKey(0)

    src = cv2.GaussianBlur(kernel_vis, (3, 3), 1.5)
    circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=50, param2=25, minRadius=20, maxRadius=0)
    '''
    param1	First method-specific parameter. In case of HOUGH_GRADIENT and HOUGH_GRADIENT_ALT, it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller). Note that HOUGH_GRADIENT_ALT uses Scharr algorithm to compute image derivatives, so the threshold value shough normally be higher, such as 300 or normally exposed and contrasty images.
    param2	Second method-specific parameter. In case of HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first. In the case of HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure. The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine. If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less. But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
    '''
    if circles is None:
        src = cv2.GaussianBlur(kernel_vis, (5, 5), 1.5)
        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=50, param2=25, minRadius=10,
                                   maxRadius=0)
    if circles is None:
        return -1, None

    circles = np.uint16(np.around(circles))
    # circles = [i for i in circles[0, :]]
    circles = doublecheck_circles(circles, kernel_vis, kernels_dict)
    if len(circles) <= 1:  # if the
        print("The effective gribber only exist for 1, not grabable!")
        return -1, None
    kernel_vis = cv2.cvtColor(kernel_vis, cv2.COLOR_GRAY2BGR)
    for i in circles:
        cv2.circle(kernel_vis, (i[0], i[1]), i[2], (255, 0, 0), 10)
        cv2.circle(kernel_vis, (i[0], i[1]), 10, (0, 0, 255), -1)
    cv2.imwrite("./output/kernel_vis/kernel_vis_num{}_sn{}_id{}.png".format(num, sn, id), kernel_vis)
    # cv2.imwrite("./output/kernel.png", postprocess_torch(kernel))
    return max_index, circles


def doublecheck_circles(circles, img, kernels_dict):
    img = preprocess_torch(img)
    circles_filtered = []
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        if radius <= 20 or radius >= 70:
            continue
        kernel = kernels_dict['c100'] if radius > 30 else kernels_dict['c50']
        rad_ref = 50 if radius > 30 else 25
        if center[1]-rad_ref < 0 or center[0]-rad_ref < 0 or center[1]+rad_ref+1 > img.shape[0] or center[0]+rad_ref+1 > img.shape[1]:
            continue
        src = img[center[1]-rad_ref:center[1]+rad_ref+1, center[0]-rad_ref:center[0]+rad_ref+1]
        score1 = torch.sum(src * kernel) / (2*rad_ref)**2
        score2 = torch.sum(src * kernel) / (2*radius)**2
        if score1 >= 0.72 and score2 >= 0.61:  # pi/4 = 0.7853
            circles_filtered.append(i)
    return circles_filtered


def visualize_grab_point(piece, kernel, sn='0', id=0, num=0):
    _, h, w = piece.shape
    hk, wk = kernel.shape
    pad_y = np.ceil((h - hk)/2).astype(np.int) if h > hk else 0
    pad_x = np.ceil((w - wk)/2).astype(np.int) if w > wk else 0
    if pad_x != 0 or pad_y != 0:
        kernel = F.pad(kernel, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)

    piece = postprocess_torch(piece)
    kernel = postprocess_torch(kernel)
    kernel_vis = np.zeros((kernel.shape[0], kernel.shape[1], 3), dtype=np.uint8)
    kernel_vis[:, :, -1] = kernel
    piece = cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
    image = cv2.addWeighted(piece, 0.5, kernel_vis, 0.5, 0)

    cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), 50, (255, 0, 0), -1)
    cv2.imwrite("./output/grab_results/grab_results_num{}_sn{}_id{}.png".format(num, sn, id), image)
    # cv2.namedWindow("visualize", cv2.WINDOW_NORMAL)
    # cv2.imshow("visualize", image)
    # cv2.waitKey(0)


def calculate_center_and_principal(c):
    center_x = np.round(c['m10'] / c['m00']).astype(np.int)
    center_y = np.round(c['m01'] / c['m00']).astype(np.int)
    theta = math.atan2(2*c['mu11'], (c['mu20'] - c['mu02'])) / 2.0 * 180 / math.pi
    return center_x, center_y, theta


def detect_area_center_and_principal(piece: np.ndarray, with_hole=True, binary=False):
    """

    :param piece:
    :return: [(img, (center_x, center_y), mode), ...], ret
    ret: 0 for no split, 1 for vertical split ([Upper, Buttom]), 2 for horizonal split ([Left, Right]),
    """
    if not binary:
        ret, thresh = cv2.threshold(piece, 180, 255, 1)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = np.zeros_like(thresh)

        # cv2.namedWindow("Piece", cv2.WINDOW_NORMAL)
        # cv2.imshow("Piece", piece)
        # cv2.waitKey(0)

        max_level = count_max_level(contours, hierarchy)
        if max_level == 1:
            for i in range(len(contours)):
                if outer_level_target(i, hierarchy, max_level - 1):
                    cv2.drawContours(img, contours, i, 255, -1)
        elif max_level == 2:
            max_level += 1
            for i in range(len(contours)):
                if outer_level_target(i, hierarchy, max_level - 3):
                    cv2.drawContours(img, contours, i, 255, -1)
            for i in range(len(contours)):
                if outer_level_target(i, hierarchy, max_level - 1) and with_hole:
                    cv2.drawContours(img, contours, i, 0, -1)
        elif max_level == 3:
            for i in range(len(contours)):
                if outer_level_target(i, hierarchy, max_level - 3):
                    cv2.drawContours(img, contours, i, 255, -1)
            for i in range(len(contours)):
                if outer_level_target(i, hierarchy, max_level - 1) and with_hole:
                    cv2.drawContours(img, contours, i, 0, -1)
        else:
            raise ValueError("max level is {}, which is illegal! Please check the dxf parse result!".format(max_level))

        rect = cv2.boundingRect(contours[0])
        img = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    else:
        img = piece

    ret, mode = select_kernel_by_area(img)
    h, w = img.shape

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    if ret:
        if h > w:
            imgL, imgR = img[:h//2, :], img[h//2:, :]
            ret = 1
        else:
            imgL, imgR = img[:, :w//2], img[:, w//2:]
            ret = 2

        c = cv2.moments(imgL, binaryImage=True)
        center_x, center_y, princ = calculate_center_and_principal(c)
        cL = cv2.moments(imgL, binaryImage=True)
        center_xL, center_yL, _ = calculate_center_and_principal(cL)
        cR = cv2.moments(imgR, binaryImage=True)
        center_xR, center_yR, _ = calculate_center_and_principal(cR)
        if ret == 1:
            center_yR += h//2
        elif ret == 2:
            center_xR += w//2
        return [(imgL, (center_xL, center_yL), mode), (imgR, (center_xR, center_yR), mode)], ret, img, princ, \
               (center_x, center_y, center_xL, center_yL, center_xR, center_yR)
    else:
        c = cv2.moments(img, binaryImage=True)
        center_x, center_y, princ = calculate_center_and_principal(c)
        return [(img, (center_x, center_y), mode)], 0, img, princ, (center_x, center_y)


def preprocess_torch(img: np.ndarray):
    img = torch.Tensor(img)/255.
    img = img.cuda() if CUDA else img
    return img


def postprocess_torch(img: torch.Tensor):
    img = img.cpu() if CUDA else img
    return (img.squeeze() * 255.).numpy().astype(np.uint8)


def to_decimal(effect):
    assert len(effect) == 12 or len(effect) == 24, "The effect list length is illegal!"
    dec = 0
    for i, e in enumerate(effect):
        delta = (2 * e) ** i if e != 0 else 0
        dec += delta
    return dec


def generate_magnite_effective_list(grabpoint, rot, circles, coords_dict, mode, id, split, r=625):
    '''
    :param circles: the circles detected
    :param rot: the rotation angle in degree
    :param coords_dict: the coordinates of the magnite center on the kernels
    :param mode: the mode of the kernels
    :param id: the part id of the workpiece, 0 for the upper or left part, 1 for the buttom or right part
    :param split: 0 if the workpiece is not splited, 1 for the vertical split, 2 for the horizonal split.
    :return:
    '''
    deployment = {}
    grabpoint_ref = grabpoint
    deployL = []
    deployS = []

    coords_dict = coords_dict[mode]
    keys = list(coords_dict.keys())
    keys.sort()

    coordsL, coordsS = [], []
    keysL, keysS = [], []
    for k in keys:
        if k[0] == 'L':
            coordsL.append(coords_dict[k])
            keysL.append(k)
        elif k[0] == 'S':
            coordsS.append(coords_dict[k])
            keysS.append(k)
        deployment[k] = 0
    coordsL = np.array(coordsL)
    coordsS = np.array(coordsS)

    for c in circles:
        center = np.array((c[0], c[1]))[None, ...]
        radius = c[2]
        if radius > 30:
            centers = np.repeat(center, coordsL.shape[0], axis=0)
            dist = np.sum(np.abs(centers - coordsL), axis=-1)
            min_dist = dist.min(axis=0)
            if min_dist < LOCTHRESH:
                index = dist.argmin(axis=0)
                deployment[keysL[index]] = 1
        else:
            centers = np.repeat(center, coordsS.shape[0], axis=0)
            dist = np.sum(np.abs(centers - coordsS), axis=-1)
            min_dist = dist.min(axis=0)
            if min_dist < LOCTHRESH:
                index = dist.argmin(axis=0)
                deployment[keysS[index]] = 1

    if mode == 'large':
        grabpoint_ref = grabpoint
        deployL = [deployment[k] for k in keysL]
        deployS = [deployment[k] for k in keysS]
    elif mode == 'small':
        grabpoint_ref = grabpoint
        deployL = [deployment[k] for k in keysL]
        deployL = [0 for _ in range(4)] + [d for d in deployL[:4]] + [0 for _ in range(8)] + [d for d in deployL[4:]] + [0 for _ in range(4)]
        deployS = [0 for _ in range(4)] + [deployment[k] for k in keysS] + [0 for _ in range(4)]
    elif mode == '100':
        x, y = grabpoint
        x_r = x - 125 * math.sin(rot/180*math.pi)  # (125)
        y_r = y + 125 * math.cos(rot/180*math.pi)
        grabpoint_ref = (x_r, y_r)
        deployL = [deployment[k] for k in keysL]
        deployL = [0 for _ in range(4)] + [d for d in deployL[:4]] + [0 for _ in range(16)]
        deployS = [0 for _ in range(12)]
    elif mode == '50':
        x, y = grabpoint
        x_r = x + 25 * math.sin(rot / 180 * math.pi)  # (-25)
        y_r = y - 25 * math.cos(rot / 180 * math.pi)
        grabpoint_ref = (x_r, y_r)
        deployL = [0 for _ in range(24)]
        deployS = [0 for _ in range(4)] + [deployment[k] for k in keysS] + [0 for _ in range(4)]
    elif mode == 'medium':  # use the left side by default
        deployL = [deployment[k] for k in keysL]
        deployL = [d for d in deployL[:8]] + [0 for _ in range(4)] + [d for d in deployL[8:]] + [0 for _ in range(4)]
        deployS = [deployment[k] for k in keysS] + [0 for _ in range(4)]
        grabpoint_ref = (grabpoint[0] + r * math.cos(rot / 180 * math.pi), grabpoint[1] + r * math.sin(rot / 180 * math.pi))

        # if split == 1:  # split [upper, buttom]
        #     if id == 0:  # upper
        #         if 0 <= rot <= 180:
        #             grabpoint_ref = (
        #             grabpoint[0] + r * math.cos(rot / 180 * math.pi), grabpoint[1] + r * math.sin(rot / 180 * math.pi))
        #         else:
        #             grabpoint_ref = (grabpoint[0] - r * math.cos(rot / 180 * math.pi),
        #                              grabpoint[1] - r * math.sin(rot / 180 * math.pi))
        #             deployL.reverse()
        #             deployS.reverse()
        #
        #     else:  # buttom
        #         deployL = [deployment[k] for k in keysL]
        #         deployL = [d for d in deployL[:8]] + [0 for _ in range(4)] + [d for d in deployL[8:]] + [0 for _ in
        #                                                                                                  range(4)]
        #         deployS = [deployment[k] for k in keysS] + [0 for _ in range(4)]
        #         if 0 <= rot <= 180:
        #             grabpoint_ref = (grabpoint[0] - r * math.cos(rot / 180 * math.pi),
        #                              grabpoint[1] - r * math.sin(rot / 180 * math.pi))
        #             deployL.reverse()
        #             deployL.reverse()
        #         else:
        #             grabpoint_ref = (grabpoint[0] + r * math.cos(rot / 180 * math.pi),
        #                              grabpoint[1] + r * math.sin(rot / 180 * math.pi))
        #
        # elif split == 2: # split [left, right]
        #     if id == 0:  # left
        #         if 90 <= rot <= 270:
        #             grabpoint_ref = (
        #             grabpoint[0] + r * math.cos(rot / 180 * math.pi), grabpoint[1] + r * math.sin(rot / 180 * math.pi))
        #         else:
        #             grabpoint_ref = (grabpoint[0] - r * math.cos(rot / 180 * math.pi),
        #                              grabpoint[1] - r * math.sin(rot / 180 * math.pi))
        #             deployL.reverse()
        #             deployS.reverse()
        #     else:  # right
        #         if 90 <= rot <= 270:
        #             grabpoint_ref = (grabpoint[0] - r * math.cos(rot / 180 * math.pi),
        #                              grabpoint[1] - r * math.sin(rot / 180 * math.pi))
        #             deployL.reverse()
        #             deployL.reverse()
        #         else:
        #             grabpoint_ref = (grabpoint[0] + r * math.cos(rot / 180 * math.pi),
        #                              grabpoint[1] + r * math.sin(rot / 180 * math.pi))
        # else:  #
        #     grabpoint_ref = (grabpoint[0] + r * math.cos(rot/180*math.pi), grabpoint[1] + r * math.sin(rot/180*math.pi))
        #     deployL = [deployment[k] for k in keysL]
        #     deployL = [d for d in deployL[:8]] + [0 for _ in range(4)] + [d for d in deployL[8:]] + [0 for _ in range(4)]
        #     deployS = [deployment[k] for k in keysS] + [0 for _ in range(4)]
    else:
        raise ValueError

    return deployL, deployS, grabpoint_ref


def refine_effect_list(src: list):
    out = []
    for i, j in zip(range(12), range(12, 24)):
        out.append(src[j])
        out.append(src[i])
    return out


def grab_plan_sort(data, config):

    kernel_path = config['GribberKernelPath']
    gribber_config = SortGribberConfig
    gribber_config.update(config)
    grib = Gribber(gribber_config, kernel_path=kernel_path)
    # grib = Gribber({'cuda': CUDA})
    kernels = {
        'large': grib.kernels_large,
        'medium': grib.kernels_medium,
        'small': grib.kernels_small,
        '100': grib.kernels_100,
        '50': grib.kernels_50,
        'c100': grib.circle_100,
        'c50': grib.circle_50
    }
    coords = {
        'large': grib.magnit_coords_large,
        'medium': grib.magnit_coords_medium,
        'small': grib.magnit_coords_small,
        '100': grib.magnit_coords_100,
        '50': grib.magnit_coords_50,
    }

    for nestID in data.keys():
        print("Planning for nest {}".format(nestID))
        nest = data[nestID]["Parts"]
        nestHeight = data[nestID]["Height"]
        nestWidth = data[nestID]["Width"]
        for i in list(nest.keys()):
            print("Planning part {} for grab!".format(nest[i]['PartSN']))
            if not nest[i]['Grabbability']:
                print("The part {} is not grabbable!".format(nest[i]['PartSN']))
                continue
            areas_and_centers, split, piece, principcal_direction, gravity_center = detect_area_center_and_principal(
                nest[i]['Part'])
            nest[i]['Center'] = gravity_center
            nest[i]['Orientation'] = principcal_direction
            nest[i]['Piece'] = piece
            nest[i]['Grab'] = {}
            partWidth = nest[i]['PartWidth']
            partHeight = nest[i]['PartWidth']

            for id, ac in enumerate(areas_and_centers):
                area = ac[0]
                center = ac[1]
                mode = ac[2]
                origin = nest[i]['Origin']

                # if origin[0] >= nestWidth - grib.width:
                #     index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                #                                             center, id, split=split, sn=nest[i]['PartSN'], num=i, tgt_angle=90)
                # elif max(partWidth, partHeight) <= HORIGRABTHRESHOLD_PART:
                #     index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                #                                             center, id, split=split, sn=nest[i]['PartSN'], num=i, tgt_angle=0)
                # else:
                #     index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                #                                             center, id, split=split, sn=nest[i]['PartSN'], num=i)
                index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                                                            center, id, split=split, sn=nest[i]['PartSN'], num=i)
                if index == -1:
                    # print("Programming failed, the part {} is not grabbable!".format(nest[i]['PartSN']))
                    # nest[i]['Grabbability'] = False
                    # data[nestID]["Parts"].update(nest)
                    # continue
                    # print("Retrying programming grab for the part {} ignoring hole ...".format(nest[i]['PartSN']))
                    # areas_and_centers_local, _, _, _, _ = detect_area_center_and_principal(nest[i]['Part'], with_hole=False)
                    # area, center, mode = areas_and_centers_local[id][0], areas_and_centers_local[id][1], areas_and_centers_local[id][2]
                    # center, mode = areas_and_centers_local[id][1], areas_and_centers_local[id][2]

                    print("Retrying programming grab for the part {} with drift ...".format(nest[i]['PartSN']))
                    while center[0] > area.shape[1]//2 - kernels[mode].shape[1] // 2 and area.shape[1] > kernels[mode].shape[1]:
                        center = (center[0]-10, center[1])
                        index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                                                                center, id, sn=nest[i]['PartSN'], num=i)
                        if index != -1:
                            break
                    if index == -1:
                        print("Reprogramming Failed, the part {} is not grabbable!".format(nest[i]['PartSN']))
                        nest[i]['Grabbability'] = False
                        data[nestID]["Parts"].update(nest)
                        continue
                    else:
                        print("Reprogramming Success, continuing ...")
                theta = index * grib.rotation_angle
                effect_list_100, effect_list_50, center_refined = generate_magnite_effective_list(center,
                                                                                                  theta,
                                                                                                  circles,
                                                                                                  coords,
                                                                                                  mode,
                                                                                                  id,
                                                                                                  split,
                                                                                                  r=grib.c2g_med)

                center_refined = (center_refined[0] + origin[0], center_refined[1] + origin[1])  # transform to the plane reference

                nest[i]['Mode'] = mode
                nest[i]['Grab'][id] = {
                    'Grabpoint': (center_refined[0], nestHeight - center_refined[1]),  # transfer the origin from top left to buttom left
                    'Theta': index * grib.rotation_angle,
                    '100': to_decimal(refine_effect_list(effect_list_100)),
                    '50': to_decimal(effect_list_50),
                    'List100': effect_list_100,
                    'List50': effect_list_50,
                }
        data[nestID].update({"GrabStatus": 1})
        data[nestID]["Parts"].update(nest)


def grab_plan_stack(data, config):
    kernel_path = config['GribberKernelPath']
    gribber_config = StackGribberConfig
    gribber_config.update(config)
    grib = Gribber(gribber_config, kernel_path=kernel_path)
    kernels = {
        'large': grib.kernels_large,
        'medium': grib.kernels_medium,
        'small': grib.kernels_small,
        '100': grib.kernels_100,
        '50': grib.kernels_50,
        'c100': grib.circle_100,
        'c50': grib.circle_50
    }
    coords = {
        'large': grib.magnit_coords_large,
        'medium': grib.magnit_coords_medium,
        'small': grib.magnit_coords_small,
        '100': grib.magnit_coords_100,
        '50': grib.magnit_coords_50,
    }
    for nestID in data.keys():
        print("Planning for nest {}".format(nestID))
        nest = data[nestID]["Parts"]
        nestHeight = data[nestID]["Height"]
        nestWidth = data[nestID]["Width"]
        for i in list(nest.keys()):
            print("Planning part {} for grab!".format(nest[i]['PartSN']))
            if not nest[i]['Grabbability']:
                print("The part {} is not grabbable!".format(nest[i]['PartSN']))
                continue
            areas_and_centers, split, piece, principcal_direction, gravity_center = detect_area_center_and_principal(
                nest[i]['Part'], binary=True)
            nest[i]['Center'] = gravity_center
            nest[i]['Orientation'] = principcal_direction
            nest[i]['Piece'] = piece
            nest[i]['Grab'] = {}
            partWidth = nest[i]['PartWidth']
            partHeight = nest[i]['PartWidth']

            for id, ac in enumerate(areas_and_centers):
                area = ac[0]
                center = ac[1]
                mode = ac[2]
                origin = nest[i]['Origin']

                # if origin[0] >= nestWidth - grib.width:
                #     index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                #                                             center, id, split=split, sn=nest[i]['PartSN'], num=i, tgt_angle=90)
                # elif max(partWidth, partHeight) <= HORIGRABTHRESHOLD_PART:
                #     index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                #                                             center, id, split=split, sn=nest[i]['PartSN'], num=i, tgt_angle=0)
                # else:
                #     index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                #                                             center, id, split=split, sn=nest[i]['PartSN'], num=i)

                index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                                                            center, id, split=split, sn=nest[i]['PartSN'], num=i)

                if index == -1:
                    # print("Programming failed, the part {} is not grabbable!".format(nest[i]['PartSN']))
                    # nest[i]['Grabbability'] = False
                    # data[nestID]["Parts"].update(nest)
                    # continue
                    # print("Retrying programming grab for the part {} ignoring hole ...".format(nest[i]['PartSN']))
                    # areas_and_centers_local, _, _, _, _ = detect_area_center_and_principal(nest[i]['Part'], with_hole=False)
                    # area, center, mode = areas_and_centers_local[id][0], areas_and_centers_local[id][1], areas_and_centers_local[id][2]
                    # index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                    #                                         center, id, sn=nest[i]['PartSN'], num=i)
                    # if index == -1:
                    #     print("Reprogramming Failed, the part {} is not grabbable!".format(nest[i]['PartSN']))
                    #     nest[i]['Grabbability'] = False
                    #     data[nestID]["Parts"].update(nest)
                    #     continue
                    # else:
                    #     print("Reprogramming Success, continuing ...")
                    print("Retrying programming grab for the part {} with drift ...".format(nest[i]['PartSN']))
                    while center[0] > area.shape[1]//2 - kernels[mode].shape[1] // 2 and area.shape[1] > kernels[mode].shape[1]:
                        center = (center[0]-10, center[1])
                        index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle,
                                                                center, id, sn=nest[i]['PartSN'], num=i)
                        if index != -1:
                            break
                    if index == -1:
                        print("Reprogramming Failed, the part {} is not grabbable!".format(nest[i]['PartSN']))
                        nest[i]['Grabbability'] = False
                        data[nestID]["Parts"].update(nest)
                        continue
                    else:
                        print("Reprogramming Success, continuing ...")
                theta = index * grib.rotation_angle
                effect_list_100, effect_list_50, center_refined = generate_magnite_effective_list(center,
                                                                                                  theta,
                                                                                                  circles,
                                                                                                  coords,
                                                                                                  mode,
                                                                                                  id,
                                                                                                  split,
                                                                                                  r=grib.c2g_med)

                center_refined = (center_refined[0] + origin[0], center_refined[1] + origin[1])  # transform to the plane reference

                nest[i]['Mode'] = mode
                nest[i]['Grab'][id] = {
                    'Grabpoint': (center_refined[0], nestHeight - center_refined[1]),  # transfer the origin from top left to buttom left
                    'Theta': index * grib.rotation_angle,
                    '100': to_decimal(refine_effect_list(effect_list_100)),
                    '50': to_decimal(effect_list_50),
                    'List100': effect_list_100,
                    'List50': effect_list_50,
                }
        data[nestID].update({"GrabStatus": 1})
        data[nestID]["Parts"].update(nest)


def grab_apply_offset(part: dict, offset: tuple, references, device_code="LC-01"):
    '''

    rotation matrix = [
    cos, sin
    -sin, cos
    ]

    x_new = delta_x + x * sin(delta_theta) + y * cos(delta_theta)
    y_new = delta_y + x * cos(delta_theta) - y * sin(delta_theta)
    theta_new = theta + delta_theta
    Orientation = Orientation + delta_theta
    :param part: {"PartID": xxx, "Origin": xxx, "Part": xxx, "Center": xxx, "Orientation": xxx, "Grab": {
    0: {"Grabpoint": (x, y), "Theta": degree, 100: list, 50: list},
    1: {"Grabpoint": (x, y), "Theta": degree, 100: list, 50: list}
    }}
    :param offset: (delta_x, delta_y, delta_theta), delta_x, delta_y is the coordinate of the plane origin in the global reference,
    delta_theta is in degree
    :param references: {
    "MidDict": {}
    :return:
    '''
    grib_set_length = 1175

    delta_x, delta_y, delta_theta = offset

    part["Orientation"] += delta_theta
    grab_num = len(part["Grab"].keys())
    for n in part["Grab"].keys():
        x, y = part["Grab"][n]["Grabpoint"]
        part["Grab"][n]["Grabpoint"] = (delta_x + x * math.cos(delta_theta/180*math.pi) + y * math.sin(delta_theta/180*math.pi),
                                        delta_y - x * math.sin(delta_theta/180*math.pi) + y * math.cos(delta_theta/180*math.pi))
        part["Grab"][n]["Theta"] += -delta_theta  # the grab is counter-clockwise, the data is clockwise

        # Refine the grabpoint and effect list according to LC-01 and LC-02 limits
        mode = part["Mode"]
        if mode not in ['100', '50', 'small', 'medium'] or grab_num > 1:
            continue
        gb = part["Grab"][n]["Grabpoint"]
        theta = part["Grab"][n]["Theta"]
        effect_list_50 = part["Grab"][n]["List50"]
        effect_list_100 = part["Grab"][n]["List100"]

        # Support for angle 0, 180
        split_x = references["PlatesDict"][device_code][0] - 1600
        if gb[0] >= split_x:
            if mode == 'medium': # default use the left and mid
                if theta < 180: # use the mid and right
                    effect_list_50 = effect_list_50[8:] + effect_list_50[:8]
                    effect_list_100 = effect_list_100[8:12] + effect_list_100[:8] + effect_list_100[20:] + effect_list_100[12:20]
                    gb = (gb[0] - grib_set_length * math.cos(theta / 180 * math.pi),
                          gb[1] - grib_set_length * math.sin(theta / 180 * math.pi))
            else:
                if theta < 180:  # use the right side
                    effect_list_50 = effect_list_50[8:] + effect_list_50[:8]
                    effect_list_100 = effect_list_100[8:12] + effect_list_100[:8] + effect_list_100[20:] + effect_list_100[12:20]
                    gb = (gb[0] - grib_set_length * math.cos(theta/180*math.pi), gb[1] - grib_set_length * math.sin(theta/180*math.pi))
                else:  # use the left side
                    effect_list_50 = effect_list_50[4:] + effect_list_50[:4]
                    effect_list_100 = effect_list_100[4:12] + effect_list_100[:4] + effect_list_100[16:] + effect_list_100[12:16]
                    gb = (gb[0] + grib_set_length * math.cos(theta/180*math.pi), gb[1] + grib_set_length * math.sin(theta/180*math.pi))
        else:
            continue

        # Support for angle 0, 90, 180, 270
        # split_points = references["SplitDict"][device_code]
        # split_up = split_points[0][1]
        # split_down = split_points[1][1]
        # if gb[1] >= split_up:
        #     if mode == 'medium':  # default use the left and mid
        #         if theta <= 180:  # use the right side
        #             effect_list_50 = effect_list_50[8:] + effect_list_50[:8]
        #             effect_list_100 = effect_list_100[8:12] + effect_list_100[:8] + effect_list_100[20:] + effect_list_100[12:20]
        #             gb = (gb[0] - grib_set_length * math.cos(theta / 180 * math.pi),
        #                   gb[1] - grib_set_length * math.sin(theta / 180 * math.pi))
        #     else:  # default use the mid
        #         if theta <= 180:  # use the right side
        #             effect_list_50 = effect_list_50[8:] + effect_list_50[:8]
        #             effect_list_100 = effect_list_100[8:12] + effect_list_100[:8] + effect_list_100[20:] + effect_list_100[12:20]
        #             gb = (gb[0] - grib_set_length * math.cos(theta/180*math.pi), gb[1] - grib_set_length * math.sin(theta/180*math.pi))
        #         else:  # use the left side:
        #             effect_list_50 = effect_list_50[4:] + effect_list_50[:4]
        #             effect_list_100 = effect_list_100[4:12] + effect_list_100[:4] + effect_list_100[16:] + effect_list_100[12:16]
        #             gb = (gb[0] + grib_set_length * math.cos(theta/180*math.pi), gb[1] + grib_set_length * math.sin(theta/180*math.pi))
        # elif gb[1] <= split_down:
        #     if mode == 'medium':
        #         if theta > 180:  # use the right side
        #             effect_list_50 = effect_list_50[8:] + effect_list_50[:8]
        #             effect_list_100 = effect_list_100[8:12] + effect_list_100[:8] + effect_list_100[20:] + effect_list_100[12:20]
        #             gb = (gb[0] - grib_set_length * math.cos(theta / 180 * math.pi),
        #                   gb[1] - grib_set_length * math.sin(theta / 180 * math.pi))
        #     else:  # default use the mid
        #         if theta <= 180:  # use the left side
        #             effect_list_50 = effect_list_50[4:] + effect_list_50[:4]
        #             effect_list_100 = effect_list_100[4:12] + effect_list_100[:4] + effect_list_100[16:] + effect_list_100[12:16]
        #             gb = (gb[0] + grib_set_length * math.cos(theta/180*math.pi), gb[1] + grib_set_length * math.sin(theta/180*math.pi))
        #         else:  # use the right side
        #             effect_list_50 = effect_list_50[8:] + effect_list_50[:8]
        #             effect_list_100 = effect_list_100[8:12] + effect_list_100[:8] + effect_list_100[20:] + effect_list_100[12:20]
        #             gb = (gb[0] - grib_set_length * math.cos(theta/180*math.pi), gb[1] - grib_set_length * math.sin(theta/180*math.pi))
        # else:
        #     continue

        part["Grab"][n]["Grabpoint"] = gb
        part["Grab"][n]["List50"] = effect_list_50
        part["Grab"][n]["List100"] = effect_list_100
        part["Grab"][n]["50"] = to_decimal(effect_list_50)
        part["Grab"][n]["100"] = to_decimal(refine_effect_list(effect_list_100))

        # print(n)
        # print(effect_list_100)
        # print(effect_list_50)
        # print(refine_effect_list(effect_list_100))
        # print(to_decimal(refine_effect_list(effect_list_100)))
        # print(to_decimal(effect_list_50))

    return part


def visualize_grab_plan(data, with_offset=False):
    colors = {
        "50": (255, 0, 0),  # Blue  只使用一组阵列中一行4个直径50吸头的情况为蓝色
        "100": (0, 255, 0),  # Green  只使用一组阵列中一行4个直径100吸头的情况为绿色
        "small": (0, 0, 255),  # Red  只使用一组阵列的情况为红色
        "medium": (240, 32, 160),  # Purple 使用2组阵列的情况为紫色
        "large": (112, 20, 20),  # deep blue 使用3组阵列的情况为深蓝色
    }
    length = 125
    pad = 2000
    for nestID in data.keys():
        img = cv2.imread("./output/sample_{}.png".format(nestID))
        if with_offset:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.pad(img, pad, 'maximum')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output_path = "./dump/grab_vis/{}_offset.png".format(nestID) if with_offset else "./dump/grab_vis/{}.png".format(nestID)
        parts = data[nestID]["Parts"]
        nestHeight = data[nestID]["Height"]
        for n in parts.keys():
            if parts[n]["Grabbability"]:
                for g in parts[n]["Grab"].keys():
                    pt = parts[n]["Grab"][g]["Grabpoint"]
                    pt = (int(pt[0]), int(nestHeight - pt[1]))
                    theta = parts[n]["Grab"][g]["Theta"]
                    mode = parts[n]["Mode"]
                    # ds = (int(pt[0]+length*math.cos(theta/180*math.pi)), int(pt[1]+length*math.sin(theta/180*math.pi)))
                    ds = (int(pt[0]-length*math.cos((theta-90)/180*math.pi)), int(pt[1]+length*math.sin((theta-90)/180*math.pi)))
                    if with_offset:
                        pt = (pt[0] + pad, pt[1] + pad)
                        ds = (ds[0] + pad, ds[1] + pad)
                    cv2.circle(img, pt, 30, colors[mode], -1)
                    cv2.line(img, pt, ds, colors[mode], thickness=10)
        cv2.imwrite(output_path, img)

def visualize_grab_plan_stack(inp_img, data, with_offset=False):
    colors = {
        "50": (255, 0, 0),  # Blue  只使用一组阵列中一行4个直径50吸头的情况为蓝色
        "100": (0, 255, 0),  # Green  只使用一组阵列中一行4个直径100吸头的情况为绿色
        "small": (0, 0, 255),  # Red  只使用一组阵列的情况为红色
        "medium": (240, 32, 160),  # Purple 使用2组阵列的情况为紫色
        "large": (112, 20, 20),  # deep blue 使用3组阵列的情况为深蓝色
    }
    length = 125
    pad = 2000
    for nestID in data.keys():
        img = cv2.imread(inp_img)
        if with_offset:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.pad(img, pad, 'maximum')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        output_path = "./dump/grab_vis/{}_offset.png".format('vis') if with_offset else "./dump/grab_vis/{}.png".format('vis')
        parts = data[nestID]["Parts"]
        nestHeight = data[nestID]["Height"]
        for n in parts.keys():
            if parts[n]["Grabbability"]:
                for g in parts[n]["Grab"].keys():
                    pt = parts[n]["Grab"][g]["Grabpoint"]
                    pt = (int(pt[0]), int(nestHeight - pt[1]))
                    theta = parts[n]["Grab"][g]["Theta"]
                    mode = parts[n]["Mode"]
                    # ds = (int(pt[0]+length*math.cos(theta/180*math.pi)), int(pt[1]+length*math.sin(theta/180*math.pi)))
                    ds = (int(pt[0]-length*math.cos((theta-90)/180*math.pi)), int(pt[1]+length*math.sin((theta-90)/180*math.pi)))
                    if with_offset:
                        pt = (pt[0] + pad, pt[1] + pad)
                        ds = (ds[0] + pad, ds[1] + pad)
                    cv2.circle(img, pt, 30, colors[mode], -1)
                    cv2.line(img, pt, ds, colors[mode], thickness=10)

        img = cv2.resize(src=img, dsize=None, fx=0.25, fy=0.25)
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite("./output/grab_vis.png", img)

if __name__ == "__main__":
    kernel_path = "./weights/gribber_kernels_sort.pth"
    grib = Gribber({'cuda': CUDA}, kernel_path=kernel_path)
    # grib = Gribber({'cuda': CUDA})
    kernels = {
        'large': grib.kernels_large,
        'medium': grib.kernels_medium,
        'small': grib.kernels_small,
        '100': grib.kernels_100,
        '50': grib.kernels_50,
        'c100': grib.circle_100,
        'c50': grib.circle_50
    }
    coords = {
        'large': grib.magnit_coords_large,
        'medium': grib.magnit_coords_medium,
        'small': grib.magnit_coords_small,
        '100': grib.magnit_coords_100,
        '50': grib.magnit_coords_50,
    }

    # piece = cv2.imread("./output/pieces/013.png", 0)
    # piece = cv2.imread("./output/pieces/010.png", 0)
    # piece = cv2.imread("./output/workpiece1.png", 0)
    # piece = cv2.imread("./output/workpiece2.png", 0)
    # piece = cv2.imread("./output/workpiece3.png", 0)
    piece = cv2.imread("output/pieces/001_id001.png", 0)
    # piece = cv2.imread("./output/pieces/000_id001.png", 0)

    areas_and_centers, split, piece, principcal_direction, gravity_center = detect_area_center_and_principal(piece)
    for id, ac in enumerate(areas_and_centers):
        area = ac[0]
        center = ac[1]
        mode = ac[2]
        src = cv2.cvtColor(area.copy(), cv2.COLOR_GRAY2BGR)
        # cv2.circle(src, center, 50, (0, 0, 255), -1)
        # cv2.namedWindow("center", cv2.WINDOW_NORMAL)
        # cv2.imshow("center", src)
        # cv2.waitKey(0)
        index, circles = grab_by_gravity_center(preprocess_torch(area), kernels, mode, grib.rotation_angle, center, id,
                                                split=split)
        theta = index * grib.rotation_angle
        effect_list_100, effect_list_50, center_refined = generate_magnite_effective_list(center,
                                                                                          theta,
                                                                                          circles,
                                                                                          coords,
                                                                                          mode,
                                                                                          id,
                                                                                          split,
                                                                                          r=grib.c2g_med)
        print(center_refined[0], center_refined[1], index*grib.rotation_angle)
        print(effect_list_100)
        print(effect_list_50)
        print(principcal_direction)
        print(refine_effect_list(effect_list_100))
        print(to_decimal(refine_effect_list(effect_list_100)))
        print(to_decimal(effect_list_50))

