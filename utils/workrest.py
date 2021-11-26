import numpy as np
import torch
import cv2

class WorkRest(object):

    default_config = {
        'id': 0,
        'factor_r2p': 0.1,
        'margin': 10,
        'width_real': 2500,  # mm
        'height_real': 1500,
        'depth_real': 300,
        'with_plan': True,
        'threshold': 300 * 300,
        'cuda': False,
        'simple': False,
        'technic': -1,
        'visualize': True,
    }

    def __init__(self, config):
        super(WorkRest, self).__init__()
        self.config = {**self.default_config, **config}

        self.id = self.config['id']
        self.technic = self.config['technic']
        self.factor = self.config['factor_r2p']
        self.margin = self.config['margin']
        self.simple = self.config['simple']
        self.width_real = self.config['width_real']
        self.height_real = self.config['height_real']
        self.width = np.round(self.config['width_real'] * self.factor).astype(int)
        self.height = np.round(self.config['height_real'] * self.factor).astype(int)
        self.depth = np.round(self.config['depth_real'] * self.factor).astype(int)
        self.with_plan = self.config['with_plan']
        self.threshold = self.config['threshold'] * self.factor * self.factor
        self.cuda = self.config['cuda'] and torch.cuda.is_available()
        self.if_vis = self.config['visualize']

        self.mask = torch.zeros([1, 1, self.height, self.width])
        if self.if_vis:
            self.vis = np.zeros([self.config['height_real'], self.config['width_real']], dtype=np.uint8)
        self.isFull = False

        if self.cuda:
            self.mask = self.mask.cuda()

    def place(self):
        pass

    def is_full(self) -> bool:
        return self.isFull

    def set_technic(self, tech):
        self.technic = tech

    def process_kernel(self, kernel):
        factor = np.round(1 / self.factor).astype(int)
        margin = self.margin

        h, w = kernel.shape
        if h % 2 != 0:
            kernel = np.pad(kernel, ((0, 1), (0, 0)))
        if w % 2 != 0:
            kernel = np.pad(kernel, ((0, 0), (0, 1)))
        h, w = kernel.shape

        ht = (np.ceil(h/20.)*20.).astype(int)
        wt = (np.ceil(w/20.)*20.).astype(int)

        kernel = np.pad(kernel, (((ht-h)//2+margin, (ht-h)//2+margin), ((wt-w)//2+margin, (wt-w)//2+margin)))
        h, w = kernel.shape
        kernel_vis = kernel.copy()
        kernel = cv2.resize(kernel, (w//factor, h//factor))
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0) / 255.
        if self.cuda:
            kernel = kernel.cuda()
        return kernel, kernel_vis

    def is_fit(self, kernel: np.ndarray) -> (bool, (int, int), bool):
        factor = np.round(1 / self.factor).astype(int)

        fit = False
        coord = (-1, -1)  ## The coord in the workrest reference, x right, y down
        rot = False  ## rotate 90 degrees
        coord_mask = (-1, -1)

        kernel, kernel_vis = self.process_kernel(kernel)

        if self.simple:
            _, _, h, w = kernel.shape
            if h <= self.height_real and w <= self.width_real:
                fit = True
                coord = (self.width_real // 2, self.height_real // 2)
                vh, vw = kernel_vis.shape
                if self.if_vis:
                    self.vis[coord[1] - vh // 2:coord[1] + vh // 2,
                    coord[0] - vw // 2:coord[0] + vw // 2] = kernel_vis.copy()
            elif h <= self.width_real and w <= self.height_real:
                fit = True
                coord = (self.width_real // 2, self.height_real // 2)
                rot = True
                kernel_vis = np.flip(kernel_vis.T, 1)
                vh, vw = kernel_vis.shape
                if self.if_vis:
                    self.vis[coord[1] - vh // 2:coord[1] + vh // 2,
                    coord[0] - vw // 2:coord[0] + vw // 2] = kernel_vis.copy()
            coord = self.refine_coord(kernel_vis, coord)
            return fit, coord, rot, kernel_vis


        if kernel.shape[2] > self.mask.shape[2] or kernel.shape[3] > self.mask.shape[3]:
            if kernel.shape[3] <= self.mask.shape[2] or kernel.shape[2] <= self.mask.shape[3]:
                kernel = kernel.transpose(2, 3)  # rotate 90 degrees
                kernel = torch.flip(kernel, [3])
                kernel_vis = np.flip(kernel_vis.T, 1)
                rot = True
            else:
                return fit, coord, rot, kernel_vis

        result = torch.conv2d(self.mask, kernel, stride=(1, 1), padding=0)
        indexes = torch.where(result == 0.)
        if len(indexes[0]) != 0:
            fit = True
            _, _, h, w = kernel.shape
            coord_mask = ((indexes[3][0] + w // 2), (indexes[2][0] + h // 2))  # (x, y)
            coord = ((indexes[3][0] + w // 2) * factor, (indexes[2][0] + h // 2) * factor)  # (x, y)
        else:
            kernel = kernel.transpose(2, 3)  # rotate 90 degrees
            kernel = torch.flip(kernel, [3])
            kernel_vis = np.flip(kernel_vis.T, 1)
            rot = True
            if kernel.shape[2] <= self.mask.shape[2] and kernel.shape[3] <= self.mask.shape[3]:
                result = torch.conv2d(self.mask, kernel, stride=(1, 1), padding=0)
                indexes = torch.where(result == 0.)
                if len(indexes[0]) != 0:
                    fit = True
                    _, _, h, w = kernel.shape
                    coord_mask = ((indexes[3][0] + w // 2), (indexes[2][0] + h // 2))  # (x, y)
                    coord = ((indexes[3][0] + w // 2) * factor, (indexes[2][0] + h // 2) * factor)  # (x, y)

        if fit:
            _, _, h, w = kernel.shape
            self.mask[:, :, coord_mask[1]-h//2:coord_mask[1]+h//2, coord_mask[0]-w//2:coord_mask[0]+w//2] = torch.ones_like(self.mask[:, :, coord_mask[1]-h//2:coord_mask[1]+h//2, coord_mask[0]-w//2:coord_mask[0]+w//2])
            vh, vw = kernel_vis.shape
            if self.if_vis:
                # print(self.vis[coord[1]-vh//2:coord[1]+vh//2, coord[0]-vw//2:coord[0]+vw//2].shape)
                self.vis[coord[1]-vh//2:coord[1]+vh//2, coord[0]-vw//2:coord[0]+vw//2] = kernel_vis.copy()

            coord = (coord[0].cpu().numpy(), coord[1].cpu().numpy()) if self.cuda else (coord[0].numpy(), coord[1].numpy())
            coord = self.refine_coord(kernel_vis, coord)  # refine the coordinate to the gravity center

            if (self.height * self.width - torch.sum(self.mask)) < self.threshold:
                self.isFull = True

        return fit, coord, rot, kernel_vis  # rot 90 degrees

    def calculate_gravity_center(self, c):
        center_x = np.round(c['m10'] / c['m00']).astype(int)
        center_y = np.round(c['m01'] / c['m00']).astype(int)
        return center_x, center_y

    def refine_coord(self, kernel_vis, coord):
        vh, vw = kernel_vis.shape
        c = cv2.moments(kernel_vis, binaryImage=True)
        gc_x, gc_y = self.calculate_gravity_center(c)
        cc_x, cc_y = vw//2, vh//2
        coord = (coord[0] + (gc_x - cc_x), coord[1] + (gc_y - cc_y))
        return coord

    def visualize(self):
        if self.if_vis:
            cv2.imwrite("./output/workrest_result_tech{}_width{}_id{}.png".format(self.technic, self.width_real, self.id), self.vis)

def choose_workrest_size(piece):
    h, w = piece.shape

    dmax, dmin = max(h, w), min(h, w)
    if dmin <= 1500 and dmax <= 2300:
        return 0
    elif dmin <= 2300 and dmax <= 5600:
        return 1
    elif dmin <= 5600 and dmax <= 11500:
        return 2
    else:
        raise ValueError("The workpiece size doesn't make sense!")


if __name__ == '__main__':
    CUDA = False

    def detect_area_and_center(piece: np.ndarray):
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

        return img

    workrestsS = {
        1: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 1})],
        2: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 2})],
        3: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 3})],
        4: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 4})],
        5: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 5})]
    }

    workrestsM = {
        1: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 1})],
        2: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 2})],
        3: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 3})],
        4: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 4})],
        5: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 5})]
    }

    workrestsL = {
        1: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'id': 0, 'technic': 1, 'simple': True})],
        2: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'id': 0, 'technic': 2, 'simple': True})],
        3: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'id': 0, 'technic': 3, 'simple': True})],
        4: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'id': 0, 'technic': 4, 'simple': True})],
        5: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'id': 0, 'technic': 5, 'simple': True})]
    }

    workrests = {
        0: workrestsS,
        1: workrestsM,
        2: workrestsL
    }

    pieces = []
    technics = []
    for i in range(3):
        pieces += [detect_area_and_center(cv2.imread("./output/workpiece1.png", 0)) for _ in range(2)]
        technics += [1, 2]
        pieces += [detect_area_and_center(cv2.imread("./output/workpiece2.png", 0)) for _ in range(2)]
        technics += [2, 3]
        pieces += [detect_area_and_center(cv2.imread("./output/workpiece3.png", 0)) for _ in range(2)]
        technics += [3, 4]

    zipped = list(zip(pieces, technics))
    zipped.sort(key=lambda x: np.sum(x[0]), reverse=True)

    for p, tech in zipped:
        t = choose_workrest_size(p)
        width = 2500
        if t == 0:
            width = 2500
        elif t == 1:
            width = 6000
        elif t == 2:
            width = 12000

        fit = False
        coord = (-1, -1)
        rot = False

        # check if the current workrests would do the work?
        id = 0
        while workrests[t][tech][id].is_full() and id < len(workrests[t][tech]):
            id += 1
        while id >= len(workrests[t][tech]):
            workrests[t][tech].append(WorkRest({'cuda': CUDA, 'width_real': width, 'id': len(workrests[t][tech]), 'technic': tech}))

        while not fit:
            fit, coord, rot, vis = workrests[t][tech][id].is_fit(p)
            if fit:
                break
            else:
                id += 1
                while id >= len(workrests[t][tech]):
                    workrests[t][tech].append(
                        WorkRest({'cuda': CUDA, 'width_real': width, 'id': len(workrests[t][tech]), 'technic': tech}))

        print(workrests[t][tech][id].technic, workrests[t][tech][id].id, fit, coord, rot)
        workrests[t][tech][id].visualize()
