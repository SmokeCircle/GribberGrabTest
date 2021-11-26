import numpy as np
import torch
import cv2

class Conveyor(object):

    default_config = {
        'id': 0,
        'factor_r2p': 0.1,
        'margin': 100,
        'width_real': 10000,  # mm, 传送带的长度
        'height_real': 1900,  # mm, 传送带的宽度
        'depth_real': 300,
        'with_plan': True,
        'threshold': 300 * 300,
        'cuda': False,
        'visualize': True,
    }

    def __init__(self, config):
        super(Conveyor, self).__init__()
        self.config = {**self.default_config, **config}

        self.id = self.config['id']
        self.factor = self.config['factor_r2p']
        self.margin = self.config['margin']
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
        coord = (-1, -1)  ## The coord in the conveyor frame
        rot = False  ## rotate 90 degrees

        kernel, kernel_vis = self.process_kernel(kernel)

        if kernel.shape[2] > kernel.shape[3]:
            rot = True
            kernel = kernel.transpose(2, 3)  # rotate 90 degrees
            kernel = torch.flip(kernel, [3])
            kernel_vis = np.flip(kernel_vis.T, 1)

        if kernel.shape[2] > self.mask.shape[2] or kernel.shape[3] > self.mask.shape[3]:
            print("Warning:: The workpiece size {} exceeds the conveyor size {}".format(
                (kernel.shape[2], kernel.shape[3]), (self.mask.shape[2], self.mask.shape[3])))
            return fit, coord, rot, kernel_vis

        result = torch.conv2d(self.mask, kernel, stride=(1, 1), padding=0)
        indexes = torch.where(result == 0.)

        if len(indexes[0]) != 0:
            fit = True
            _, _, h, w = kernel.shape
            coord_mask = ((indexes[3][0] + w // 2), (indexes[2][0] + h // 2))  # (x, y)
            coord = ((indexes[3][0] + w // 2) * factor, (indexes[2][0] + h // 2) * factor)  # (x, y)
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

    def visualize(self, nestID):
        if self.if_vis:
            cv2.imwrite("./dump/conveyor_vis/conveyor_id{}_nestID={}.png".format(self.id, nestID), self.vis)

if __name__ == '__main__':
    pass
