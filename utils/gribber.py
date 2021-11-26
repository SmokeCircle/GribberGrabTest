import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch

StackGribberConfig = {
    'set_x': 3,
    'set_y': 1,
    'space_set_x_real': 495,
    'space_set_y_real': 0,
    'height_set_real': 350,
    'width_set_real': 780,
    'row_100_real': 2,
    'col_100_real': 4,
    'radius_100_real': 50,
    'space_100_x_real': [0, 280, 120, 280],
    'space_100_y_real': 250,
    'origin_100_real': (50, 50),  # (x, y)
    'row_50_real': 1,
    'col_50_real': 4,
    'radius_50_real': 25,
    'space_50_x_real': [0, 280, 120, 280],
    'space_50_y_real': 0,
    'origin_50_real': (50, 200),  # (50, 150),  # (x, y)
    'factor_r2p': 1,
    'width_real': 3330,
    'height_real': 350,
    # 'width_pixel': 3500,
    # 'height_pixel': 350,
    # 'rotation_angle': 10,
    'rotation_angle': 180,  # counter-clockwise
    'margin': 20,
    'cuda': False,
    'task': "stack"
}

SortGribberConfig = {
    'set_x': 3,
    'set_y': 1,
    'space_set_x_real': 365,
    'space_set_y_real': 0,
    'height_set_real': 350,
    'width_set_real': 810,
    'row_100_real': 2,
    'col_100_real': 4,
    'radius_100_real': 50,
    'space_100_x_real': [0, 280, 150, 280],
    'space_100_y_real': 250,
    'origin_100_real': (50, 50),  # (x, y)
    'row_50_real': 1,
    'col_50_real': 4,
    'radius_50_real': 25,
    'space_50_x_real': [0, 280, 150, 280],
    'space_50_y_real': 0,
    'origin_50_real': (50, 200),  # (50, 150),  # (x, y)
    'factor_r2p': 1,
    'width_real': 3160,
    'height_real': 350,
    # 'width_pixel': 3500,
    # 'height_pixel': 350,
    # 'rotation_angle': 10,
    'rotation_angle': 180,  # counter-clockwise
    'margin': 20,
    'cuda': False,
    'task': "sort"
}


class Gribber(object):

    default_config = {
        'set_x': 3,
        'set_y': 1,
        'space_set_x_real': 365,
        'space_set_y_real': 0,
        'height_set_real': 350,
        'width_set_real': 810,
        'row_100_real': 2,
        'col_100_real': 4,
        'radius_100_real': 50,
        'space_100_x_real': [0, 280, 150, 280],
        'space_100_y_real': 250,
        'origin_100_real': (50, 50),  # (x, y)
        'row_50_real': 1,
        'col_50_real': 4,
        'radius_50_real': 25,
        'space_50_x_real': [0, 280, 150, 280],
        'space_50_y_real': 0,
        'origin_50_real': (50, 200),  #(50, 150),  # (x, y)
        'factor_r2p': 1,
        'width_real': 3160,
        'height_real': 350,
        # 'width_pixel': 3500,
        # 'height_pixel': 350,
        # 'rotation_angle': 10,
        'rotation_angle': 90,  # counter-clockwise
        'margin': 20,
        'cuda': True,
        'task': "sort"
    }

    def __init__(self, config, kernel_path=None):
        super(Gribber, self).__init__()
        self.config = {**self.default_config, **config}
        self.check_config()
        self.kernel_path = kernel_path
        self.factor = self.config['factor_r2p']
        self.margin = self.config['margin']
        self.rotation_angle = self.config['rotation_angle']
        self.config['width_pixel'] = np.round(self.config['width_real'] * self.factor).astype(int)
        self.config['height_pixel'] = np.round(self.config['height_real'] * self.factor).astype(int)

        self.config['width_set_pixel'] = np.round(self.config['width_set_real'] * self.factor).astype(int)
        self.config['height_set_pixel'] = np.round(self.config['height_set_real'] * self.factor).astype(int)

        self.config['space_set_x_pixel'] = np.round(self.config['space_set_x_real'] * self.factor).astype(int)
        self.config['space_set_y_pixel'] = np.round(self.config['space_set_y_real'] * self.factor).astype(int)

        self.radius_100 = np.round(self.config['radius_100_real'] * self.factor).astype(int)
        self.radius_50 = np.round(self.config['radius_50_real'] * self.factor).astype(int)

        height, width = self.config['height_pixel'], np.round(self.config['width_set_real']*self.factor).astype(int)
        self.mask_small = np.zeros([height+height%2, width+width%2])

        height, width = self.config['height_pixel'], np.round((2*self.config['width_set_real']+self.config['space_set_x_pixel'])*self.factor).astype(int)
        self.mask_medium = np.zeros([height+height%2, width+width%2])

        height, width = self.config['height_pixel'], self.config['width_pixel']
        self.mask_large = np.zeros([height+height%2, width+width%2])

        height, width = self.radius_100*2, self.config['width_set_pixel']
        self.mask_100 = np.zeros([height+height%2, width+width%2])

        height, width = self.radius_50*2, self.config['width_set_pixel']
        self.mask_50 = np.zeros([height+height%2, width+width%2])

        self.magnit_coords_small = {}  # store the coordinates of the magnit in the original reference, L for 100, S for 50, (x, y)
        self.magnit_coords_medium = {}  # store the coordinates of the magnit in the original reference, L for 100, S for 50, (x, y)
        self.magnit_coords_large = {}  # store the coordinates of the magnit in the original reference, L for 100, S for 50, (x, y)
        self.magnit_coords_100 = {}  # store the coordinates of the magnit in the original reference, L for 100, (x, y)
        self.magnit_coords_50 = {}  # store the coordinates of the magnit in the original reference, S for 50, (x, y)

        self.c2g_med = (self.config['width_set_real'] + self.config['space_set_x_real']) / 2 * self.factor
        self.height = self.config['height_real']
        self.width = self.config['width_real']

        self.cuda = self.config['cuda'] & torch.cuda.is_available()
        self.configure()

    def configure(self):

        # draw 100 mask
        x0, y0 = self.config['origin_100_real'][0], self.radius_100
        x0, y0 = np.round(x0 * self.factor).astype(int), np.round(y0 * self.factor).astype(int)
        for i in range(self.config['col_100_real']):
            center_x = np.round(x0 + sum(self.config['space_100_x_real'][:i + 1]) * self.factor).astype(int)
            center_y = np.round(y0).astype(int)
            cv2.circle(self.mask_100, (center_x, center_y), self.radius_100, 255, -1)
            self.magnit_coords_100['Lr00c{0:02d}'.format(i)] = (center_x, center_y)

        # draw 50 mask
        x0, y0 = self.config['origin_50_real'][0], self.radius_50
        x0, y0 = np.round(x0 * self.factor).astype(int), np.round(y0 * self.factor).astype(int)
        for i in range(self.config['col_50_real']):
            center_x = np.round(x0 + sum(self.config['space_50_x_real'][:i + 1]) * self.factor).astype(int)
            center_y = np.round(y0).astype(int)
            cv2.circle(self.mask_50, (center_x, center_y), self.radius_50, 255, -1)
            self.magnit_coords_50['Sr00c{0:02d}'.format(i)] = (center_x, center_y)

        # draw set mask 100
        x0, y0 = self.config['origin_100_real']
        x0, y0 = np.round(x0*self.factor).astype(int), np.round(y0*self.factor).astype(int)
        for i in range(self.config['row_100_real']):
            for j in range(self.config['col_100_real']):
                center_x = np.round(x0 + sum(self.config['space_100_x_real'][:j+1])*self.factor).astype(int)
                center_y = np.round(y0 + i*self.config['space_100_y_real']*self.factor).astype(int)
                cv2.circle(self.mask_small, (center_x, center_y), self.radius_100, 255, -1)
                self.magnit_coords_small['Lr{0:02d}c{1:02d}'.format(i, j)] = (center_x, center_y)
                for k in range(self.config['set_x']-1):
                    self.magnit_coords_medium['Lr{0:02d}c{1:02d}'.format(i, j+k*self.config['col_100_real'])] = (center_x+k*(self.config['space_set_x_pixel']+self.config['width_set_real']), center_y)
                for k in range(self.config['set_x']):
                    self.magnit_coords_large['Lr{0:02d}c{1:02d}'.format(i, j+k*self.config['col_100_real'])] = (center_x+k*(self.config['space_set_x_pixel']+self.config['width_set_real']), center_y)

        # draw set mask 50
        x0, y0 = self.config['origin_50_real']
        x0, y0 = np.round(x0*self.factor).astype(int), np.round(y0*self.factor).astype(int)
        for i in range(self.config['row_50_real']):
            for j in range(self.config['col_50_real']):
                center_x = np.round(x0 + sum(self.config['space_50_x_real'][:j+1])*self.factor).astype(int)
                center_y = np.round(y0 + i*self.config['space_50_y_real']*self.factor).astype(int)
                cv2.circle(self.mask_small, (center_x, center_y), self.radius_50, 255, -1)
                self.magnit_coords_small['Sr{0:02d}c{1:02d}'.format(i, j)] = (center_x, center_y)
                for k in range(self.config['set_x']-1):
                    self.magnit_coords_medium['Sr{0:02d}c{1:02d}'.format(i, j+k*self.config['col_50_real'])] = (center_x+k*(self.config['space_set_x_pixel']+self.config['width_set_real']), center_y)
                for k in range(self.config['set_x']):
                    self.magnit_coords_large['Sr{0:02d}c{1:02d}'.format(i, j+k*self.config['col_50_real'])] = (center_x+k*(self.config['space_set_x_pixel']+self.config['width_set_real']), center_y)

        # draw 2set mask (medium)
        x0, y0 = 0, 0
        delta_y = np.round(self.config['height_set_real']*self.factor).astype(int)
        delta_x = np.round(self.config['width_set_real'] * self.factor).astype(int)
        for i in range(self.config['set_y']):
            for j in range(self.config['set_x']-1):
                y = np.round(y0+i*self.config['space_set_y_real']*self.factor).astype(int)
                x = np.round(x0+j*(self.config['space_set_x_real']*self.factor+delta_x)).astype(int)
                self.mask_medium[y:y+delta_y, x:x+delta_x] = self.mask_small.copy()

        # draw whole mask (large)
        x0, y0 = 0, 0
        delta_y = np.round(self.config['height_set_real']*self.factor).astype(int)
        delta_x = np.round(self.config['width_set_real'] * self.factor).astype(int)
        for i in range(self.config['set_y']):
            for j in range(self.config['set_x']):
                y = np.round(y0+i*self.config['space_set_y_real']*self.factor).astype(int)
                x = np.round(x0+j*(self.config['space_set_x_real']*self.factor+delta_x)).astype(int)
                self.mask_large[y:y+delta_y, x:x+delta_x] = self.mask_small.copy()

        pad_large = (self.config['width_pixel'] - self.config['height_pixel']) // 2
        self.mask_large = np.pad(self.mask_large, ((pad_large+self.margin, pad_large+self.margin), (self.margin, self.margin)))
        for key in self.magnit_coords_large.keys():
            self.magnit_coords_large[key] = (self.magnit_coords_large[key][0]+self.margin, self.magnit_coords_large[key][1]+self.margin+pad_large)
            
        pad_medium = ((2*self.config['width_set_real']+self.config['space_set_x_pixel']) - self.config['height_pixel']) // 2
        self.mask_medium = np.pad(self.mask_medium, ((pad_medium+self.margin, pad_medium+self.margin), (self.margin, self.margin)))
        for key in self.magnit_coords_medium.keys():
            self.magnit_coords_medium[key] = (self.magnit_coords_medium[key][0]+self.margin, self.magnit_coords_medium[key][1]+self.margin+pad_medium)

        pad_small = (self.config['width_set_pixel'] - self.config['height_set_pixel']) // 2
        self.mask_small = np.pad(self.mask_small, ((pad_small+self.margin, pad_small+self.margin), (self.margin, self.margin)))
        for key in self.magnit_coords_small.keys():
            self.magnit_coords_small[key] = (self.magnit_coords_small[key][0]+self.margin, self.magnit_coords_small[key][1]+self.margin+pad_small)

        pad_100 = (self.config['width_set_pixel'] - self.radius_100*2) // 2
        self.mask_100 = np.pad(self.mask_100, ((pad_100+self.margin, pad_100+self.margin), (self.margin, self.margin)))
        for key in self.magnit_coords_100.keys():
            self.magnit_coords_100[key] = (self.magnit_coords_100[key][0]+self.margin, self.magnit_coords_100[key][1]+self.margin+pad_100)

        pad_50 = (self.config['width_set_pixel'] - self.radius_50*2) // 2
        self.mask_50 = np.pad(self.mask_50, ((pad_50+self.margin, pad_50+self.margin), (self.margin, self.margin)))
        for key in self.magnit_coords_50.keys():
            self.magnit_coords_50[key] = (self.magnit_coords_50[key][0]+self.margin, self.magnit_coords_50[key][1]+self.margin+pad_50)

        # Configure circle doublecheck kernels
        self.circle_100 = self.generate_circle_kernel(50)
        self.circle_50 = self.generate_circle_kernel(25)

        if self.kernel_path:
            kernels = torch.load(self.kernel_path)
            self.kernels_large = kernels['kernels_large']
            self.kernels_medium = kernels['kernels_medium']
            self.kernels_small = kernels['kernels_small']
            self.kernels_100 = kernels['kernels_100']
            self.kernels_50 = kernels['kernels_50']
            self.circle_50 = kernels['circle_50']
            self.circle_100 = kernels['circle_100']
        else:
            kernels_large = self.generate_kernel_rotation(self.mask_large)
            kernels_medium = self.generate_kernel_rotation(self.mask_medium)
            kernels_small = self.generate_kernel_rotation(self.mask_small)
            kernels_100 = self.generate_kernel_rotation(self.mask_100)
            kernels_50 = self.generate_kernel_rotation(self.mask_50)
            circle_100 = self.generate_circle_kernel(self.radius_100)
            circle_50 = self.generate_circle_kernel(self.radius_50)
            self.kernels_small = torch.Tensor(kernels_small)/255.
            self.kernels_medium = torch.Tensor(kernels_medium)/255.
            self.kernels_large = torch.Tensor(kernels_large)/255.
            self.kernels_100 = torch.Tensor(kernels_100)/255.
            self.kernels_50 = torch.Tensor(kernels_50)/255.
            self.circle_100 = torch.Tensor(circle_100)/255.
            self.circle_50 = torch.Tensor(circle_50)/255.
            kernels = {"kernels_small": self.kernels_small,
                       "kernels_medium": self.kernels_medium,
                       "kernels_large": self.kernels_large,
                       "kernels_50": self.kernels_50,
                       "kernels_100": self.kernels_100,
                       "circle_100": self.circle_100,
                       "circle_50": self.circle_50,}
            torch.save(kernels, "./weights/gribber_kernels_{}.pth".format(self.config['task']))

        if self.cuda:
            self.kernels_small = self.kernels_small.cuda()
            self.kernels_medium = self.kernels_medium.cuda()
            self.kernels_large = self.kernels_large.cuda()
            self.kernels_100 = self.kernels_100.cuda()
            self.kernels_50 = self.kernels_50.cuda()

        # self.visualize()

    def generate_circle_kernel(self, radius):
        kernel = np.zeros([2*radius+1, 2*radius+1], dtype=np.uint8)
        kernel = cv2.circle(kernel, (radius, radius), radius, 255, -1)
        return kernel

    def generate_rotation_matrices(self, h, w):
        center_x = w // 2
        center_y = h // 2  # same size
        matrices = [cv2.getRotationMatrix2D((center_x, center_y), i*self.config['rotation_angle'], 1.0) for i in range(360//self.config['rotation_angle'])]
        return matrices

    def generate_kernel_rotation(self, mask):
        h, w = mask.shape
        rot_matrices = self.generate_rotation_matrices(h, w)
        kernels = [cv2.warpAffine(mask, m, (w, h)) for m in rot_matrices]
        # print(len(kernels))
        return kernels

    def visualize(self):
        plt.figure(0)
        plt.imshow(self.mask_small)
        plt.figure(1)
        plt.imshow(self.mask_large)
        plt.figure(2)
        plt.imshow(self.mask_50)
        plt.figure(3)
        plt.imshow(self.mask_100)
        plt.figure(4)
        plt.imshow(self.kernels_small[1].cpu().numpy()*255.)
        # plt.figure(5)
        # plt.subplot(2, 2, 1)
        # plt.imshow(self.kernels_large[0])
        # plt.subplot(2, 2, 2)
        # plt.imshow(self.kernels_large[1])
        # plt.subplot(2, 2, 3)
        # plt.imshow(self.kernels_large[2])
        # plt.subplot(2, 2, 4)
        # plt.imshow(self.kernels_large[3])
        # plt.figure(6)
        # plt.subplot(2, 2, 1)
        # plt.imshow(self.kernels_small[0])
        # plt.subplot(2, 2, 2)
        # plt.imshow(self.kernels_small[1])
        # plt.subplot(2, 2, 3)
        # plt.imshow(self.kernels_small[2])
        # plt.subplot(2, 2, 4)
        # plt.imshow(self.kernels_small[3])
        plt.show()

    def check_config(self):
        pass
        # if self.config['origin_real'][0] - self.config['radius_real'] < 0:
        #     raise ValueError
        # elif self.config['origin_real'][0] + self.config['space_x_real'] * (self.config['col_real']-1) +\
        #         self.config['radius_real'] > self.config['width_real']:
        #     raise ValueError
        # elif self.config['origin_real'][1] - self.config['radius_real'] < 0:
        #     raise ValueError
        # elif self.config['origin_real'][1] + self.config['space_y_real'] * (self.config['row_real']-1) +\
        #         self.config['radius_real'] > self.config['height_real']:
        #     raise ValueError
        # else:
        #     print("DEBUG:: Gribber Configuring")

if __name__ == '__main__':
    # kernel_path = "./weights/gribber_kernels.pth"
    # grib = Gribber({'cuda': False}, kernel_path=kernel_path)
    # grib = Gribber({'cuda': False})
    grib = Gribber(SortGribberConfig)
    grib.visualize()
    del grib
    grib = Gribber(StackGribberConfig)
    grib.visualize()

    # keys = list(grib.magnit_coords_large.keys())
    # print(keys)
    # keys.sort()
    # print(keys)
    # coords = []
    # for k in keys:
    #     coords.append(np.array(grib.magnit_coords_large[k]))
    # coords = np.array(coords)
    # print(coords)
    #
    # center = np.array([68, 1747])
    # center = center[None, ...]
    # centers = np.repeat(center, coords.shape[0], axis=0)
    #
    # dist = np.sum(np.abs(centers - coords), axis=-1)
    # index = dist.argmin(axis=0)
    # min_dist = dist.min(axis=0)
    # print(dist)



