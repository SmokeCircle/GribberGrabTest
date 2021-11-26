import yaml
import cv2
import numpy as np
import argparse
import os
def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--image_path', help='Path to folder containing images', type=str)
    parser.add_argument('--mask_path', help='Path to folder containing images', type=str)
    return parser.parse_args()

class MaskDetector:
    def __init__(self, mask, resolution=256, pth=None):
        self.mask = mask
        self.resolution = resolution
        self.pth = pth

        inp_img = cv2.imread(pth)  # original input image
        h, w, c = inp_img.shape
        self.orig_size = (w, h)
        size = max(h, w)
        self.size = size
        self.padding_h = (size - h) // 2
        self.padding_w = (size - w) // 2

    def write_yaml(self, path=None):
        nccomps = cv2.connectedComponentsWithStats(image=self.mask)
        labels, status, centroids = nccomps[1], nccomps[2], nccomps[3]
        print("all connected components nums:", len(status)-1)
        stats = []

        #orig_img = cv2.imread(self.pth)
        for i in range(1, len(status)):
            if status[i][-1] > 100:
                x = status[i][0] * self.size / self.resolution - self.padding_w
                y = status[i][1] * self.size / self.resolution - self.padding_h
                w = status[i][2] * self.size / self.resolution
                h = status[i][3] * self.size / self.resolution
                piece_id = 0
                stats.append({"x":int(x), "y":int(y), "w":int(w), "h":int(h), "piece_id":piece_id})

                #cv2.rectangle(orig_img, (int(x), int(y)), (int(x+w), int(y+h)), (255,0,0))

        #cv2.imwrite("./orig.png", orig_img)
        if path==None:
            yaml_path = os.path.abspath(self.pth).split('.')[0] + '.yaml'
        else:
            yaml_path = path
        print("saving infos to", os.path.abspath(yaml_path))
        with open(yaml_path, 'w', encoding="utf-8") as f:
            masked_path = os.path.abspath(self.pth)[:-4]+"_masked.png"
            yaml.dump({"nums":len(stats), "infos":stats, "path":masked_path}, f)

    def write_img(self, path=None):
        # find original input image in the padded-resized mask
        x = self.padding_w / self.size * self.resolution
        y = self.padding_h / self.size * self.resolution
        w = self.orig_size[0] / self.size * self.resolution
        h = self.orig_size[1] / self.size * self.resolution
        print("x,y,w,h,shape",x,y,w,h,self.mask.shape)
        resized_inp_img = self.mask[int(y):int(y+h),int(x):int(x+w)]
        mask_orig = cv2.resize(resized_inp_img, self.orig_size)
        print('saving salient mask image to', os.path.abspath(path))
        cv2.imwrite(path, mask_orig*255)


if __name__ == "__main__":
    args = parse_arguments()
    mask = cv2.imread(args.mask_path, cv2.IMREAD_UNCHANGED)[:, :, 1]
    M = MaskDetector(mask, 256, args.image_path)
    M.write_yaml()
    cv2.imshow("mask", mask)
    cv2.waitKey(0)