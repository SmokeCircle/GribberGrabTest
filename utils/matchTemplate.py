import os
import argparse
import cv2
try:
    from utils.Saliency.inference import run_inference
except:
    pass
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters.')
    # for salienct detection
    parser.add_argument('--imgs_folder', default='/data/Salient_Pytorch/data/workpieces_demo/O211010SG10006A05-2021-10-13-01-07-31-448.bmp', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='/data/Salient_Pytorch/models/alph-0.7_wbce_w0-1.0_w1-1.15/weights/best-model_epoch-140_mae-0.0020_loss-0.0082.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=512, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)
    parser.add_argument('--use_SOD', action="store_true", default=False, help='whether to generate mask yaml file thourgh saliency detecion')

    # for match template
    parser.add_argument('--debug', default=0, help='if show debug visualization (1 or 0)', type=int)
    parser.add_argument('--tmpl_num', default=6, help='template num to be used', type=int)
    parser.add_argument('--max_match', default=1, help='choose the top N result', type=int)
    parser.add_argument('--threshold', default=60, help='match threshold', type=int)
    parser.add_argument('--resize', default=0.5, help='image resize', type=float)
    parser.add_argument('--blur', default=5, help='image blur (odd value only)', type=int)
    parser.add_argument('--yaml', default='./data/yaml/mask.yaml', help='yaml file saved by Saliency Detection module', type=str)
    return parser.parse_args()


def _transform(src, angle, scale):
    cols, rows = src.shape[:2]
    edge = max(cols, rows) - min(cols, rows)
    padded_src = cv2.copyMakeBorder(src, edge,edge,edge,edge, cv2.BORDER_CONSTANT, value=0)
    cols, rows = padded_src.shape[:2]

    rot_mat = cv2.getRotationMatrix2D((cols*0.5, rows*0.5), angle, scale)
    rot_img = cv2.warpAffine(padded_src, rot_mat, (cols, rows))
    [x, y, w, h] = cv2.boundingRect(rot_img)
    rot_img = rot_img[y:y+h, x:x+w]
    _, dst = cv2.threshold(rot_img, 180, 255, cv2.THRESH_BINARY)
    return dst


class Matcher:
    def __init__(self, args):
        self.args = args
        self.prefix = "/data/GribberGrabTest"
        self.exe = os.path.join(self.prefix, "utils/linemod-2D", "build/shape_based_matching_test")
    def get_img_size(self):
        img = cv2.imread(self.args.imgs_folder, cv2.IMREAD_COLOR)
        #if img.shape[0] > img.shape[1]:  # if H > W, it's a vertical image
        #    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # need to rotate to be a lie-down image
        return img.shape[:2]  # necessarily H < W

    def get_salient_mask(self):
        run_inference(self.args)

    def train(self, path):
        """

        :param path: path of training image dir
        :return:
        """
        cmdline = "{} --mode {} --filename {} --resize {} --blur {} --vis {} --nest {}".format(
            self.exe, 'train', path, self.args.resize, self.args.blur,
            self.args.debug, self.args.groupID)
        os.system(cmdline)

    def match(self):
        res = []
        cmdline = "{} --yaml {} --resize {} --blur {} -t {} -n {} -x {} --debug {} --nest {}".format(
            self.exe, self.args.yaml, self.args.resize, self.args.blur,
            self.args.threshold, self.args.tmpl_num, self.args.max_match,
            self.args.debug, self.args.groupID)
        #print(cmdline)
        output = os.popen(cmdline)
        for line in output.readlines():
            print(line, end="")
            if not self.args.debug:
                if "match" in line:
                    line = line.strip().split(' ')
                    res.append({"id": int(line[2].split('_')[-1]), "x": int(line[4]), "y": int(line[6]),
                                "angle": int(line[8]), "scale": float(line[10]), "similarity": float(line[12])})
        return res

    def src_of(self, info):
        """
        :param info: dict with match.x/y, match.scale, match.angle...
        :return: transformed binary image array, PartSN
        """
        path = os.path.join(self.prefix, 'data/training_img', self.args.groupID)
        yamlpath = os.path.join(path, 'piece_id.yaml')
        with open(yamlpath, 'r') as f:
            piece_id_dict = yaml.load(f, Loader=yaml.FullLoader)
        piece_id = info["id"]
        img_name = piece_id_dict[piece_id]
        filename = os.path.join(path, img_name)
        src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return _transform(src, info["angle"], info["scale"]), img_name.split(".")[0]

    def get_tmpl_num(self):
        path = os.path.join(self.prefix, 'data/training_img', self.args.groupID, 'piece_id.yaml')
        with open(path, 'rb') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        return len(data.keys())
if __name__=="__main__":
    from Saliency.inference import run_inference
    args = parse_arguments()
    m = Matcher(args)
    if args.use_SOD:
        m.get_salient_mask()
    m.match()
    print("done")
