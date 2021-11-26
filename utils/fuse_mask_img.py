import cv2
import numpy as np

def fuse_mask_img(mask_path, inp_img_path, vis = False, output_path=None):
    mask = cv2.imread(mask_path)
    inp_img = cv2.imread(inp_img_path)
    assert mask.shape == inp_img.shape

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    rgb = int(np.mean(inp_img[mask > 0]))
    inp_img[mask > 0] = rgb
    h, w = inp_img.shape[0], inp_img.shape[1]
    if output_path == None:
        output_path = inp_img_path[:-4] + "_masked.png"
    cv2.imwrite(output_path, inp_img)

    if vis:
        inp_img = cv2.resize(inp_img, (int(w/4), int(h/4)))
        cv2.imshow("res", inp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return

if __name__ == "__main__":

    fuse_mask_img("/data/GribberGrabTest/data/yaml/mask.png",
                  "/data/GribberGrabTest/data/input_img/1118B.png", True)

