import numpy as np
import cv2
import torch
import argparse
from datetime import datetime
from utils.parse_dxf_org import *
from utils.parse_json import parse_json
from utils.workrest import WorkRest
from utils.grab_estimate import *
from utils.store_plan import *
from utils.DB.DbResposity import *
from utils.data_to_sql import *
from utils.matchTemplate import Matcher
import os
import yaml
os.chdir("/data/GribberGrabTest")


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # for api 4
    parser.add_argument('--nestID', type=str, help='a single NestID needed to be planed in cc_NestInfo')
    parser.add_argument('--taskID', type=str, default='', help='the taskID of the Nests need to be planed')
    parser.add_argument('--mode', type=str, default='test', help='mode: train or test')
    parser.add_argument('--first_mirror', action="store_true", default=False, help='First time mirror the img')

    # for saliency detection
    parser.add_argument('--imgs_folder', help='Path to folder containing images', type=str)
    # parser.add_argument('--model_path',default='/data/Salient_Pytorch/models/alph-0.7_wbce_w0-1.0_w1-1.15/weights/best-model_epoch-140_mae-0.0020_loss-0.0082.pth',  help='Path to model', type=str)
    parser.add_argument('--model_path', default='/data/GribberGrabTest/weights/best-model_epoch-140_mae-0.0020_loss-0.0082.pth' , help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=False, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=512, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=1, help='Batch Size for testing', type=int)
    parser.add_argument('--use_SOD', action="store_true", default=False, help='whether to generate mask yaml file through SOD')

    # for match template
    parser.add_argument('--debug', default=0, help='if debug (1 or 0)', type=int)
    parser.add_argument('--tmpl_num', default=0, help='template num to be used', type=int)
    parser.add_argument('--max_match', default=1, help='choose the top N result', type=int)
    parser.add_argument('--threshold', default=45, help='match threshold', type=int)
    parser.add_argument('--resize', default=0.25, help='image resize', type=float)
    parser.add_argument('--blur', default=5, help='image blur (odd value only)', type=int)
    parser.add_argument('--yaml', default='./data/yaml/mask.yaml', help='yaml file saved by Saliency Detection module', type=str)
    parser.add_argument('--use_prior', default=True, help='use Conveyor prior information to get piece_id',
                        type=bool)
    parser.add_argument('--align_pix', default=1400, help='right-handed align pixel length',type=int)
    return parser.parse_args()


def plan(args):
    '''
    1. parse the dxf or json
    :return:
    '''
    # Configuring
    for path in ["./output", "./output/pieces", "./output/kernel_vis", "./output/grab_results", "./data",
                 "./data/yaml", "./dump", "./dump/nests", "./dump/nests_grab_offset", "./dump/nests_store_offset",
                 "./weights"]:
        if not os.path.exists(path):
            os.makedirs(path)

    print("Configuring ...")
    CUDA = False
    config = {
        'CUDA': CUDA,
        'GribberKernelPath': "./weights/gribber_kernels_stack.pth",
    }


    if args.mode == 'train':
        data = {}
        from utils.grab_estimate import detect_area_center_and_principal
        nestID = args.nestID
        print("nestID:", nestID)
        groupID = nestID.split('_')[0]
        data.update({nestID: torch.load("./dump/nests/{}.pth".format(groupID))})

        yaml_path = os.path.join('./data/yaml', groupID)
        if os.path.exists(yaml_path):
            import shutil
            shutil.rmtree(yaml_path)
        os.makedirs(yaml_path)
        path = os.path.join('./data/training_img', groupID)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path)
        num = 0
        for i in data[nestID]["Parts"].keys():
            if data[nestID]["Parts"][i]["Grabbability"]:
                img = data[nestID]["Parts"][i]["Part"]
                # if img.shape[0]*img.shape[1] < 500000:
                if True:
                #    print("{}: too small, not used as template".format(data[nestID]["Parts"][i]["PartSN"]))
                # else:
                    img = detect_area_center_and_principal(img)[2]
                    if not os.path.exists(os.path.join(path, data[nestID]["Parts"][i]["PartSN"]+'.png')):
                        cv2.imwrite(os.path.join(path, data[nestID]["Parts"][i]["PartSN"]+'.png'), img)
                        num += 1
                # Debug
                # cv2.imshow(data[nestID]["Parts"][i]["PartSN"], img)
                # cv2.waitKey(0)
                # print("{}: \t\t{} x {} = {}".format(data[nestID]["Parts"][i]["PartSN"], img.shape[0], img.shape[1], img.shape[0]*img.shape[1]))
        print("Saved {} template image to {}".format(num, os.path.abspath(path)))
        # begin training
        print("Begin template training...")
        m = Matcher(args)
        m.train(path)

    elif args.mode == 'test':
        assert ('-' in args.imgs_folder and '.' in args.imgs_folder)
        args.batchIndex = int(args.imgs_folder.split('-')[-1].split('.')[0])

        # obligatory mirror operation:
        if args.first_mirror:
            img = cv2.imread(args.imgs_folder, cv2.IMREAD_UNCHANGED)
            (h, w) = img.shape
            if h > w:
                img = cv2.flip(img, flipCode=1)
            else:
                img = cv2.flip(img, flipCode=0)
            cv2.imwrite(args.imgs_folder, img)

        M = calibration()

        nestID = args.nestID
        print("nestID:", nestID)
        groupID = nestID.split('_')[0]
        data_like = {}
        data_like.update({nestID: torch.load("./dump/nests/{}.pth".format(groupID))})

        # add data generation from vision
        m = Matcher(args)
        # get templates number
        if args.tmpl_num == 0:
            args.tmpl_num = m.get_tmpl_num()
            print(args.tmpl_num)
        if args.use_SOD:
            m.get_salient_mask()
        if args.use_prior:
            update_piece_id(data_like, args)

        from utils.fuse_mask_img import fuse_mask_img
        fuse_mask_img("/data/GribberGrabTest/data/yaml/mask.png", args.imgs_folder)

        if args.debug:
            m.match()
            print("Debug ends, Exit")
            exit(0)
        else:
            res_dict = m.match()
            data = {}
            Reverse, Mirror = data_like[nestID]['Reverse'], data_like[nestID]['Mirror']
            (Height, Width) = m.get_img_size()
            data.update({nestID: {'Reverse': Reverse,
                                  'Mirror': Mirror,
                                  'Parts': {},
                                  'Height': Height,
                                  'Width': Width,
                                  'batchIndex': args.batchIndex}})
            for i in range(len(res_dict)):
                #cv2.imshow("part",m.src_of(i))
                #cv2.waitKey(0)
                Part, PartSN = m.src_of(res_dict[i])
                assert (len(Part.shape) == 2)

                d = {'PartID': '{}'.format(i),
                     'Part': Part,
                     'Origin': (res_dict[i]["x"], res_dict[i]["y"]),
                     'PartHeight': Part.shape[0],
                     'PartWidth': Part.shape[1],
                     'NestID': nestID,
                     'PartSN': PartSN,
                     'Technics': '1',
                     'NestPlanID': None,
                     'RequireSort': 1,
                     'Grabbability': True,
                     }
                data[nestID]['Parts'].update({i: d})
                #cv2.imshow('Part', Part)
                #cv2.waitKey(0)

        # plan grab
        grab_plan_stack(data, config)
        print("H, W = ", data[nestID]["Height"], data[nestID]["Width"])
        visualize_grab_plan_stack(args.imgs_folder, data)

        # transform the grab point to the gribber reference
        print("Transform from the image frame to the gribber reference ...")
        Store = {}
        for Parts in data_like[nestID]["Parts"].values():
            if Parts['Grabbability']:
                Store[Parts['PartSN']] =  Parts['Store']

        for nestID in data.keys():
            offset_grab = (0, 0, 0)
            for n in data[nestID]["Parts"].keys():
                # part["Grab"][n]["Grabpoint"] * M for homogeneous transformation
                for m in data[nestID]["Parts"][n]['Grab'].keys():
                    uv = data[nestID]["Parts"][n]['Grab'][m]['Grabpoint']
                    uv = np.array([uv[0], data[nestID]["Height"] - uv[1], 1.])
                    xy = np.dot(uv, M)
                    print("uv = {}, xy = {}".format(uv, xy))
                    xy = (xy[0], xy[1])
                    data[nestID]["Parts"][n]['Grab'][m]['Grabpoint'] = xy

                # parts_to_store key "store" copy to data key "store" (by partSN)
                data[nestID]["Parts"][n].update({'Store': Store[data[nestID]["Parts"][n]['PartSN']]})
                # 对2个以上抓手的情况进行 synchronization, 以第1个抓手(m=0)的travel_length(Dx,Dy)为标准
                if len(data[nestID]["Parts"][n]['Store'].keys()) >= 2:
                    Storepoint = data[nestID]["Parts"][n]['Store'][0]['Storepoint']
                    Grabpoint = data[nestID]['Parts'][n]['Grab'][0]['Grabpoint']
                    travel_length = (Storepoint[0] - Grabpoint[0], Storepoint[1] - Grabpoint[1])
                    synced_Storepoint = (data[nestID]['Parts'][n]['Grab'][1]['Grabpoint'][0] + travel_length[0],
                                         data[nestID]['Parts'][n]['Grab'][1]['Grabpoint'][1] + travel_length[1])
                    data[nestID]["Parts"][n]['Store'][1]['Storepoint'] = synced_Storepoint
                    # 角度也要统一，因为在超长件的情况下不能旋转
                    for m in data[nestID]["Parts"][n]['Grab'].keys():
                        data[nestID]["Parts"][n]['Store'][m]['Theta'] = data[nestID]["Parts"][n]['Grab'][m]['Theta']

            data[nestID].update({"StoreStatus": 1})

        print("Inserting Stack Strategy into SQL Server ...")
        for nestID in data.keys():
           insert_strategy_result(args.nestID_full, 2, data[nestID]['StoreStatus'], "")
        store_insert_sql_stack(data, with_offset=True)
        print("Done")


def calibration():
    # The pixel values were taken before resize x1.111... so the factor 1.111 is implied in the matrix.
    # We do no more resize x1.111
    # mm = np.array([[-5120., 1429.9, 1], [-1757.4, 1419.7, 1], [-1689.6, 3080.7, 1], [-4851.5, 3092.3, 1]])
    # Pix = np.array([[1566, 335, 1], [4642, 327, 1], [4697, 1814, 1], [1803, 1820, 1]])
    mm = np.array([[-4348.0, 3160.7, 1], [4830.4, 2904.7, 1], [5330.7, 1456.3, 1], [-4651.2, 1474.4, 1]])
    Pix = np.array([[2232, 153, 1], [10656, 319, 1], [11124, 1610, 1], [1970, 1665, 1]])

    M = np.linalg.lstsq(Pix, mm, rcond=None)[0]
    return M


def update_piece_id(data, args, yaml_path = "./data/yaml/mask.yaml", id_path = './data/training_img/{}/piece_id.yaml'):
    # 计算包围框中心和Convey Point的距离， 距离最近的就是该包围框对应的piece_id
    nestID = args.nestID
    batchIndex = args.batchIndex

    print('\nupdating piece_id with Conveyor prior, batchIndex={}'.format(batchIndex))

    with open(id_path.format(args.groupID), 'r', encoding="utf-8") as f:
        id_to_PartSN_dict = yaml.load(f, Loader=yaml.FullLoader)
    PartSN_to_id_dict = {v.split('.')[0]: k for k, v in id_to_PartSN_dict.items()}

    piece_on_conveyor = []
    for i in data[nestID]['Parts'].keys():
        if data[nestID]['Parts'][i]['PartSN'] in PartSN_to_id_dict.keys():
            if 'Convey' in data[nestID]['Parts'][i].keys():
                if data[nestID]['Parts'][i]['Convey'][0]['ConveyorID'] == batchIndex - 1: #ConveyorID: 0,1,2... batchIndex: 1,2,3
                    tmp = data[nestID]['Parts'][i]['Convey']
                    tmp[0].update({'PartSN': data[nestID]['Parts'][i]['PartSN']})
                    piece_on_conveyor.append(tmp)
    with open(yaml_path, 'r', encoding="utf-8") as f:
        mask_yaml = yaml.load(f, Loader=yaml.FullLoader)
    infos = mask_yaml['infos']
    print('{} pieces on planned Conveyor, {} pieces in the actual image'.format(len(piece_on_conveyor), len(infos)))

    # 对齐Conveyor和线扫相机两图
    x0, y0 = 65535, 65536  # 相片中最靠左/最靠上的工件
    for info in infos:
        x,y = info['x'], info['y']
        if x < x0:
            x0 = x
        if y < y0:
            y0 = y
    print("x0 = ", x0)
    print("y0 = ", y0)
    #img = cv2.imread(args.imgs_folder, cv2.IMREAD_COLOR)
    # 对每一个由显著性检测提取的包围框， 计算一遍每个Conveypoint到包围框中心的距离，距离最小的Conveypoint就是对应的piece_id.
    # 有的工件有多个Conveypoint, 取距离平均值
    for info in infos:
        x,y,w,h = info['x'], info['y'], info['w'], info['h']
        piece_id = info['piece_id']
        center = np.array([int(x + w/2), int(y + h/2)])
        min_dist = 65535
        PartSN = ''
        for piece in piece_on_conveyor:
            dist = []
            for i in piece.keys():
                ConveyPoint = np.array(piece[i]['Conveypoint'])
                #print("{} = {} - {}". format(data[nestID]['Height'] - ConveyPoint[1], data[nestID]['Height'], ConveyPoint[1]) )
                #ConveyPoint[1] = data[nestID]['Height'] - ConveyPoint[1]  # 由于坐标系方向不同所以y轴坐标要颠倒一下
                ConveyPoint[1] = 1900 - ConveyPoint[1]
                ConveyPoint[0] = int((ConveyPoint[0] / 1.1 + x0))
                ConveyPoint[1] = int((ConveyPoint[1] / 1.1) + y0)
                #cv2.circle(img, (ConveyPoint[0], ConveyPoint[1]), 10, (0,0,255), 5)
                #cv2.rectangle(img, pt1 =(x,y), pt2=(x+w, y+h),  color = (0,0,255) )
                dist.append(np.linalg.norm(ConveyPoint - center))
            dist = np.mean(dist)
            if dist < min_dist:
                min_dist = dist
                PartSN = piece[0]['PartSN']
        #print("PartSN={}".format(PartSN))
        if PartSN != '' and PartSN in PartSN_to_id_dict.keys():
            piece_id = PartSN_to_id_dict[PartSN]
        else:
            print("Warning: PartSN doesn't exist, keep piece_id as 0")
        info['piece_id'] = piece_id
    #img = cv2.resize(img, (3000,500))
    #cv2.imshow("img",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    with open(yaml_path, 'w', encoding="utf-8") as f:
        mask_yaml['infos'] = infos
        yaml.dump(mask_yaml, f)
        print('updating mask.yaml done\n')


if __name__ == "__main__":
    args = parse_arguments()
    args.nestID_full = args.nestID
    args.groupID = args.nestID.split("_")[0]

    plan(args)
