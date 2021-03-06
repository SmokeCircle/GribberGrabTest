import numpy as np
import cv2
from utils.DB.DbResposity import *
import math
from utils.conveyor import *

ConveyorPlaceOriginRightMid = (4400, 5354)  # 工件放置的边界（x, y），x轴以工件的有边界为基准

def convey_plan(data, config):
    # Calculate the store coordinate
    for nestID in data.keys():
        nest = data[nestID]["Parts"]
        nestHeight = data[nestID]["Height"]
        conveyors = [Conveyor({'cuda': config['CUDA'], 'id': 0})]

        part_keys = list(nest.keys())
        part_keys = sorted(part_keys, key=lambda x: nest[x]['PartHeight']*nest[x]['PartWidth'], reverse=True)

        # for i in list(nest.keys()):
        for i in part_keys:
            if not nest[i]["Grabbability"]:
                continue
            nest[i]['Convey'] = {}

            part_grab = nest[i]
            partID = part_grab['PartID']
            partSN = part_grab['PartSN']
            orientation_grab = part_grab['Orientation']
            piece_grab = part_grab['Piece']
            center_grab = part_grab['Center']
            origin_grab = part_grab['Origin']
            grab_dict = part_grab['Grab']
            mode = part_grab['Mode']

            fit = False
            coord = (-1, -1)
            rot = False

            cid = 0
            while conveyors[cid].is_full() and cid < len(conveyors):
                cid += 1
            while cid >= len(conveyors):
                conveyors.append(Conveyor({'cuda': config['CUDA'], 'id': len(conveyors)}))
            while not fit:
                fit, coord, rot, piece_convey = conveyors[cid].is_fit(piece_grab)
                if fit:
                    break
                else:
                    cid += 1
                    while cid >= len(conveyors):
                        conveyors.append(Conveyor({'cuda': config['CUDA'], 'id': len(conveyors)}))

            center_convey = coord
            rotation_angle_to_store = 90 if rot else 0

            # refine the center_convey if the grab mode is '100' or '50'
            # if mode == '100':
            #     x, y = center_convey
            #     rot = grab_dict[0]['Theta'] + rotation_angle_to_store
            #     x_r = x - 125 * math.sin(rot / 180 * math.pi)
            #     y_r = y + 125 * math.cos(rot / 180 * math.pi)
            #     center_convey = (x_r, y_r)
            # elif mode == '50':
            #     x, y = center_convey
            #     rot = grab_dict[0]['Theta'] + rotation_angle_to_store
            #     x_r = x - 25 * math.sin(rot / 180 * math.pi)
            #     y_r = y + 25 * math.cos(rot / 180 * math.pi)
            #     center_convey = (x_r, y_r)

            center_convey = (center_convey[0], conveyors[cid].height_real - center_convey[1])  # change the origin from top left to button left

            num_gribbers = len(grab_dict.keys())
            if num_gribbers == 1:
                center_grab_refined = (center_grab[0]+origin_grab[0], nestHeight - (center_grab[1]+origin_grab[1]))
                grabpoint_one = grab_dict[0]['Grabpoint']
                conveypoint_one = (center_convey[0]+grabpoint_one[0]-center_grab_refined[0],
                                  center_convey[1]+grabpoint_one[1]-center_grab_refined[1])
                nest[i]['Convey'][0] = {
                    'ConveyorID': cid,
                    'Theta': grab_dict[0]['Theta'] + rotation_angle_to_store,
                    'Conveypoint': conveypoint_one,
                }
            elif num_gribbers == 2:
                grabpoint_one = grab_dict[0]['Grabpoint']
                grabpoint_two = grab_dict[1]['Grabpoint']
                center_grab_refined = (center_grab[0]+origin_grab[0], nestHeight - (center_grab[1]+origin_grab[1]))

                conveypoint_one = (center_convey[0]+grabpoint_one[0]-center_grab_refined[0],
                                  center_convey[1]+grabpoint_one[1]-center_grab_refined[1])
                conveypoint_two = (center_convey[0]+grabpoint_two[0]-center_grab_refined[0],
                                  center_convey[1]+grabpoint_two[1]-center_grab_refined[1])

                nest[i]['Convey'][0] = {
                    'ConveyorID': cid,
                    'Theta': grab_dict[0]['Theta'],
                    'Conveypoint': conveypoint_one,
                }
                nest[i]['Convey'][1] = {
                    'ConveyorID': cid,
                    'Theta': grab_dict[1]['Theta'],
                    'Conveypoint': conveypoint_two,
                }
        data[nestID].update({"ConveyStatus": 1})
        data[nestID]["Parts"].update(nest)
        for i in range(len(conveyors)):
            conveyors[i].visualize(nestID)
    visualize_convey_plan(data)


def convey_apply_offset_all(data: dict, width=10000, height=1900):  # width和height为Conveyor中的长度和宽度
    delta_x = ConveyorPlaceOriginRightMid[0] - width
    delta_y = ConveyorPlaceOriginRightMid[1] - height // 2

    for nestID in data.keys():
        nest = data[nestID]["Parts"]
        for i in list(nest.keys()):
            if not nest[i]["Grabbability"]:
                continue
            for n in nest[i]["Convey"].keys():
                x, y = nest[i]["Convey"][n]["Conveypoint"]
                nest[i]["Convey"][n]["Conveypoint"] = (delta_x + x, delta_y + y)
        data[nestID]["Parts"].update(nest)


def visualize_convey_plan(data, conveyorHeight=1900):
    colors = {
        "50": (255, 0, 0),  # Blue  只使用一组阵列中一行4个直径50吸头的情况为蓝色
        "100": (0, 255, 0),  # Green  只使用一组阵列中一行4个直径100吸头的情况为绿色
        "small": (0, 0, 255),  # Red  只使用一组阵列的情况为红色
        "medium": (240, 32, 160),  # Purple 使用2组阵列的情况为紫色
        "large": (139, 0, 0),  # deep blue 使用3组阵列的情况为深蓝色
    }
    length = 125
    for nestID in data.keys():
        parts = data[nestID]["Parts"]
        cid = 0
        for n in parts.keys():
            if parts[n]["Grabbability"]:
                cid = max(cid, parts[n]["Convey"][0]["ConveyorID"])
        ConveyorBatch = cid + 1
        imgs = {}
        for i in range(ConveyorBatch):
            imgs[i] = cv2.imread("./dump/conveyor_vis/conveyor_id{}_nestID={}.png".format(i, nestID))
        for n in parts.keys():
            if parts[n]["Grabbability"]:
                img = imgs[parts[n]["Convey"][0]["ConveyorID"]]
                for g in parts[n]["Convey"].keys():
                    pt = parts[n]["Convey"][g]["Conveypoint"]
                    pt = (int(pt[0]), int(conveyorHeight - pt[1]))
                    theta = parts[n]["Convey"][g]["Theta"]
                    mode = parts[n]["Mode"]
                    ds = (int(pt[0]-length*math.cos((theta-90)/180*math.pi)), int(pt[1]+length*math.sin((theta-90)/180*math.pi)))
                    cv2.circle(img, pt, 30, colors[mode], -1)
                    cv2.line(img, pt, ds, colors[mode], thickness=10)
                imgs[parts[n]["Convey"][0]["ConveyorID"]] = img
        for i in range(ConveyorBatch):
             cv2.imwrite("./dump/conveyor_vis/conveyor_id{}_nestID={}.png".format(i, nestID), imgs[i])