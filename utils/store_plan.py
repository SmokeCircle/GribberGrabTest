import numpy as np
import cv2
from utils.workrest import *
from utils.DB.DbResposity import *
import math


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


def crop_area(binary):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.boundingRect(contours[0])
    binary = binary[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return binary


def store_plan(data, config, workrests):
    pieces_to_store = {}
    # n = 0
    for nestID in data.keys():
        for i in data[nestID]["Parts"].keys():
            piece = data[nestID]["Parts"][i]
            if piece['PartSN'] in list(pieces_to_store.keys()) or not piece['Grabbability']:
                continue
            pieces_to_store[piece['PartSN']] = {
                'PartSN': piece['PartSN'],
                'Orientation': piece['Orientation'],
                # 'Center': piece['Center'],
                'Piece': piece['Piece'],
                'Technics': piece['Technics']
            }
            # n += 1

    # Plan the place of the workpieces on the workrests
    keys = list(pieces_to_store.keys())
    keys.sort(key=lambda x: np.sum(pieces_to_store[x]['Piece']))  # sort according to the piece grabbable area size
    # print(keys)
    # keys = keys[15:]
    for k in keys:
        # print(k)
        workrest_type = choose_workrest_size(pieces_to_store[k]['Piece'])
        tech = pieces_to_store[k]['Technics']

        width = -1
        if workrest_type == 0:
            width = 2500
        elif workrest_type == 1:
            width = 6000
        elif workrest_type == 2:
            width = 12000
        else:
            raise ValueError("Store_Plane:: The workrest type is illegal!")

        if tech not in workrests[workrest_type].keys():
            workrests[workrest_type][tech] = [WorkRest({'cuda': config['CUDA'], 'width_real': width, 'id': 0,
                                                        'technic': tech, 'simple': workrest_type==2})]

        fit = False
        coord = (-1, -1)
        rot = False

        wid = 0
        while workrests[workrest_type][tech][wid].is_full() and wid < len(workrests[workrest_type][tech]):
            wid += 1
        while wid >= len(workrests[workrest_type][tech]):
            workrests[workrest_type][tech].append(WorkRest({'cuda': config['CUDA'], 'width_real': width,
                                                'id': len(workrests[workrest_type][tech]),
                                                'technic': tech, 'simple': workrest_type==2}))

        piece_store = None
        while not fit:
            fit, coord, rot, piece_store = workrests[workrest_type][tech][wid].is_fit(pieces_to_store[k]['Piece'])
            if fit:
                break
            else:
                wid += 1
                while wid >= len(workrests[workrest_type][tech]):
                    workrests[workrest_type][tech].append(
                        WorkRest({'cuda': config['CUDA'], 'width_real': width,
                                  'id': len(workrests[workrest_type][tech]),
                                  'technic': tech, 'simple': workrest_type==2}))
        angle = 90 if rot else 0
        orientation = pieces_to_store[k]['Orientation'] + angle
        pieces_to_store[k]['Store'] = {
            'WorkRestID': wid,
            'WorkRestType': workrest_type,
            'Technics': tech,
            'GravityCenterCoord': coord,
            'Orientation': orientation,
            'Piece': piece_store,
        }
        # print(workrests[t][tech][id].technic, workrests[t][tech][id].id, fit, coord, rot)
        workrests[workrest_type][tech][wid].visualize()

    # Calculate the store coordinate
    for nestID in data.keys():
        nest = data[nestID]["Parts"]
        for i in list(nest.keys()):
            nest[i]['Store'] = {}
            if not nest[i]["Grabbability"]:
                continue

            part_grab = nest[i]
            partID = part_grab['PartID']
            partSN = part_grab['PartSN']
            orientation_grab = part_grab['Orientation']
            piece_grab = part_grab['Piece']
            center_grab = part_grab['Center']
            grab_dict = part_grab['Grab']
            mode = part_grab['Mode']

            part_store = pieces_to_store[partSN]['Store']
            orientation_store = part_store['Orientation']
            piece_store = part_store['Piece']
            workrest_id = part_store['WorkRestID']
            workrest_type = ['S', 'M', 'L'][part_store['WorkRestType']]
            center_store = part_store['GravityCenterCoord']
            tech_store = part_store['Technics']

            rotation_angle_to_store = orientation_grab - orientation_store

            h, w = piece_grab.shape
            d = max(h, w)
            piece_grab = np.pad(piece_grab, ((d, d), (d, d)))
            rot = cv2.getRotationMatrix2D((center_grab[0] + d, center_grab[1] + d), rotation_angle_to_store, 1.0)
            piece_correct = cv2.warpAffine(piece_grab, rot, (w + 2 * d, h + 2 * d))
            piece_correct = crop_area(piece_correct)
            flip_angle = 180 if check_flip(piece_store, piece_correct) else 0

            rotation_angle_to_store += flip_angle

            # refine the center_store if the grab mode is '100' or '50'
            if mode == '100':
                x, y = center_store
                rot = grab_dict[0]['Theta'] + rotation_angle_to_store
                x_r = x - 125 * math.sin(rot / 180 * math.pi)
                y_r = y + 125 * math.cos(rot / 180 * math.pi)
                center_store = (x_r, y_r)
            elif mode == '50':
                x, y = center_store
                rot = grab_dict[0]['Theta'] + rotation_angle_to_store
                x_r = x - 25 * math.sin(rot / 180 * math.pi)
                y_r = y + 25 * math.cos(rot / 180 * math.pi)
                center_store = (x_r, y_r)

            num_gribbers = len(grab_dict.keys())
            if num_gribbers == 1:
                nest[i]['Store'][0] = {
                    'WorkRestID': workrest_id,
                    'WorkRestType': workrest_type,
                    'Technics': tech_store,
                    'Theta': grab_dict[0]['Theta'] + rotation_angle_to_store,
                    'Storepoint': center_store,
                }
            elif num_gribbers == 2:
                grabpoint_one = grab_dict[0]['Grabpoint']
                grabpoint_two = grab_dict[1]['Grabpoint']

                storepoint_one = (center_store[0]+grabpoint_one[0]-center_grab[0],
                                  center_store[1]+grabpoint_one[1]-center_grab[1])
                storepoint_two = (center_store[0]+grabpoint_two[0]-center_grab[0],
                                  center_store[1]+grabpoint_two[1]-center_grab[1])

                nest[i]['Store'][0] = {
                    'WorkRestID': workrest_id,
                    'WorkRestType': workrest_type,
                    'Technics': tech_store,
                    'Theta': grab_dict[0]['Theta'],
                    'Storepoint': storepoint_one,
                }
                nest[i]['Store'][1] = {
                    'WorkRestID': workrest_id,
                    'WorkRestType': workrest_type,
                    'Technics': tech_store,
                    'Theta': grab_dict[1]['Theta'],
                    'Storepoint': storepoint_two,
                }
        data[nestID].update({"StoreStatus": 1})
        data[nestID]["Parts"].update(nest)

    return pieces_to_store, workrests

def store_apply_offset(part: dict, offset: tuple):
    '''
    Store doesn't support rotation
    :param part: {"PartID": xxx, "Origin": xxx, "Part": xxx, "Center": xxx, "Orientation": xxx, "Grab": {
    0: {"Grabpoint": (x, y), "Theta": degree, 100: list, 50: list},
    1: {"Grabpoint": (x, y), "Theta": degree, 100: list, 50: list}
    },
    "Store":{
    0: {"Storepoint": (x, y), "Theta": degree, "WorkRestID": xxx, "Technics": xxx},
    1: {"Storepoint": (x, y), "Theta": degree, "WorkRestID": xxx, "Technics": xxx}
    }}
    :param offset: (delta_x, delta_y, delta_theta), delta_x, delta_y is the coordinate of the plane origin in the global reference,
    delta_theta is in degree
    :return:
    '''

    delta_x, delta_y, delta_theta = offset
    for n in part["Store"].keys():
        x, y = part["Store"][n]["Storepoint"]
        part["Store"][n]["Storepoint"] = (delta_x + x, delta_y + y)
    return part





