import torch
import numpy as np
import json
import cv2
from utils.DB.DbResposity import *


def counterclock2clock(angle):
    result = 360 - angle
    result = 0 if result == 360 else result
    return result


def generate_grab_messages(data, nestID, grab_times, keys_single_left, keys_single_right, keys_dual, config,
                           device_code=None, nest_shape=(0, 0), visualize=False):
    '''
    the grabpoint is already in the gribber after applying the offset.

    reference: plate -> conveyor -> leveler -> store

    :param data:
    :param nestID:
    :param grab_times:
    :param keys_single_left:
    :param keys_single_right:
    :param keys_dual:
    :param config: {
        "Plate2ConveyorDict": reference_plate2conveyor,
        "Conveyor2Leveler": reference_conveyor2leveler,
        "Gribber2WorkRests": reference_gribber2workrests,
        "Grab2Store": reference_grab2store,
        'PlatesDict': reference_grab2plate,
    }
    :param device_code:
    :param nest_shape: (nestHeight, nestWidth)
    :return:
    '''
    device_code = 'LC-01' if device_code is None else device_code
    reference_plate2conveyor = config['Plate2ConveyorDict'][device_code]

    message_grab_one = []
    message_grab_two = []

    if visualize:
        colors = {
            "50": (255, 0, 0),  # Blue  只使用一组阵列中一行4个直径50吸头的情况为蓝色
            "100": (0, 255, 0),  # Green  只使用一组阵列中一行4个直径100吸头的情况为绿色
            "small": (0, 0, 255),  # Red  只使用一组阵列的情况为红色
            "medium": (240, 32, 160),  # Purple 使用2组阵列的情况为紫色
            "large": (112, 20, 20),  # deep blue 使用3组阵列的情况为深蓝色
        }
        img = cv2.imread("./dump/grab_vis/{}.png".format(nestID))
        font_scale = 5
        pad = 200
        dst = (np.ones([img.shape[0]+2*pad, img.shape[1]+2*pad, img.shape[2]]) * 255.).astype(np.uint8)
        dst[pad:img.shape[0]+pad, pad:img.shape[1]+pad] = img
        img = dst
        # img = np.pad(img, pad, 'maximum')

    i = 0
    batch_quantity = 0

    for kl in keys_single_left:
        part = data[kl]
        batch_quantity = max(batch_quantity, part["Convey"][0]["ConveyorID"])

        json_one = {
            "NestID": nestID,
            "GrabTimes": grab_times,
            "GrabIndex": i,
            "RobotIndex": 1,
            "Origin_X": part["Grab"][0]["Grabpoint"][0],
            "Origin_Y": part["Grab"][0]["Grabpoint"][1],
            "Origin_A": counterclock2clock(part["Grab"][0]["Theta"]),  # from counter-clockwise to clockwise
            "Destination_X": part["Convey"][0]["Conveypoint"][0],
            "Destination_Y": part["Convey"][0]["Conveypoint"][1],
            "Destination_A":  counterclock2clock(part["Convey"][0]["Theta"]),
            "Magnetic50": part["Grab"][0]["50"],
            "Magnetic100": part["Grab"][0]["100"],
            "Power50": 0,
            "Power100": 0,
            "BatchIndex": part["Convey"][0]["ConveyorID"],
        }
        message_grab_one.append(json_one)
        if visualize:
            pt = part["Grab"][0]["Grabpoint"]
            pt = (int(pt[0])+pad, int(nest_shape[0] - pt[1])+pad)
            mode = part["Mode"]
            cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[mode], thickness=5)
        i += 1

    for kr in keys_single_right:
        part = data[kr]
        batch_quantity = max(batch_quantity, part["Convey"][0]["ConveyorID"])

        json_two = {
            "NestID": nestID,
            "GrabTimes": grab_times,
            "GrabIndex": i,
            "RobotIndex": 2,
            "Origin_X": part["Grab"][0]["Grabpoint"][0],
            "Origin_Y": part["Grab"][0]["Grabpoint"][1],
            "Origin_A":  counterclock2clock(part["Grab"][0]["Theta"]),
            "Destination_X": part["Convey"][0]["Conveypoint"][0],
            "Destination_Y": part["Convey"][0]["Conveypoint"][1],
            "Destination_A": counterclock2clock(part["Convey"][0]["Theta"]),
            "Magnetic50": part["Grab"][0]["50"],
            "Magnetic100": part["Grab"][0]["100"],
            "Power50": 0,
            "Power100": 0,
            "BatchIndex": part["Convey"][0]["ConveyorID"],
        }
        message_grab_two.append(json_two)
        if visualize:
            pt = part["Grab"][0]["Grabpoint"]
            pt = (int(pt[0])+pad, int(nest_shape[0] - pt[1])+pad)
            mode = part["Mode"]
            cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[mode], thickness=5)
        i += 1

    for i, k in enumerate(keys_dual):
        part = data[k]
        batch_quantity = max(batch_quantity, part["Convey"][0]["ConveyorID"])
        json_one = {
            "NestID": nestID,
            "GrabTimes": grab_times,
            "GrabIndex": i + len(keys_single_left) +len(keys_single_right),
            "RobotIndex": 31,
            "Origin_X": part["Grab"][0]["Grabpoint"][0],
            "Origin_Y": part["Grab"][0]["Grabpoint"][1],
            "Origin_A":  counterclock2clock(part["Grab"][0]["Theta"]),
            "Destination_X": part["Convey"][0]["Conveypoint"][0],
            "Destination_Y": part["Convey"][0]["Conveypoint"][1],
            "Destination_A":  counterclock2clock(part["Convey"][0]["Theta"]),
            "Magnetic50": part["Grab"][0]["50"],
            "Magnetic100": part["Grab"][0]["100"],
            "Power50": 0,
            "Power100": 0,
            "BatchIndex": part["Convey"][0]["ConveyorID"],
        }

        json_two = {
            "NestID": nestID,
            "GrabTimes": grab_times,
            "GrabIndex": i + len(keys_single_left) +len(keys_single_right),
            "RobotIndex": 32,
            "Origin_X": part["Grab"][1]["Grabpoint"][0],
            "Origin_Y": part["Grab"][1]["Grabpoint"][1],
            "Origin_A":  counterclock2clock(part["Grab"][1]["Theta"]),
            "Destination_X": part["Convey"][1]["Conveypoint"][0],
            "Destination_Y": part["Convey"][1]["Conveypoint"][1],
            "Destination_A":  counterclock2clock(part["Convey"][0]["Theta"]),
            "Magnetic50": part["Grab"][1]["50"],
            "Magnetic100": part["Grab"][1]["100"],
            "Power50": 0,
            "Power100": 0,
            "BatchIndex": part["Convey"][0]["ConveyorID"],
        }
        if visualize:
            pt0 = part["Grab"][0]["Grabpoint"]
            pt0 = (int(pt0[0])+pad, int(nest_shape[0] - pt0[1])+pad)
            pt1 = part["Grab"][1]["Grabpoint"]
            pt1 = (int(pt1[0])+pad, int(nest_shape[0] - pt1[1])+pad)
            mode = part["Mode"]
            cv2.putText(img, str(i + len(keys_single_left) +len(keys_single_right)), pt0, cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[mode], thickness=5)
            cv2.putText(img, str(i + len(keys_single_left) +len(keys_single_right)), pt1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[mode], thickness=5)

        message_grab_one.append(json_one)
        message_grab_two.append(json_two)

    batch_quantity = batch_quantity + 1
    for i in range(len(message_grab_one)):
        message_grab_one[i].update({"BatchQuantity": batch_quantity})
    for i in range(len(message_grab_two)):
        message_grab_two[i].update({"BatchQuantity": batch_quantity})

    if visualize:
        cv2.imwrite("./dump/grab_vis/{}_index.png".format(nestID), img)

    return message_grab_one, message_grab_two


def generate_store_messages(data, nestID, grab_times, keys_single_left, keys_single_right, keys_dual,
                            nest_shape=(0, 0), visualize=False):
    '''

    :param data:
    :param nestID:
    :param grab_times:
    :param keys_single_left:
    :param keys_single_right:
    :param keys_dual:
    :param config: {
        "Plate2ConveyorDict": reference_plate2conveyor,
        "Conveyor2Leveler": reference_conveyor2leveler,
        "Grab2Store": reference_grab2store,
        'PlatesDict': reference_grab2plate,
    }
    :return:
    '''

    message_store_one = []
    message_store_two = []

    i = 0
    for kl in keys_single_left:
        part = data[kl]
        json_one = {
            "NestID": nestID,
            "GrabTimes": grab_times,
            "GrabIndex": i,
            "RobotIndex": 1,
            "Origin_X": part["Grab"][0]["Grabpoint"][0],
            "Origin_Y": part["Grab"][0]["Grabpoint"][1],
            "Origin_A":  counterclock2clock(part["Grab"][0]["Theta"]),
            "Destination_X": part["Store"][0]["Storepoint"][0],
            "Destination_Y": part["Store"][0]["Storepoint"][1],
            "Destination_A":  counterclock2clock(part["Store"][0]["Theta"]),
            "Magnetic50": part["Grab"][0]["50"],
            "Magnetic100": part["Grab"][0]["100"],
            "Power50": 0,
            "Power100": 0,
            "StockIndex": "{}_{}_{}".format(part["Store"][0]["Technics"],
                                            part["Store"][0]["WorkRestType"],
                                            part["Store"][0]["WorkRestID"]),
        }
        i += 1
        message_store_one.append(json_one)

    for kr in keys_single_right:
        part = data[kr]
        json_two = {
            "NestID": nestID,
            "GrabTimes": grab_times,
            "GrabIndex": i,
            "RobotIndex": 2,
            "Origin_X": part["Grab"][0]["Grabpoint"][0],
            "Origin_Y": part["Grab"][0]["Grabpoint"][1],
            "Origin_A":  counterclock2clock(part["Grab"][0]["Theta"]),
            "Destination_X": part["Store"][0]["Storepoint"][0],
            "Destination_Y": part["Store"][0]["Storepoint"][1],
            "Destination_A":  counterclock2clock(part["Store"][0]["Theta"]),
            "Magnetic50": part["Grab"][0]["50"],
            "Magnetic100": part["Grab"][0]["100"],
            "Power50": 0,
            "Power100": 0,
            "StockIndex": "{}_{}_{}".format(part["Store"][0]["Technics"],
                                            part["Store"][0]["WorkRestType"],
                                            part["Store"][0]["WorkRestID"]),
        }
        i += 1
        message_store_two.append(json_two)

    for i, k in enumerate(keys_dual):
        part = data[k]
        json_one = {
            "NestID": nestID,
            "GrabTimes": grab_times,
            "GrabIndex": i + len(keys_single_left)+len(keys_single_right),
            "RobotIndex": 31,
            "Origin_X": part["Grab"][0]["Grabpoint"][0],
            "Origin_Y": part["Grab"][0]["Grabpoint"][1],
            "Origin_A":  counterclock2clock(part["Grab"][0]["Theta"]),
            "Destination_X": part["Store"][0]["Storepoint"][0],
            "Destination_Y": part["Store"][0]["Storepoint"][1],
            "Destination_A":  counterclock2clock(part["Store"][0]["Theta"]),
            "Magnetic50": part["Grab"][0]["50"],
            "Magnetic100": part["Grab"][0]["100"],
            "Power50": 0,
            "Power100": 0,
            "StockIndex": "{}_{}_{}".format(part["Store"][0]["Technics"],
                                            part["Store"][0]["WorkRestType"],
                                            part["Store"][0]["WorkRestID"]),
        }

        json_two = {
            "NestID": nestID,
            "GrabTimes": grab_times,
            "GrabIndex": i + len(keys_single_left)+len(keys_single_right),
            "RobotIndex": 32,
            "Origin_X": part["Grab"][1]["Grabpoint"][0],
            "Origin_Y": part["Grab"][1]["Grabpoint"][1],
            "Origin_A":  counterclock2clock(part["Grab"][1]["Theta"]),
            "Destination_X": part["Store"][1]["Storepoint"][0],
            "Destination_Y": part["Store"][1]["Storepoint"][1],
            "Destination_A":  counterclock2clock(part["Store"][1]["Theta"]),
            "Magnetic50": part["Grab"][1]["50"],
            "Magnetic100": part["Grab"][1]["100"],
            "Power50": 0,
            "Power100": 0,
            "StockIndex": "{}_{}_{}".format(part["Store"][1]["Technics"],
                                            part["Store"][1]["WorkRestType"],
                                            part["Store"][1]["WorkRestID"]),
        }

        message_store_one.append(json_one)
        message_store_two.append(json_two)

    return message_store_one, message_store_two


def grab_insert_sql(data_all, references, with_offset=False, visualize=False):
    for nestID in data_all.keys():
        if not data_all[nestID]["GrabStatus"]:
            continue
        data = data_all[nestID]["Parts"]
        device_code = data_all[nestID]["DeviceCode"] if "DeviceCode" in data_all[nestID].keys() else None
        nestHeight = data_all[nestID]["Height"]
        nestWidth = data_all[nestID]["Width"]
        ref_local = references.copy()

        if with_offset:
            ref_p2c = ref_local["Plate2ConveyorDict"][device_code]
            margin = data_all[nestID]["MarginGrab"]
            ref_p2c = [ref_p2c[0], ref_p2c[1]-margin[1]]
            ref_local["Plate2ConveyorDict"][device_code] = ref_p2c
            grab_orders = torch.load("./dump/nest_grab_orders/{}.pth".format(nestID.split("_")[0]))
            keys_single_left = grab_orders["SingleLeft"]
            keys_single_right = grab_orders["SingleRight"]
            keys_dual = grab_orders["Dual"]
            keys_single = grab_orders["Single"]
        else:
            keys = list(data.keys())
            keys_single, keys_dual = [], []
            for k in keys:
                if data[k]["Grabbability"]:
                    if len(list(data[k]['Grab'].keys())) == 1:
                        keys_single.append(k)
                    elif len(list(data[k]['Grab'].keys())) == 2:
                        keys_dual.append(k)

            nestMiddle = nestWidth/2 + data_all[nestID]["OffsetGrab"][0] if with_offset else nestWidth/2
            keys_single.sort(key=lambda x: data[x]['Grab'][0]['Grabpoint'])
            keys_single_left, keys_single_right = [], []
            for k in keys_single:
                if data[k]['Grab'][0]['Grabpoint'][0] <= nestMiddle:
                    keys_single_left.append(k)
                else:
                    keys_single_right.append(k)
            keys_dual.sort(key=lambda x: data[x]['Grab'][0]['Grabpoint'])
            grab_orders = {
                "Single": keys_single,
                "SingleLeft": keys_single_left,
                "SingleRight": keys_single_right,
                "Dual": keys_dual,
            }
            torch.save(grab_orders, "./dump/nest_grab_orders/{}.pth".format(nestID))


        grab_times = len(keys_single) + len(keys_dual)

        # Grab
        message_grab_one, message_grab_two = generate_grab_messages(data,
                                                                    nestID,
                                                                    grab_times,
                                                                    keys_single_left,
                                                                    keys_single_right,
                                                                    keys_dual,
                                                                    references,
                                                                    device_code,
                                                                    (nestHeight, nestWidth),
                                                                    visualize=visualize)

        insert_function = insert_sort_strategy_offset if with_offset else insert_sort_strategy
        # id = len(select_result_all_from_table("cc_SortStrategyOffset"))+1 if with_offset else len(select_result_all_from_table("cc_SortStrategy"))+1
        for m in message_grab_one:
            insert_function(m)
        for m in message_grab_two:
            insert_function(m)


def store_insert_sql(data_all, with_offset=False, visualize=False):
    for nestID in data_all.keys():
        if not data_all[nestID]["GrabStatus"]:
            continue
        data = data_all[nestID]["Parts"]
        nestHeight = data_all[nestID]["Height"]
        nestWidth = data_all[nestID]["Width"]

        keys = list(data.keys())
        keys_single, keys_dual = [], []
        for k in keys:
            if data[k]["Grabbability"]:
                if len(list(data[k]['Grab'].keys())) == 1:
                    keys_single.append(k)
                elif len(list(data[k]['Grab'].keys())) == 2:
                    keys_dual.append(k)

        nestMiddle = nestWidth/2 + data_all[nestID]["OffsetGrab"][0] if with_offset else nestWidth/2
        keys_single.sort(key=lambda x: data[x]['Grab'][0]['Grabpoint'])
        keys_single_left, keys_single_right = [], []
        for k in keys_single:
            if data[k]['Grab'][0]['Grabpoint'][0] <= nestMiddle:
                keys_single_left.append(k)
            else:
                keys_single_right.append(k)
        # keys_single_left = keys_single[:len(keys_single) // 2]
        # keys_single_right = keys_single[len(keys_single) // 2:]
        # if len(keys_single_right) > len(keys_single_left):
        #     keys_single_left.append(-1)
        keys_dual.sort(key=lambda x: data[x]['Grab'][0]['Grabpoint'])
        grab_times = len(keys_single) + len(keys_dual)

        # Sorting
        message_store_one, message_store_two = generate_store_messages(data,
                                                                       nestID,
                                                                       grab_times,
                                                                       keys_single_left,
                                                                       keys_single_right,
                                                                       keys_dual,
                                                                       (nestHeight, nestWidth),
                                                                       visualize=visualize
                                                                       )

        insert_function = insert_stack_strategy_offset if with_offset else insert_stack_strategy
        # id = len(select_result_all_from_table("cc_StackStrategyOffset"))+1 if with_offset else len(select_result_all_from_table("cc_StackStrategy"))+1
        for m in message_store_one:
            insert_function(m)
        for m in message_store_two:
            insert_function(m)


def store_insert_sql_stack(data_all, with_offset=False, visualize=False):
    for nestID in data_all.keys():
        if not data_all[nestID]["GrabStatus"]:
            continue
        data = data_all[nestID]["Parts"]
        nestHeight = data_all[nestID]["Height"]
        nestWidth = data_all[nestID]["Width"]

        keys = list(data.keys())
        keys_single, keys_dual = [], []
        for k in keys:
            if data[k]["Grabbability"]:
                if len(list(data[k]['Grab'].keys())) == 1:
                    keys_single.append(k)
                elif len(list(data[k]['Grab'].keys())) == 2:
                    keys_dual.append(k)

        nestMiddle = nestWidth/2
        keys_single.sort(key=lambda x: data[x]['Grab'][0]['Grabpoint'])
        keys_single_left, keys_single_right = [], []
        for k in keys_single:
            if data[k]['Grab'][0]['Grabpoint'][0] <= nestMiddle:
                keys_single_left.append(k)
            else:
                keys_single_right.append(k)

        keys_dual.sort(key=lambda x: data[x]['Grab'][0]['Grabpoint'])
        grab_times = len(keys_single) + len(keys_dual)

        # Sorting
        message_store_one, message_store_two = generate_store_messages(data,
                                                                       nestID,
                                                                       grab_times,
                                                                       keys_single_left,
                                                                       keys_single_right,
                                                                       keys_dual,
                                                                       (nestHeight, nestWidth),
                                                                       visualize=visualize
                                                                       )

        # delete before insert
        delete_stack_strategy_offset(nestID, data_all[nestID]['batchIndex'])
        insert_function = insert_stack_strategy_offset if with_offset else insert_stack_strategy
        for m in message_store_one:
            insert_function(m, data_all[nestID]['batchIndex'])
        for m in message_store_two:
            insert_function(m, data_all[nestID]['batchIndex'])


if __name__ == "__main__":
    reference_plate2conveyor = (0.0, 0.0)
    reference_conveyor2leveler = (0.0, 0.0)
    reference_gribber2workrests = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]

    config = {
        "Plate2Conveyor": reference_plate2conveyor,
        "Conveyor2Leveler": reference_conveyor2leveler,
        "Gribber2WorkRests": reference_gribber2workrests
    }

    groupID = "P201519A0900"
    nestID = "00000"

    data = torch.load("./weights/plan_result_{}.pth".format(groupID))
    keys = list(data.keys())
    keys_single, keys_dual = [], []
    for k in keys:
        if data[k]["Grabbability"]:
            if len(list(data[k]['Grab'].keys())) == 1:
                keys_single.append(k)
            elif len(list(data[k]['Grab'].keys())) == 2:
                keys_dual.append(k)

    keys_single.sort(key=lambda x: data[x]['Center'])
    keys_single_left = keys_single[:len(keys_single)//2]
    keys_single_right = keys_single[len(keys_single)//2:]
    keys_dual.sort(key=lambda x: data[x]['Center'])
    if len(keys_single_right) > len(keys_single_left):
        keys_single_left.append(-1)
    grab_times = len(keys)

    # Grab
    message_grab_one, message_grab_two = generate_grab_messages(data,
                                                                nestID,
                                                                grab_times,
                                                                keys_single_left,
                                                                keys_single_right,
                                                                keys_dual,
                                                                config)
    print(message_grab_one)
    print(message_grab_two)


    # Sorting
    message_store_one, message_store_two = generate_store_messages(data,
                                                                   nestID,
                                                                grab_times,
                                                                keys_single_left,
                                                                keys_single_right,
                                                                keys_dual,
                                                                config)

    print(message_store_one)
    print(message_store_two)



