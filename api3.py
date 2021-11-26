import numpy as np
import cv2
import torch
import argparse

from utils.parse_dxf_org import *
from utils.parse_json import parse_json
from utils.workrest import WorkRest
from utils.grab_estimate import *
from utils.store_plan import *
from utils.convey_plan import *
from utils.DB.DbResposity import *
from utils.data_to_sql import *
import os
os.chdir("/data/GribberGrabTest")


def plan(nestIDs):
    '''
    1. parse the dxf or json
    :return:
    '''

    # Configuring
    for path in ["./output", "./output/pieces", "./output/kernel_vis", "./output/grab_results",
                 "./dump", "./dump/nests", "./dump/nests_grab_offset", "./dump/nests_store_offset", "./weights"]:
        if not os.path.exists(path):
            os.makedirs(path)

    print("Configuring ...")
    reference_grab2plate = {
        'LC-01': (6443.1, 10595.2),  # 平面切割机原点V点坐标 (x, y)
        'LC-02': (6408.02, 590.9),  # 坡口切割机原点W点坐标  (x, y)
    }
    reference_plate2conveyor = {
        'LC-01': (0.0, -4228.0),  # (0.0, 线段X1), V点到传送带的距离
        'LC-02': (0.0, 3827),  # (0.0, 线段X2)， W点到传送带的距离
    }
    reference_conveyor2leveler = (36600, 0.0)  # (线段d3, 0.0)，工件在传送带经过校平后的距离
    reference_grab2store = (-37290., 0.)  # (B - A, 0.0), 下料桁架坐标系下的分拣桁架坐标系原点A点的坐标
    reference_plateGrabSplit = {
        'LC-01': [(0.0, reference_grab2plate['LC-01'][1]-180),  # (0.0, 按照LC-01切割机的坐标上下分割添加offset，避免LC-01上方与立柱的干涉)
                  (0.0, reference_grab2plate['LC-01'][1]-1722),  # (0.0, 按照LC-01切割机的坐标上下分割添加offset，避免LC-01下方与传送带和LC-02的干涉)
                  ],
        'LC-02': [(0.0, reference_grab2plate['LC-02'][1]+380),  # (0.0, 按照LC-02切割机的坐标上下分割添加offset，避免LC-02下方与立柱的干涉)
                  (0.0, reference_grab2plate['LC-02'][1]+2247),  # (0.0, 按照LC-02切割机的坐标上下分割添加offset，避免LC-02上方与传送带和LC-01的干涉)
                  ]
    }

    references = {
        "Plate2ConveyorDict": reference_plate2conveyor,
        "Conveyor2Leveler": reference_conveyor2leveler,
        "Grab2Store": reference_grab2store,
        "PlatesDict": reference_grab2plate,
        "SplitDict": reference_plateGrabSplit,
    }

    print("Parsing file ...")
    if nestIDs is None or len(nestIDs) == 0:
        print("No nest selected, terminating ...")
        exit(0)
    # parse dxf
    data = {}
    for nestID in nestIDs:
        groupID = nestID.split('_')[0]
        data.update({nestID: torch.load("./dump/nests/{}.pth".format(groupID))})

    # Add grab offset:
    print("Adding offset ...")
    for nestID in data.keys():
        alignment = select_alignment_result_by_nestID(nestID)
        nestHeight = data[nestID]["Height"]
        nestWidth = data[nestID]["Width"]
        device_code = alignment['DeviceCode']  # 'LC-01', 'LC-02'
        data[nestID]["DeviceCode"] = device_code
        ref = reference_grab2plate[device_code]
        offset_x = float(alignment['Offset_X'])
        offset_y = float(alignment['Offset_Y'])
        offset_a = float(alignment['Offset_A'])

        margin_grab = (offset_x, offset_y, offset_a)

        if device_code == 'LC-01':
            offset_x = ref[0] + offset_x - (nestWidth*math.cos(offset_a/180*math.pi) + nestHeight*math.sin(offset_a/180*math.pi))
            offset_y = ref[1] + offset_y - (-nestWidth*math.sin(offset_a/180*math.pi) + nestHeight*math.cos(offset_a/180*math.pi))
        elif device_code == 'LC-02':
            offset_x = ref[0] + offset_x - nestWidth * math.cos(offset_a/180*math.pi)
            offset_y = ref[1] + offset_y + nestWidth * math.sin(offset_a/180*math.pi)
        else:
            raise ValueError("The device_code is not valid!")

        offset_grab = (offset_x, offset_y, offset_a)
        # offset_grab = (0, 0, 0)
        data[nestID]["OffsetGrab"] = offset_grab
        data[nestID]["MarginGrab"] = margin_grab
        if alignment['AlignmentResult']:
            for n in data[nestID]["Parts"].keys():
                part = data[nestID]["Parts"][n]
                if part['Grabbability']:
                    part = grab_apply_offset(part, offset_grab, references, device_code=data[nestID]['DeviceCode'])
                data[nestID]["Parts"][n] = part
    # visualize_grab_plan(data, with_offset=True)

    # add offset to the Convey
    convey_apply_offset_all(data)

    print("Inserting Sort Strategy with offset into SQL Server ...")
    grab_insert_sql(data, references, with_offset=True)

    print("Saving nest planning results after grab offset...")
    for nestID in data.keys():
        torch.save(data[nestID], "./dump/nests_grab_offset/{}.pth".format(nestID))
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nestID', type=str, nargs='+', help='the list of NestIDs needed to be planed in cc_NestInfo')
    args = parser.parse_args()
    plan(args.nestID)
