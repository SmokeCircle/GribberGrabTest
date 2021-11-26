import numpy as np
import cv2
import torch
import argparse
from datetime import datetime

from utils.parse_dxf import *
from utils.parse_json import parse_json
from utils.workrest import WorkRest
from utils.conveyor import Conveyor
from utils.grab_estimate import *
from utils.store_plan import *
from utils.convey_plan import *
from utils.DB.DbResposity import *
from utils.data_to_sql import *
import os
os.chdir("/data/GribberGrabTest")

def plan(nestIDs, reverses, mirrors, taskID):
    '''
    1. parse the dxf or json
    :return:
    '''

    # Configuring
    for path in ["./output", "./output/pieces", "./output/kernel_vis", "./output/grab_results",
                 "./dump", "./dump/nests", "./dump/nests_grab_offset", "./dump/nests_store_offset", "./dump/grab_vis",
                 "./weights", "./dump/nest_grab_orders", "./dump/parts_to_store", "./dump/conveyor_vis"]:
        if not os.path.exists(path):
            os.makedirs(path)

    rootDir =  "/data/GribberGrabTest/data/dxf"
    dxfDir = os.path.join(rootDir, "{0:04d}{1:02d}{2:02d}".format(datetime.now().year, datetime.now().month, datetime.now().day))

    print("Configuring ...")
    CUDA = False
    config = {
        'CUDA': CUDA,
        'GribberKernelPath': "./weights/gribber_kernels_sort.pth",
    }

    workrestsS = {
        # 1: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'height_real': 1500, 'id': 0, 'technic': 1})],  # width_real 和 height_real 分别为料框的宽度和高度
        # 2: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 2})],
        # 3: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 3})],
        # 4: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 4})],
        # 5: [WorkRest({'cuda': CUDA, 'width_real': 2500, 'id': 0, 'technic': 5})]
    }

    workrestsM = {
        # 1: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'height_real': 1500, 'id': 0, 'technic': 1})],  # width_real 和 height_real 分别为料框的宽度和高度
        # 2: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 2})],
        # 3: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 3})],
        # 4: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 4})],
        # 5: [WorkRest({'cuda': CUDA, 'width_real': 6000, 'id': 0, 'technic': 5})]
    }

    workrestsL = {
        # 1: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'height_real': 1500, 'id': 0, 'technic': 1, 'simple': True})],  # width_real 和 height_real 分别为料框的宽度和高度
        # 2: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'height_real': 2500, 'id': 0, 'technic': 2, 'simple': True})],
        # 3: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'height_real': 2500, 'id': 0, 'technic': 3, 'simple': True})],
        # 4: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'height_real': 2500, 'id': 0, 'technic': 4, 'simple': True})],
        # 5: [WorkRest({'cuda': CUDA, 'width_real': 12000, 'height_real': 2500, 'id': 0, 'technic': 5, 'simple': True})]
    }

    workrests = {
        0: workrestsS,
        1: workrestsM,
        2: workrestsL
    }

    reference_grab2plate = {
        'LC-01': (0., 0.),
        'LC-02': (0., 0.),
    }
    reference_plate2conveyor = {
        'LC-01': (0.0, 0.0),
        'LC-02': (0.0, 0.0),  # need selection from sql
    }
    reference_conveyor2leveler = (0.0, 0.0)  # need selection from sql
    reference_grab2store = (0., 0.)


    references = {
        "Plate2ConveyorDict": reference_plate2conveyor,
        "Conveyor2Leveler": reference_conveyor2leveler,
        "Grab2Store": reference_grab2store,
        'PlatesDict': reference_grab2plate,
    }

    print("Parsing file ...")
    if nestIDs is None or len(nestIDs) == 0:
        print("No nest selected, terminating ...")
        exit(0)
    # parse dxf
    data = {}
    for nestID, rev, mir in zip(nestIDs, reverses, mirrors):
        # data.update(select_nest_info_by_nestID(nestID))  # changed to /ftp_root/yearmonthday/nestID.dxf
        data.update({nestID: {
            'Reverse': rev,
            'Mirror': mir
        }})

    for nestID in data.keys():
        part_info = select_part_info_by_nestID(nestID)
        # dxf = data[nestID]['LocalLink']
        dxf = os.path.join(dxfDir, "{}.dxf".format(nestID))
        partDB, height, width = parse_dxf_by_info(dxf, part_info, nestID, reverse=data[nestID]['Reverse'], mirror=data[nestID]['Mirror'])
        partDB = check_for_grab(partDB)
        data[nestID].update({
            "Parts": partDB,
            "Height": height,
            "Width": width,
        })

    print("Sorting ...")
    grab_plan_sort(data, config)
    visualize_grab_plan(data)
    convey_plan(data, config)

    print("Inserting Sort Strategy into SQL Server ...")
    for nestID in data.keys():
        insert_strategy_result(nestID, 1, data[nestID]['GrabStatus'], "")
    grab_insert_sql(data, references, visualize=True)

    print("Stacking ...")
    pieces_to_store, workrests = store_plan(data, config, workrests)
    print("Inserting Stack Strategy into SQL Server ...")
    for nestID in data.keys():
        insert_strategy_result(nestID, 2, data[nestID]['StoreStatus'], "")
    store_insert_sql(data)

    print("Saving nest planning results before offset...")
    for nestID in data.keys():
        torch.save(data[nestID], "./dump/nests/{}.pth".format(nestID))

    print("Saving parts to store ...")
    torch.save(pieces_to_store, "./dump/parts_to_store/{}.pth".
               format(taskID))
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nestID', type=str, nargs='+', help='the list of NestIDs needed to be planed in cc_NestInfo')
    parser.add_argument('--taskID', type=str, help='the taskID of the Nests need to be planed')
    parser.add_argument('--reverse', type=int, nargs='+', help='the list of bools indicating the nest needed to be reversed')
    parser.add_argument('--mirror', type=int, nargs='+', help='the list of bools indicating the nest needed to be mirrored')

    args = parser.parse_args()
    assert len(args.nestID) == len(args.reverse) == len(args.mirror), "The input parameter is not valid!"
    plan(args.nestID, args.reverse, args.mirror, args.taskID)
