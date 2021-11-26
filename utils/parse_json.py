import json
import cv2
import numpy as np
from collections import Counter

def parse_json(json_path):

    with open(json_path,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
        # print('这是文件中的json数据：',json_data)
        # print('这是读取到文件数据的数据类型：', type(json_data))

    json_data = json.loads(json_data['data'])

    groupID = json_data['GroupID']
    nestID = json_data['NestID']
    plate = json_data['Plate']
    part_list = json_data['PartList']

    Xs = []
    Ys = []
    for s in plate['SegmentList']:
        if s['type'] == 'line':
            Xs.append(float(s['X1']))
            Xs.append(float(s['X2']))
            Ys.append(float(s['Y1']))
            Ys.append(float(s['Y2']))

    width = np.ceil(max(Xs) - min(Xs)).astype(np.int)
    height = np.ceil(max(Ys) - min(Ys)).astype(np.int)

    tl = (min(Xs), min(Ys))
    br = (max(Xs), max(Ys))

    overall = np.zeros([height, width], dtype=np.uint8)

    pieces = []
    origins = []
    ids = []
    sns = []
    technics = []
    nestPlanIDs = []

    for p in part_list:
        id = p['PartName']
        sn = p['PartSN'] if 'PartSN' in p.keys() else '0'
        tech = p['Technics'] if 'Technics' in p.keys() else '0'
        pid = p['NestPlanID'] if 'NestPlanID' in p.keys() else '0'
        contours = p['contours']

        technics.append(tech)
        nestPlanIDs.append(pid)

        rects = []
        for c in contours:
            outside = c['IsOutsideContour']
            color = 255 if outside else 0
            seg = c['SegmentList']
            elements = []
            for s in seg:
                if s['type'] == 'line':
                    # type (X1, Y1), (X2, Y2)
                    elements.append((
                        'line',
                        (np.round(float(s['X1']) - tl[0]).astype(np.int), np.round(float(s['Y1']) - tl[1]).astype(np.int)),
                        (np.round(float(s['X2']) - tl[0]).astype(np.int), np.round(float(s['Y2']) - tl[1]).astype(np.int))
                    ))
                elif s['type'] == 'arc':
                    # type (XC, YC), R, (Angle1, Angle2), clockwise
                    elements.append((
                        'arc',
                        (np.round(float(s['XC']) - tl[0]).astype(np.int), np.round(float(s['YC']) - tl[1]).astype(np.int)),
                        np.round(float(s['R'])).astype(np.int),
                        (float(s['Angle1']), float(s['Angle2'])),
                        bool(s['clockwise'])
                    ))  # TODO: Ensure the angles are degrees
            temp = np.zeros_like(overall)
            for p in elements:
                if p[0] == 'line':
                    cv2.line(temp, p[1], p[2], 255, thickness=3)
                elif p[0] == 'arc':
                    start_angle = p[3][0]
                    end_angle = p[3][1]
                    if not p[-1]:
                        start_angle, end_angle = end_angle, 360. - start_angle
                    cv2.ellipse(temp, p[1], (p[2], p[2]), 0.0, start_angle, end_angle, 255, thickness=3)

            cv2.namedWindow("temp", cv2.WINDOW_NORMAL)
            cv2.imshow("temp", temp)
            cv2.waitKey(0)

            contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [contours[i] for i in range(len(contours))]
            contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
            for i in range(len(contours)):
                cnt = contours[i]
                cv2.drawContours(overall, [cnt], 0, color, -1)
            cv2.drawContours(overall, [contours[0]], 0, color, -1)
            rect = cv2.boundingRect(contours[0])
            rects.append(rect)

        rects.sort(key=lambda x: x[2] * x[3], reverse=True)
        rect = rects[0]
        piece = overall[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        pieces.append(piece)
        origins.append((rect[0], rect[1]))  # x, y
        ids.append(id)
        sns.append(sn)

    zipped = sorted(zip(ids, sns, origins, pieces, technics, nestPlanIDs), key=lambda x: x[-1].shape[0]*x[-1].shape[1],
                    reverse=True)  # from large to small
    data = {}
    for n, (i, s, o, p, t, n) in enumerate(zipped):
        data[n] = {
            'PartID': i,
            'PartSN': s,
            'Origin': o,
            'Part': p,
            'Technics': t,
            'NestPlanID': n
        }

    nestPlanIDs_count = Counter(nestPlanIDs)
    nestPlanIDs_main = nestPlanIDs_count.most_common(1)[0][0]

    #     cv2.namedWindow("piece", cv2.WINDOW_NORMAL)
    #     cv2.imshow("piece", piece)
    #     cv2.namedWindow("overall", cv2.WINDOW_NORMAL)
    #     cv2.imshow("overall", overall)
    #     cv2.waitKey(0)
    #
    cv2.namedWindow("overall", cv2.WINDOW_NORMAL)
    cv2.imshow("overall", overall)
    cv2.waitKey(0)
    print(len(pieces))
    print("Done")

    return {nestID: {
        "MainPlan": nestPlanIDs_main,
        "PlanCount": nestPlanIDs_count,
        'Parts': data,
    }}

if __name__ == '__main__':
    data = parse_json("./data/json/test.json")
    print(data)