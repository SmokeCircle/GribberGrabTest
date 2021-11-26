from logging import exception
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import cv2
import numpy as np
import os

def dxf2img(dxf, nestID=None, reverse=False):
    doc = ezdxf.readfile(dxf)
    img_path = "output/sample_{}.png".format(nestID) if nestID is not None else "output/sample.png"
    msp = doc.modelspace()
    margin = 5

    text_for_parts = []
    textp = msp.query('TEXT[layer!="plane"]')
    for t in textp:
        text_for_parts.append(t.dxf.text)

    redundants = msp.query('*[layer!="plane"]')
    for e in redundants:
        msp.delete_entity(e)
        e.destroy()

    textdb = {}
    textsz = {}
    texts = msp.query('TEXT[layer=="plane"]')
    print("There should be {} parts.".format(len(texts)))
    num_expected = len(texts)
    for t in texts:
        key = t.dxf.text
        height = t.dxf.height
        if key not in textdb.keys():
            textdb[key] = []
        textdb[key] = textdb[key] + [(t.get_pos()[1][0], t.get_pos()[1][1])]
        textsz[key] = height
        msp.delete_entity(t)
        t.destroy()

    lines = msp.query('LINE[layer=="plane"]')
    lf_x = 99999999
    lf_y = 99999999
    hr_x = -9999999
    hr_y = -9999999
    for l in lines:
        point1 = l.dxf.start
        point2 = l.dxf.end

        lf_x = min(point1[0], point2[0], lf_x)
        lf_y = min(point1[1], point2[1], lf_y)
        hr_x = max(point1[0], point2[0], hr_x)
        hr_y = max(point1[1], point2[1], hr_y)

    # print("The lower left point is {}, the higher right point is {}".format((lf_x, lf_y), (hr_x, hr_y)))

    resize = (np.round(hr_x - lf_x).astype(np.int), np.round(hr_y - lf_y).astype(np.int))

    auditor = doc.audit()
    if len(auditor.errors) != 0:
        raise exception("The DXF document is damaged and can't be converted!")
    else:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ctx = RenderContext(doc)
        ctx.set_current_layout(msp)
        ctx.current_layout.set_colors(bg='#FFFFFF')
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp, finalize=True)
        # Frontend(ctx, out).draw_entities(lines)
        # fig.savefig(img_path)

        fig.savefig(img_path, dpi=1500)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 210, 255, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.boundingRect(contours[0])
    img = img[rect[1]+margin: rect[1] + rect[3]-margin, rect[0]+margin: rect[0] + rect[2]-margin]
    img = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    img = cv2.resize(img, (resize))
    for k in list(textdb.keys()):
        for i in range(len(textdb[k])):
            textdb[k][i] = (np.round(textdb[k][i][0] - lf_x).astype(int),
                            np.round(hr_y - textdb[k][i][1]).astype(int))

    if reverse:
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)

        for k in list(textdb.keys()):
            for i in range(len(textdb[k])):
                textdb[k][i] = (np.round(resize[0] - textdb[k][i][0]).astype(int),
                                np.round(resize[1] - textdb[k][i][1]).astype(int))

    cv2.imwrite(img_path, img)
    return img, textdb, textsz, num_expected


def inner_level(index, hierarchy):
    level = 0
    while hierarchy[0][index][2] != -1:
        level += 1
        index = hierarchy[0][index][2]
    return level


def inner_level_target(index, hierarchy, target):
    level = 0
    while hierarchy[0][index][2] != -1:
        level += 1
        index = hierarchy[0][index][2]
    return level == target


def outer_level(index, hierarchy):
    level = 0
    while hierarchy[0][index][-1] != -1:
        level += 1
        index = hierarchy[0][index][-1]
    return level


def outer_level_target(index, hierarchy, target):
    level = 0
    while hierarchy[0][index][-1] != -1:
        level += 1
        index = hierarchy[0][index][-1]
    return level == target


def count_max_level(contours, hierarchy):
    level_global = 0
    for i in range(len(contours)):
        level = 0
        index = i
        while hierarchy[0][index][-1] != -1:
            level += 1
            index = hierarchy[0][index][-1]
        level_global = max(level, level_global)
    return level_global


def segmentPieces(img, textDB, textSZ, num_expected):
    thickness = 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 180, 255, 1)


    # thresh = cv2.GaussianBlur(thresh, (3, 3), 1.5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("./output/thresh.png", thresh)

    externals = []
    pieces = []
    origins = []
    rects = []

    max_level = count_max_level(contours, hierarchy)  # 3/5/7/9

    if max_level == 9:
        temp = np.zeros_like(gray)
        inverse = temp.copy()
        for i in range(len(contours)):
            if outer_level_target(i, hierarchy, max_level-1):
                cv2.drawContours(temp, contours, i, 255, thickness)
                cv2.drawContours(inverse, contours, i, 255, -1)
                rect = cv2.boundingRect(contours[i])
                piece = temp[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]].copy()
                piece = cv2.bitwise_not(piece, piece)
                pieces.append(piece)
                externals.append(contours[i])
                origins.append((rect[0], rect[1]))
                rects.append(rect)

        inverse = cv2.bitwise_not(inverse, inverse)
        thresh = cv2.bitwise_and(thresh, inverse)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite("./output/thresh_cut.png", thresh)
        max_level = count_max_level(contours, hierarchy)  # 7

    canvas = np.zeros_like(gray)
    inners = canvas.copy()
    middles = canvas.copy()
    # draw all the inner contours
    inner_nohole_level = max_level - 2 if max_level == 7 else max_level
    for i in range(len(contours)):
        if hierarchy[0][i][2] == -1:
            if outer_level_target(i, hierarchy, 1):
                continue
            cv2.drawContours(canvas, contours, i, 255, thickness)
            cv2.drawContours(inners, contours, i, 255, thickness)
            if outer_level_target(i, hierarchy, 3) and max_level > 3:
                cv2.drawContours(middles, contours, i, 255, thickness)
            if outer_level(i, hierarchy) == inner_nohole_level and max_level > 3:
                temp = inners.copy()
                rect = cv2.boundingRect(contours[i])
                piece = temp[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]].copy()
                piece = cv2.bitwise_not(piece, piece)
                pieces.append(piece)
                externals.append(contours[i])
                origins.append((rect[0], rect[1]))
                rects.append(rect)
                inners[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]] = 0

    # inners_save = cv2.bitwise_not(inners, inners)
    # inners_save = cv2.cvtColor(inners_save, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite("./output/inners.png", inners_save)
    # cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
    # cv2.imshow("Canvas", canvas)
    # cv2.waitKey(0)

    # draw the contours in the hole
    for i in range(len(contours)):
        if (hierarchy[0][i][2] != -1) and (hierarchy[0][i][-1] != -1):
            if inner_level_target(i, hierarchy, 3) and outer_level_target(i, hierarchy, 4):
                temp = inners.copy()
                inverse = np.zeros_like(temp)
                cv2.drawContours(canvas, contours, i, 255, thickness)
                cv2.drawContours(temp, contours, i, 255, thickness)
                cv2.drawContours(inverse, contours, i, 255, -1)
                temp = cv2.bitwise_and(temp, inverse)
                rect = cv2.boundingRect(contours[i])
                piece = temp[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]].copy()
                piece = cv2.bitwise_not(piece, piece)
                pieces.append(piece)
                externals.append(contours[i])
                origins.append((rect[0], rect[1]))
                rects.append(rect)
                inners[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]] = 0
            elif (inner_level_target(i, hierarchy, 5) and outer_level_target(i, hierarchy, 2)) or \
                    ((inner_level_target(i, hierarchy, 3) and outer_level_target(i, hierarchy, 2))):
                cv2.drawContours(canvas, contours, i, 255, thickness)
                cv2.drawContours(inners, contours, i, 255, thickness)
                cv2.drawContours(middles, contours, i, 255, thickness)

    # cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
    # cv2.imshow("Canvas", canvas)
    # cv2.waitKey(0)

    # draw the external contours
    for i in range(len(contours)):
        if hierarchy[0][i][-1] == -1:
            temp = middles.copy()
            inverse = np.zeros_like(temp)
            cv2.drawContours(canvas, contours, i, 255, thickness)
            cv2.drawContours(temp, contours, i, 255, thickness)
            cv2.drawContours(inverse, contours, i, 255, -1)
            temp = cv2.bitwise_and(temp, inverse)

            rect = cv2.boundingRect(contours[i])
            piece = temp[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]].copy()
            piece = cv2.bitwise_not(piece, piece)
            pieces.append(piece)
            externals.append(contours[i])
            origins.append((rect[0], rect[1]))
            rects.append(rect)

    # for i, p in enumerate(pieces):
    #     cv2.imwrite("./output/pieces/{0:03d}.png".format(i), p)

    print("Detected {} parts".format(len(pieces)))
    num_detected = len(pieces)

    canvas = cv2.bitwise_not(canvas, canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for n in list(textDB.keys()):
        for p in textDB[n]:
            cv2.circle(canvas, (p[0], p[1]), 5, (255, 0, 0), -1)

    zipped = sorted(zip(externals, pieces, origins, rects), key=lambda x: cv2.contourArea(x[0]))
    text_keys = sorted(list(textDB.keys()), key=lambda x: textSZ[x])

    # Assign ID to the pieces
    partDB = {}
    textleft = []
    count = 0
    pop_count = 0
    for n in list(text_keys):
        for p in textDB[n]:
            assigned = False
            i = 0
            for epo in zipped:
                contour, piece, origin, rect = epo
                if cv2.pointPolygonTest(contour, p, False) >= 1:
                    partDB[count] = {
                        'PartID': n,
                        'Part': piece,
                        'Origin': origin,
                        'PartHeight': piece.shape[0],  # 使用从dxf解析出的图形尺寸，不使用表格中的尺寸
                        'PartWidth': piece.shape[1],
                    }
                    count += 1
                    assigned = True
                    break
                i += 1
            if assigned:
                zipped.pop(i)
                pop_count += 1
            else:
                textleft.append((n, p))

    if len(zipped) > 0 and len(textleft) > 0:
        assert len(zipped) == len(textleft), "Error: the parts left doesn't match the text left"
        for epo in zipped:
            contour, piece, origin, rect = epo
            for n, p in textleft:
                if rect[0] <= p[0] <= rect[0] + rect[2] and rect[1] <= p[1] <= rect[1] + rect[3]:
                    partDB[count] = {
                        'PartID': n,
                        'Part': piece,
                        'Origin': origin,
                        'PartHeight': piece.shape[0],  # 使用从dxf解析出的图形尺寸，不使用表格中的尺寸
                        'PartWidth': piece.shape[1],
                    }
                    count += 1

    if not os.path.exists("./output/pieces"):
        os.makedirs("./output/pieces")
    for k in partDB.keys():
        cv2.imwrite("./output/pieces/{0:03d}_id{1:03d}.png".format(k, int(partDB[k]['PartID'])), partDB[k]['Part'])

    print("Assigned {} parts.".format(len(list(partDB.keys()))))
    num_assigned = len(list(partDB.keys()))

    cv2.imwrite("./output/canvas.png", canvas)
    assert num_expected == num_assigned == num_detected, "The nest is parsed illegally."

    return partDB

def parse_dxf_by_info(dxf, partInfo, nestID):
    img, textDB, textSZ, num = dxf2img(dxf, nestID, reverse=True)
    partDB = segmentPieces(img, textDB, textSZ, num)
    for key in partDB.keys():
        partID = partDB[key]['PartID']
        partDB[key].update(partInfo[partID])
    height, width, _ = img.shape
    return partDB, height, width

if __name__ == "__main__":
    # dxf = "data/dxf/O210811SG100015A02.dxf"
    # dxf = "data/dxf/O210812Q345B8A01.dxf"
    # dxf = "data/dxf/O210812Q345B8A03.dxf"
    # dxf = "data/dxf/O210908Q235B10A01.dxf"
    # dxf = "data/dxf/O210811SG100015AT22.dxf"
    # dxf = "data/dxf/O210917SG10008A07T1.dxf"
    # dxf = "data/dxf/O210918SG100035A01.dxf"
    # dxf = "data/dxf/O210918SG100020A01.dxf"
    # dxf = "data/dxf/O210919SG100010A01.dxf"
    # dxf = "data/dxf/O210919SG100010A02.dxf"
    # dxf = "data/dxf/O210924SG100015A03R.dxf"
    dxf = "data/dxf/O210929SG10008A01.dxf"

    img, textDB, textSZ, num = dxf2img(dxf, reverse=False)
    partDB = segmentPieces(img, textDB, textSZ, num)
    print(textDB)
