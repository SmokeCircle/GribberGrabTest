import ezdxf
import argparse

def dxf2info(dxf):
    doc = ezdxf.readfile(dxf)
    msp = doc.modelspace()

    text_for_parts = []
    textp = msp.query('TEXT[layer!="plane"]')
    for t in textp:
        text_for_parts.append(t.dxf.text)

    return text_for_parts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dxf_path', type=str, help='dxf path')
    args = parser.parse_args()
    info = dxf2info(args.dxf_path)
    print(info)