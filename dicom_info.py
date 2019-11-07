"""
Parse DICOM images
"""

import os
import pydicom
import cv2
import argparse

import matplotlib.pyplot as plt

from pydicom.data import get_testdata_files

DEFAULT_DICOM = '/home/jakub/inz/DICOM/S00002/SER00002/I00001'


def show_matplotlib(ds):
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.show()


def show_cv2(ds):
    mul = 2 ** (ds.BitsAllocated - ds.BitsStored)
    img = ds.pixel_array * mul
    rows = ds.Rows
    columns = ds.Columns
    img = cv2.resize(img, (int(columns / 2), int(rows / 2)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_stats(ds):
    # img = ds.pixel_array.flatten()
    # print(img.flat)
    # print(min(img))
    # print(max(img))
    print(str(ds.Columns) + ' x ' + str(ds.Rows))


def main(filename):
    # filename = '/home/jakub/inz/DICOM/S00002/SER00002/I00001'
    # filename = '/home/jakub/inz/series-000001/image-000001.dcm'
    with pydicom.dcmread(filename) as ds:
        # print(ds)
        # show_cv2(ds)
        img_stats(ds)
        # show_mareadtplotlib()


def output_file(path):
    return os.path.realpath(os.path.normpath(os.path.expanduser(path)))


def source_file(path):
    if os.path.exists(path):
        return output_file(path)
    else:
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dicom', metavar='DICOM', type=source_file, nargs='?', default=DEFAULT_DICOM,
                        help='DICOM file path')
    parser.add_argument('dictionary', metavar='dictionary', type=source_file, nargs='?',
                        help='user defined dictionary file path')
    parser.add_argument('output', metavar='output', type=output_file, nargs='?',
                        help='parsing results file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    # if args.dicom:
    main(args.dicom)
    # else:
    #     print('wrong file')
    # main()
