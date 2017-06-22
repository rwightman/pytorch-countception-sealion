import numpy as np
import scipy
import cv2

scale = 1
patch_size = 32
framesize = 2256
num_outputs = 5


def calc_ef(patch_size=32, stride=1, kern="sq"):
    if kern == "sq":
        return (patch_size / stride)**2
    else:
        return 1.0


def gen_gaus_image(framesize, mx, my, cov=1):
    x, y = np.mgrid[0:framesize, 0:framesize]
    print(x, y)
    pos = np.dstack((x, y))
    print(pos, pos.shape)
    mean = [mx, my]
    cov = [[cov, 0], [0, cov]]
    rv = scipy.stats.multivariate_normal(mean, cov).pdf(pos)
    return rv / rv.sum()


def get_density(width, markers):
    gaus_img = np.zeros((width, width))
    for k, l in markers:
        gaus_img += gen_gaus_image(len(markers), k - patch_size // 2, l - patch_size // 2, cov)
    return gaus_img


def get_markers(coords):
    markers = np.zeros()
    return markers


def get_counts(markers, x, y, h, w, scale):
    types = [0] * num_outputs
    types[0] = markers[y:y + w, x:x + h].sum()
    return types


def get_targets(img, target_coords, base_x, base_y, stride, kern='sq'):
    width = (img.shape[1]) // stride
    height = (img.shape[0]) // stride
    print("label size: ", width)
    markers = get_markers(target_coords)
    targets = np.zeros((num_outputs, width, width))

    if kern == "sq":
        for x in range(width):
            for y in range(height):
                count = get_counts(markers, base_x + x*stride, base_y + y*stride, patch_size, patch_size)
                for i in range(num_outputs):
                    targets[i][y][x] = count[i]
    else:
        for i in range(0, num_outputs):
            targets[i] = get_density(width, markers[base_y:base_y + width, base_x:base_x + width])

    count_total = get_counts(markers, base_x, base_y, framesize + patch_size, framesize + patch_size, scale)
    return targets, count_total


def get_training_example_cells(img_raw, target_coords, base_x, base_y, stride):
    img = img_raw[base_y:base_y + framesize, base_x:base_x + framesize]
    img_pad = np.pad(img, (patch_size - 1) // 2, "constant")
    labels, count = get_targets(img_pad, target_coords, base_x, base_y, stride)
    return img, labels, count