#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division  # Division in Python 2.7
import matplotlib

matplotlib.use('Agg')  # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import unittest

from matplotlib import colors


def plot_color_gradients(gradients, names):
    # For pretty latex fonts (commented out, because it does not work on some machines)
    # rc('text', usetex=True)
    # rc('font', family='serif', serif=['Times'], size=10)
    # rc('legend', fontsize=10)

    column_width_pt = 400  # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)

    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3] / 2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('gradients.pdf')


def hsv2rgb(h, s, v):
    c = v * s

    x = c * (1 - abs((h / 60) % 2 - 1))

    m = v - c

    if h >= 0 and h < 60:
        Rprim, Gprim, Bprim = (c, x, 0)
    elif h >= 60 and h < 120:
        Rprim, Gprim, Bprim = (x, c, 0)
    elif h >= 120 and h < 180:
        Rprim, Gprim, Bprim = (0, c, x)
    elif h >= 180 and h < 240:
        Rprim, Gprim, Bprim = (0, x, c)
    elif h >= 240 and h < 300:
        Rprim, Gprim, Bprim = (x, 0, c)
    elif h >= 300 and h < 360:
        Rprim, Gprim, Bprim = (c, 0, x)
    elif h == 360:
        Rprim, Gprim, Bprim = (c, x, 0)
    else:
        raise ValueError

    return (Rprim + m, Gprim + m, Bprim + m)


def gradient_rgb_bw(v):
    return (v, v, v)


def gradient_rgb_gbr(v):
    if v <= 0.5:
        return (0, 1 - v * 2, v * 2)
    else:
        return ((v - 0.5) * 2, 0, 1 - ((v - 0.5) * 2))


def gradient_rgb_gbr_full(v):
    thresholdValue = 0.25
    rangeAmount = 4

    if v <= thresholdValue:
        return (0, 1, v * rangeAmount)
    elif v > thresholdValue and v <= 2 * thresholdValue:
        return (0, 1 - ((v - thresholdValue) * rangeAmount), 1)
    elif v > 2 * thresholdValue and v <= 3 * thresholdValue:
        return (((v - 2 * thresholdValue) * rangeAmount), 0, 1)
    else:
        return (1, 0, 1 - ((v - 3 * thresholdValue) * rangeAmount))


def gradient_rgb_wb_custom(v):
    thresholdValue = 1 / 7
    rangeAmount = 7

    if v <= thresholdValue:
        return (1, 1 - (v * rangeAmount), 1)
    elif v > thresholdValue and v <= 2 * thresholdValue:
        return (1 - ((v - thresholdValue) * rangeAmount), 0, 1)
    elif v > 2 * thresholdValue and v <= 3 * thresholdValue:
        return (0, ((v - 2 * thresholdValue) * rangeAmount), 1)
    elif v > 3 * thresholdValue and v <= 4 * thresholdValue:  #
        return (0, 1, 1 - ((v - 3 * thresholdValue) * rangeAmount))
    elif v > 4 * thresholdValue and v <= 5 * thresholdValue:
        return (((v - 4 * thresholdValue) * rangeAmount), 1, 0)
    elif v > 5 * thresholdValue and v <= 6 * thresholdValue:
        return (1, 1 - ((v - 5 * thresholdValue) * rangeAmount), 0)
    else:
        return (1 - ((v - 6 * thresholdValue) * rangeAmount), 0, 0)


def gradient_hsv_bw(v):
    return hsv2rgb(0, 0, v)


def gradient_hsv_gbr(v):
    return hsv2rgb(v * 240 + 120, 1, 1)


def gradient_hsv_unknown(v):
    return hsv2rgb(120 - v * 120, 0.4, 1)


def gradient_hsv_custom(v):
    return hsv2rgb(360 - v * 360, v, 1)


def gradient_hsv_gr(v):
    return hsv2rgb(120 - (v * 120), 1, 1)

def normalize_image(imageData):
    min = np.amin(imageData)
    heightRange = np.amax(imageData) - min

    for x in np.nditer(imageData, op_flags=['readwrite']):
        x[...] = (x - min) / heightRange

    return imageData

def color_image(imageData):

    imageHeight=500
    imageWidth=500
    img = np.zeros((imageWidth, imageHeight, 3))

    # imageData=imageData[...,np.newaxis]
    #
    # for x in np.nditer(imageData, op_flags=['readwrite']):
    #     x[...] = gradient_hsv_gr(x)


    return np.array([[gradient_hsv_gr(imageData[j,i]) for i in range(0,500)] for j in range(0,500)])

def plot_colored_map():
    data = np.loadtxt('big.dem', skiprows=1)

    data = normalize_image(data)

    colored_data = color_image(data)

    plt.imshow(colored_data)
    plt.savefig('colored_map.pdf')


class HSV2RGBtests(unittest.TestCase):
    def test_black(self):
        self.assertEqual(hsv2rgb(0, 0, 0), (0, 0, 0))

    def test_lime(self):
        self.assertEqual(hsv2rgb(120, 1, 1), (0, 1, 0))

    def test_silver(self):
        self.assertEqual(hsv2rgb(0, 0, 0.75), (0.75, 0.75, 0.75))

    def test_navy(self):
        self.assertEqual(hsv2rgb(240, 1, 0.5), (0, 0, 0.5))

    def test_cyan(self):
        self.assertEqual(hsv2rgb(180, 1, 1), (0, 1, 1))

    def test_teal(self):
        self.assertEqual(hsv2rgb(180, 1, 0.5), (0, 0.5, 0.5))

    def test_maroon(self):
        self.assertEqual(hsv2rgb(0, 1, 0.5), (0.5, 0, 0))


if __name__ == '__main__':
    # unittest.main()

    plot_colored_map()


    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()


    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])
