# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def broke_line(data_list):
    plt.plot(range(len(data_list)), data_list)
    plt.show()


def bar_chart(data_list):
    plt.bar(left=range(len(data_list)), height=data_list)
    plt.show()


if __name__ == '__main__':
    # broke_line([9.0, 10.0, 9.0, 21.0, 10.0, 8.0, 8.0, 9.0, 8.0, 8.0])
    bar_chart([9.0, 10.0, 9.0, 21.0, 10.0, 8.0, 8.0, 9.0, 8.0, 8.0])
