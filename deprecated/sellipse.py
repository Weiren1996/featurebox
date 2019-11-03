# -*- coding: utf-8 -*-

# @Time    : 2019/11/3 15:14
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from itertools import product

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from scipy.stats import pearsonr


def corr_plot(x_cof, x_name=None, left_down=None, right_top=None, threshold_left=0, threshold_right=0.9,
              title="pearsonr coefficient", label_axis="off", front_raito=1):
    x_cof = np.round(x_cof, 2)

    name = x_name or list(range(x_cof.shape[0]))

    size = x_cof
    or_size = np.nan_to_num((abs(size) / size) * (1 - abs(size)))

    n = size.shape[0]
    explode = (0, 0)
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0, hspace=0)

    cmap = plt.get_cmap("bwr")  # args
    fill_colors = cmap(size / 2 + 0.5)  # args

    fig = plt.figure(figsize=(6, 6), frameon=True)  # args

    title_fontsize = round(15 * front_raito)  # c_args
    ax_fontsize = round(12 * front_raito)
    score_fontsize = round(8 * front_raito)
    circle_size = round(400 * front_raito)

    fig.text(0.5, 0.05, title, fontsize=title_fontsize, horizontalalignment='center',
             verticalalignment='center')  # zou, xia

    for i, j in product(range(n), range(n)):
        if j < i and abs(size[i, j]) >= threshold_left:
            types = left_down
        elif j > i and abs(size[i, j]) >= threshold_right:
            types = right_top
        else:
            types = None

        if types is "pie":
            ax = plt.subplot(gs[i, j])
            ax.pie((size[i, j], or_size[i, j]), explode=explode, labels=None, autopct=None, shadow=False,
                   startangle=90,
                   colors=[fill_colors[i, j], 'w'], wedgeprops=dict(width=1, edgecolor='black', linewidth=0.5),
                   counterclock=False,
                   frame=False, center=(0, 0), )
            ax.set_xlim(-1, 1)
            ax.axis('equal')

        elif types is "fill":
            ax = plt.subplot(gs[i, j])
            ax.set_facecolor(fill_colors[i, j])
            [ax.spines[_].set_color('w') for _ in ['right', 'top', 'left', 'bottom']]

            ax.text(0.5, 0.5, size[i, j],
                    fontdict={"color": "b"},  # args
                    fontsize=score_fontsize,  # c_arg
                    horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
        elif types is "text":
            ax = plt.subplot(gs[i, j])
            ax.text(0.5, 0.5, size[i, j],
                    fontdict={"color": "b"},  # args
                    fontsize=score_fontsize,  # c_arg
                    horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
            # plt.axis('off')
        elif types is "circle":
            ax = plt.subplot(gs[i, j])
            ax.axis('equal')
            ax.set_xlim(-1, 1)
            ax.scatter(0, 0, color=fill_colors[i, j], s=circle_size * abs(size[i, j]) ** 2)
            ax.set_xticks([])

            ax.set_yticks([])
            # plt.axis('off')

        else:
            pass

    for k in range(n):
        ax = plt.subplot(gs[k, k])
        ax.text(0.5, 0.5, name[k], fontsize=ax_fontsize, horizontalalignment='center', verticalalignment='center')
        ax.set_xticks([])
        ax.set_yticks([])
        if label_axis is "left":
            color = ["w", "w", "b", "b"]
            [ax.spines[i].set_color(j) for i, j in zip(['right', 'top', 'left', 'bottom'], color)]
        elif label_axis is "right":
            color = ["b", "b", "w", "w"]
            [ax.spines[i].set_color(j) for i, j in zip(['right', 'top', 'left', 'bottom'], color)]
        else:
            plt.axis('off')

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.8, 0.125, 0.05, 0.75])
    ColorbarBase(cbar_ax, cmap=cmap)
    fig.set_size_inches(7, 6, forward=True)
    plt.show()


if __name__ == '__main__':

    name0 = ['a', 'b', 'd', 'e', 'f', 'a', 'f', 'a'] * 2
    datax = np.random.rand(16, 16)


    def lin_cof(x0):
        results_list = []
        xx = x0.T
        yy = x0.T
        for a in xx:
            for b in yy:
                results = pearsonr(a, b)[0]
                results_list.append(results)
        results1 = np.array(results_list).reshape((x0.shape[-1], x0.shape[-1]))
        return results1


    x_cof = lin_cof(datax)
    corr_plot(x_cof, name0, left_down="circle", right_top="pie", threshold_right=0.4, label_axis="off")
