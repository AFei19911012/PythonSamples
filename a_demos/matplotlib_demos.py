# -*- coding: utf-8 -*-
"""
 Created on 2021/6/27 20:09
 Filename   : matplotlib_demos.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description: https://seaborn.pydata.org/examples/index.html
"""

# =======================================================
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import joypy
import warnings

sns.set_theme(style="white")

warnings.simplefilter('ignore')
warnings.filterwarnings(action='once')


def line_plot():
    rs = np.random.RandomState(365)
    values = rs.randn(365, 4).cumsum(axis=0)
    dates = pd.date_range("1 1 2016", periods=365, freq="D")
    data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
    data = data.rolling(7).mean()
    sns.lineplot(data=data, linewidth=2.5)
    plt.legend(loc='upper left')
    plt.show()


def multiyy_plot():
    df = pd.read_csv("data/economics.csv")

    x = df['date']
    y1 = df['psavert']
    y2 = df['unemploy']

    # Plot Line1 (Left Y Axis)
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
    ax1.plot(x, y1, color='tab:red')

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y2, color='tab:blue')

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel('Year', fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel('Personal Savings Rate', color='tab:red', fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red')
    ax1.grid(alpha=.4)

    # ax2 (right Y axis)
    ax2.set_ylabel("# Unemployed (1000's)", color='tab:blue', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_xticks(np.arange(0, len(x), 60))
    ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize': 10})
    ax2.set_title("Personal Savings Rate vs Unemployed: Plotting in Secondary Y Axis", fontsize=22)
    fig.tight_layout()
    plt.show()


def time_series_with_peaks():
    df = pd.read_csv('data/AirPassengers.csv')

    # Get the Peaks and Troughs
    data = df['traffic'].values
    doublediff = np.diff(np.sign(np.diff(data)))
    peak_locations = np.where(doublediff == -2)[0] + 1

    doublediff2 = np.diff(np.sign(np.diff(-1 * data)))
    trough_locations = np.where(doublediff2 == -2)[0] + 1

    # Draw Plot
    plt.figure(figsize=(16, 10), dpi=80)
    plt.plot('date', 'traffic', data=df, color='tab:blue', label='Air Traffic')
    plt.scatter(df.date[peak_locations], df.traffic[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
    plt.scatter(df.date[trough_locations], df.traffic[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')

    # Annotate
    for t, p in zip(trough_locations[1::5], peak_locations[::3]):
        plt.text(df.date[p], df.traffic[p] + 15, df.date[p], horizontalalignment='center', color='darkgreen')
        plt.text(df.date[t], df.traffic[t] - 35, df.date[t], horizontalalignment='center', color='darkred')

    # Decoration
    plt.ylim(50, 750)
    xtick_location = df.index.tolist()[::6]
    xtick_labels = df.date.tolist()[::6]
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)
    plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.yticks(fontsize=12, alpha=.7)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=.3)
    plt.show()


def scatter_plot():
    midwest = pd.read_csv("data/midwest_filter.csv")
    """ Prepare Data """
    """ Create as many colors as there are unique midwest['category'] """
    categories = np.unique(midwest['category'])
    colors = [plt.cm.tab10(i / float(len(categories) - 1)) for i in range(len(categories))]
    """ Draw Plot for Each Category """
    plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
    for i, category in enumerate(categories):
        plt.scatter('area', 'poptotal',
                    data=midwest.loc[midwest.category == category, :],
                    s=20, c=colors[i], label=str(category))
    """ Decorations """
    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000), xlabel='Area', ylabel='Population')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)
    plt.legend(fontsize=12)
    plt.show()


def marginal_histogram():
    df = pd.read_csv("data/mpg_ggplot2.csv")
    """ Create Fig and gridspec """
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
    """ Define the axes """
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
    """ Scatterplot on main ax """
    ax_main.scatter('displ', 'hwy', s=df.cty * 4, c=df.manufacturer.astype('category').cat.codes, alpha=.9, data=df, cmap="tab10", edgecolors='gray',
                    linewidths=.5)
    """ histogram on the right """
    ax_bottom.hist(df.displ, 40, histtype='stepfilled', orientation='vertical', color='deeppink')
    ax_bottom.invert_yaxis()
    """ histogram in the bottom """
    ax_right.hist(df.hwy, 40, histtype='stepfilled', orientation='horizontal', color='deeppink')
    """ Decorations """
    ax_main.set(title='Scatterplot with Histograms \n displ vs hwy', xlabel='displ', ylabel='hwy')
    ax_main.title.set_fontsize(20)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(14)
    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    plt.show()


def correllogram():
    df = pd.read_csv("data/mtcars.csv")
    plt.figure(figsize=(12, 10), dpi=80)
    sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
    plt.title('Correlogram of mtcars', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def density_plot():
    df = pd.read_csv("data/mpg_ggplot2.csv")
    plt.figure(figsize=(16, 10), dpi=80)
    sns.kdeplot(df.loc[df['cyl'] == 4, "cty"], shade=True, color="g", label="Cyl=4", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 5, "cty"], shade=True, color="deeppink", label="Cyl=5", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 6, "cty"], shade=True, color="dodgerblue", label="Cyl=6", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)
    plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=22)
    plt.legend()
    plt.show()


def joy_plot():
    mpg = pd.read_csv("data/mpg_ggplot2.csv")

    # Draw Plot
    plt.figure(figsize=(16, 10), dpi=80)
    fig, axes = joypy.joyplot(mpg, column=['hwy', 'cty'], by="class", ylim='own', figsize=(14, 10))

    # Decoration
    plt.title('Joy Plot of City and Highway Mileage by Class', fontsize=22, pad=-10)
    plt.show()


def density_curves_with_histogram():
    df = pd.read_csv("data/mpg_ggplot2.csv")
    plt.figure(figsize=(13, 10), dpi=80)
    """ 手绘样式 """
    with plt.xkcd():
        sns.distplot(df.loc[df['class'] == 'compact', "cty"], color="dodgerblue", label="Compact", hist_kws={'alpha': .7}, kde_kws={'linewidth': 3})
        sns.distplot(df.loc[df['class'] == 'suv', "cty"], color="orange", label="SUV", hist_kws={'alpha': .7}, kde_kws={'linewidth': 3})
        sns.distplot(df.loc[df['class'] == 'minivan', "cty"], color="g", label="minivan", hist_kws={'alpha': .7}, kde_kws={'linewidth': 3})
        plt.ylim(0, 0.35)
        plt.title('Density Plot of City Mileage by Vehicle Type', fontsize=22)
        plt.legend()
    plt.show()


if __name__ == '__main__':
    # line_plot()
    # multiyy_plot()
    # time_series_with_peaks()
    # scatter_plot()
    # marginal_histogram()
    # correllogram()
    # density_plot()
    # joy_plot()
    density_curves_with_histogram()
