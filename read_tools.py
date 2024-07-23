import scienceplots
from matplotlib.patches import Arrow
import pandas as pd
import numpy as np
import re
import sys
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import math
from matplotlib.widgets import Button
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.colors as mcolors
from matplotlib import cbook
import cv2 as cv2
from sklearn.cluster import KMeans
from matplotlib.ticker import AutoMinorLocator,MultipleLocator,FuncFormatter,LinearLocator,NullLocator,FixedLocator,IndexLocator,AutoLocator
from PIL import Image
# from mayavi import mlab
# from vedo import Plotter, Points
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Rectangle
# print(dir(mpl_toolkits.mplot3d))
save_path = r"C:\Users\tfsn20\Desktop"
color = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(color)

class ExcelReader:
    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.df = self.read_excel()
        self.effective_columns_index = self.calculate_effective_columns_index()

    def read_excel(self):
        df = pd.read_excel(
            self.file_path, sheet_name=self.sheet_name, header=None)
        return df

    @property
    def shape(self):
        return self.df.shape

    @property
    def effective_columns(self):

        return len([col for col in self.df.columns if not self.df[col].dropna().empty])

    @property
    def header(self):
        return list(self.df.iloc[0, :].dropna())

    def get_column_data(self, row_index, col_index):

        return list(self.df.iloc[row_index:, col_index].dropna())

    def calculate_effective_columns_index(self):

        return [col for col in range(self.df.shape[1]) if not self.df[col].dropna().empty]


class TxtFileParser:
    def __init__(self, file_path, data_separators=[',', '\t',' '], attribute_separators=[':', '=']):
        self.__file_path = file_path
        self.__data_separators = data_separators
        self.__attribute_separators = attribute_separators
        self.data_separators = None
        self.data_start_index = None
        self.data_columns = None
        self.__content = self.__get_content()
        self.__lines = self.get_lines()
        self.attributes = self.get_attributes()
        self._init_property()
        self.data = self.get_data()
        self.name=os.path.splitext(os.path.basename(file_path))[0]

    def test_(self):
        return type('anonymous_class', (), {'attribute': 'value'})()

    def __get_content(self):
        with open(self.__file_path, 'r') as file:
            content = file.read()
            file.seek(0)
        return content.strip()


    def get_lines(self):
        with open(self.__file_path, 'r') as file:
            lines = file.readlines()
            file.seek(0)
        while lines and lines[-1] == '\n':
            lines.pop()
        lines = [line.strip() for line in lines]
        return lines

    def get_attributes(self):
        matches = re.findall(r'.+', self.__content)
        attributes = {}
        for match in matches:
            _ = []
            for e in self.__attribute_separators:

                if re.search(r'\b\d{2}:\d{2}:\d{2}\b', match):
                    pass
                else:
                    _ += re.findall(rf'(.+?){e}\s*(\S+.*)', match)
            if _ != []:
                attributes[_[0][0].strip()] = convert_to_number(
                    _[0][1].strip())

        return attributes

    def _init_property(self):
        for o in self.__data_separators:
            for i, e in enumerate(self.__lines):
                _ = e.split(o)
                if all(is_numeric(item.strip())for item in _):

                    self.data_separators = o
                    self.data_start_index = i
                    self.data_columns = len(_)
                    # print(o, i, len(_))
                    break
            else:
                continue
            break

    def get_data(self):
        # if self.data_separators == None:
        #     return None

        data = [[] for d in range(self.data_columns)]

        for line in self.__lines[self.data_start_index:]:
            # print(line)
            values = [float(val) if '.' in val else int(val)
                      for val in line.split(self.data_separators)]
            for i, val in enumerate(values):
                data[i].append(val)

        return data


class DataObject:


    def __init__(self, high_E, low_E, scan_rate, data):
        self.high_E = high_E
        self.low_E = low_E
        # mV/s
        self.scan_rate = scan_rate
        self.data = data

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:

        width, height = img.size
        return width, height

def hex_to_rgb(hex_color):

    hex_color = hex_color.lstrip('#')
    

    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    

    red_normalized = red / 255.0
    green_normalized = green / 255.0
    blue_normalized = blue / 255.0
    
    return (red_normalized, green_normalized, blue_normalized)

def find_files(directory, file_extension='txt', connect='_'):

    txt_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(f'.{file_extension}'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)


                relative_path_with_underscore =os.path.splitext(relative_path.replace(
                    os.path.sep, connect))[0]

                txt_files.append((relative_path,
                                 file_path, relative_path_with_underscore,os.path.basename(os.path.dirname(file_path))))

    return txt_files


def find_closest_elements(y, a, b=None):

    if b == None:
        b = sum(y) / len(y)

    return [min((y[i:i+a]), key=lambda x: abs(x - b))
            for i in range(0, len(y), a)]


def fit_and_select(y, a, degree=2):


    def func(x, *params):
        return np.polyval(params, x)


    x_data = np.arange(len(y))

    try:
        params, covariance = curve_fit(func, x_data, y, p0=np.ones(degree + 1))


        fit_curve = func(x_data, *params)


        selected_values = []
        for i in range(0, len(y), a):
            index = min(range(i, min(i+a, len(y))),
                        key=lambda j: abs(fit_curve[j] - y[j]))
            selected_values.append(y[index])

        return selected_values

    except Exception as e:
        print(f"Error during curve fitting: {e}")
        return None




def rgb_to_hex(rgb_tuple):
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        round(rgb_tuple[0]), round(rgb_tuple[1]), round(rgb_tuple[2]))
    return hex_color



def get_lines_colors_by_pic(pic_path, n_clusters, wish_clusters, diff=10, mode=16):
    img = cv2.imread(pic_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_reshaped = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(img_reshaped)

    colors = kmeans.cluster_centers_
    # print(colors)
    colors_ = colors.copy()
    mid = [sorted(d)[len(d)//2] for d in colors]
    res = []
    for i in range(len(mid)):
        if max(colors[i]-min(colors[i])) > diff:
            continue
        else:
            res.append(i)
    colors = np.delete(colors, res, 0)
    if wish_clusters < len(colors):

        row_diff = np.ptp(colors, axis=1)

        min_diff_indices = np.argsort(row_diff)[:(len(colors)-wish_clusters)]
        colors = np.delete(colors, min_diff_indices, 0)

    return [rgb_to_hex(d) for d in kmeans.cluster_centers_] if mode == 16 else colors


def plot_multiple_lines(datasets,
                        x_label,
                        y_label,
                        x_unit,
                        y_unit,
                        line_colors,
                        line_labels,
                        alpha=[1],
                        isscatter=[False],
                        datasets_scatter=None,
                        line_widths=None,
                        axis_linewidth=4,
                        title_size=35,
                        axis_graduations_font_size=35,
                        legent_font_size=25,
                        major_tick_length=6,
                        minor_tick_length=3,
                        title_weight=1000,
                        axis_graduations_font_weight=1000,
                        ticks_direction='out',
                        x_surplus=None,
                        y_surplus=None,
                        mode=None,
                        closed=True,
                        fill=[],
                        fill_config={
                            'opacity': 0.6,
                            'label': 'Capacitive',
                            'font_size': 20
                        },
                        save_path=r"C:\Users\tfsn20\Desktop",
                        marker={
                            'shape': None,
                            'size': 20
                        },
                        xticks=False,
                        yticks=False,
                        extra_label=[{
                            'label_name': '',
                            'font_size': 20,
                            'position': 'upper right'
                        }],
                        draw_type='XY'):

    # print(datasets)
    _ = alpha[0]
    if len(alpha) == 1:
        if datasets_scatter != None:
            alpha = [_ for d in range(len(datasets)+len(datasets_scatter))]
        else:
            alpha = [_ for d in range(len(datasets))]

    if mode == 'big pic':
        axis_linewidth = 4
        title_size = 35
        axis_graduations_font_size = 35
        legent_font_size = 25
        major_tick_length = 6
        minor_tick_length = 3
    elif mode == 'small pic':
        axis_linewidth = 5
        title_size = 60
        axis_graduations_font_size = 50
        legent_font_size = 30
        major_tick_length = 10
        minor_tick_length = 5
    if line_widths == None:
        line_widths = [axis_linewidth for d in datasets]
    elif (count := len(line_widths)) != (l := len(datasets)):
        line_widths = line_widths + [axis_linewidth for i in range(l - count)]

    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['font.weight'] = axis_graduations_font_weight
    plt.rcParams["axes.labelweight"] = title_weight
    # plt.figure(figsize=(8, 6),dpi=100)  
    plt.figure(figsize=(16, 12), dpi=100)  

    for i, data in enumerate(datasets):
        x = data[0].copy()
        y = data[1].copy()
        if closed:
            x.append(x[0])
            y.append(y[0])
        color = line_colors[i]
        width = line_widths[i]
        label = line_labels[i]
        plt.plot(x,
                 y,
                 color,
                 marker=marker['shape'],
                 markersize=marker['size'],
                 alpha=alpha[i],
                 linewidth=width,
                 label=label if i not in fill else None)
        if i in fill:
            plt.fill(x,
                     y,
                     color=line_colors[i],
                     alpha=fill_config['opacity'],
                     label=label)
            plt.text((max(x) + min(x)) / 2, (max(y) + min(y)) / 2,
                     fill_config['label'],
                     fontsize=fill_config['font_size'],
                     ha='center')
    if datasets_scatter != None:
        for i, data in enumerate(datasets_scatter):
            x = data[0].copy()
            y = data[1].copy()
            plt.scatter(
                x, y, color=line_colors[i], alpha=alpha[i], label=None, s=axis_linewidth*10)
    if extra_label[0]['label_name'] == '':
        pass
    else:
        for e in extra_label:
            position = e['position']
            if e['position'] == 'upper right':
                position = [max(datasets[0][0]), max(datasets[0][1])]
            elif e['position'] == 'upper left':
                position = [min(datasets[0][0]), max(datasets[0][1])]
            else:
                position = [e['position'][0], e['position'][1]]
            plt.text(position[0], position[1],
                     e['label_name'],
                     fontsize=e['font_size'],
                     ha='center', va='center')
    plt.xlabel(f'{x_label} ({x_unit})' if x_unit != None else f'{x_label}',
               fontsize=title_size,
               rotation='horizontal',
               labelpad=0)
    plt.ylabel(f'{y_label} ({y_unit})' if y_unit != None else f'{y_label}',
               fontsize=title_size,
               rotation='vertical',
               labelpad=0)
    plt.tick_params(direction=ticks_direction,
                    width=axis_linewidth,
                    labelsize=axis_graduations_font_size,
                    length=major_tick_length)


    if x_surplus is not None:
        plt.xlim(cv_obj_list[0].low_E - x_surplus,
                 cv_obj_list[0].high_E + x_surplus)

    axes = plt.subplot()

    axes.tick_params(axis='x',
                     which="minor",
                     direction="out",
                     width=axis_linewidth,
                     length=minor_tick_length)
    axes.tick_params(axis='y',
                     which="minor",
                     direction="out",
                     width=axis_linewidth,
                     length=minor_tick_length)
    axes.xaxis.set_major_locator(ticker.MaxNLocator())
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    if isinstance(xticks, np.ndarray):
        plt.xticks(xticks)
    axes.yaxis.set_major_locator(ticker.MaxNLocator(9))
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    if isinstance(yticks, np.ndarray):
        plt.yticks(yticks)
    axes.spines['bottom'].set_linewidth(axis_linewidth)  
    axes.spines['left'].set_linewidth(axis_linewidth)  
    axes.spines['top'].set_linewidth(axis_linewidth)  
    axes.spines['right'].set_linewidth(axis_linewidth) 
    axes.set_position(
        [0.2, 0.2,
         axes.get_position().width,
         axes.get_position().height])

    plt.legend(frameon=False, fontsize=legent_font_size)
    print(plt.gcf())

    # plt.subplots_adjust(top=0.5)
    s = ''.join('_' if char in r'\/|:?"<>' else char
                for char in '-'.join(map(str, line_labels)))

    plt.savefig(rf"{save_path}\{s}.jpg", bbox_inches='tight', dpi=600)
    button_ax = plt.axes([0.7, 0.01, 0.2, 0.075])  

    button = Button(button_ax, 'click me')


    def get_axis_piexl(event):
        chartBox = axes.get_position()
        x, y, w, h = chartBox.x0, chartBox.y0, chartBox.width, chartBox.height
        fig = plt.gcf()  
        fig_size_inches = fig.get_size_inches()  
        dpi = fig.get_dpi()  


        fig_width_pixels = fig_size_inches[0] * dpi
        fig_height_pixels = fig_size_inches[1] * dpi

        print(fig_width_pixels * w)
        print(fig_height_pixels * h)

    button.on_clicked(get_axis_piexl)

    plt.connect('key_press_event', lambda event: plt.close()
                if event.key == ' ' else None) 
    plt.show()


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_to_number(input_str):
    try:

        result = float(input_str)

        if result.is_integer():
            result = int(result)
        return result
    except ValueError:
        return input_str

def set_bold_ticks(ax):

    x_ticks = ax.get_xticklabels()
    y_ticks = ax.get_yticklabels()


    for tick in x_ticks + y_ticks:
        tick.set_weight('bold')

def process_abbbc(x:list,need_process:bool=True,start=0.2,end=None)->list:

    seen = set()
    duplicate_indices = []

    for i, num in enumerate(x):
        if num in seen:
            duplicate_indices.append(i)
        else:
            seen.add(num)
    result = []
    nums=duplicate_indices

    if len(nums) == 1:
        result = nums

    for i in range(len(nums) - 1):
        if nums[i] + 1 == nums[i + 1]:
            result = [nums[i], nums[i + 1]]
            break
        else:
            result.append(nums[0])
            break
    # pop一个index
    index_need_processed=[result[0]-1]+result if result !=[] else result
    
    if need_process:
        if index_need_processed == []:
            return x
        if index_need_processed[0]==0:
            x[0:index_need_processed[-1]+1]=list(np.linspace(start, x[index_need_processed[-1]], len(index_need_processed)))
            pass
        else:
            if len(x)==index_need_processed[-1]+1:
                pass
            else:
                x[index_need_processed[0]-1:index_need_processed[-1]+2]=list(np.linspace(x[index_need_processed[0]-1], x[index_need_processed[-1]+1], len(index_need_processed)+2))
        return x
    else:
        return index_need_processed
