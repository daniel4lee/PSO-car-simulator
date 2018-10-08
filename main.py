"""
the main excution file
"""
import os
from os.path import join, isfile
from collections import namedtuple
import sys
from PyQt5.QtWidgets import QApplication
from PSO_system.GUI.gui_root import GuiRoot
import numpy as np
def main():
    """Read data as dictionary"""
    sys.argv += ['--style', 'fusion']
    app = QApplication(sys.argv)
    gui_root = GuiRoot(read_file(), read_training_file())
    sys.exit(app.exec_())
def read_file():
    """Read txt file in same location"""
    road_map = namedtuple('road_map', ['start', 'x', 'y'])
    datapath = join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), "map_data")
    folderfiles = os.listdir(datapath)
    dataset = {}
    paths = (join(datapath, f) for f in folderfiles if isfile(join(datapath, f)))
    for idx, content in enumerate(list(map(lambda path: open(path, 'r'), paths))):
        i = 0
        for line in content:
            if i == 0: 
                dataset[folderfiles[idx]] = road_map(list(map(float, line.split(','))), [], [])
            else:
                dataset[folderfiles[idx]].x.append(float(line.split(',')[0]))
                dataset[folderfiles[idx]].y.append(float(line.split(',')[1]))
            i += 1
    return dataset
def read_training_file():
    """Read txt file in same location"""
    train_data = namedtuple('train_data', ['wheel_angle', 'v_x'])
    datapath = join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), "training_data")
    folderfiles = os.listdir(datapath)
    dataset = {}
    paths = (join(datapath, f) for f in folderfiles if isfile(join(datapath, f)))
    for idx, content in enumerate(list(map(lambda path: open(path, 'r'), paths))):
        i = 0
        for line in content:
            if i == 0:
                dataset[folderfiles[idx]] = train_data([], [])
            dataset[folderfiles[idx]].wheel_angle.append(float(line.split(' ')[-1]))
            t = line.split(' ')
            del t[-1]
            dataset[folderfiles[idx]].v_x.append(np.array(list(map(float, t))))
            i += 1
    return dataset
main()
