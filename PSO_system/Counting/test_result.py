import math
import random
from PSO_system.Counting.particle import Particle
import numpy as np
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, pyqtSlot
import shapely.geometry as sp
import descartes
from copy import deepcopy

class RunSignals(QObject):
    plot = pyqtSignal(object)
class TestRunning(QRunnable):
    """ test thread """
    def __init__(self, data, filename, parameters, ts):
        super(TestRunning, self).__init__()
        # for return result
        self.signals = RunSignals()
        # read data file
        self.data = data[filename]
        # initialize the particle
        if ts == None:
            self.neurl_num = parameters[1]
            self.pocket_particle = parameters[0]
            self.dim_i = parameters[2]
        elif parameters ==None:
            self.neurl_num = ts[1]
            self.pocket_particle = ts[0]
            self.dim_i = ts[2]
        
        max_px = max_py = -math.inf
        min_px = min_py = math.inf
        for i, j in zip(self.data.x[2:], self.data.y[2:]):
            max_px = max(i, max_px)
            min_px = min(i, min_px)
            max_py = max(j, max_py)
            min_py = min(j, min_py)

        self.upbound_of_map = ((max_px - min_px)**2 +
                               (max_py - min_py)**2)**(0.5)
        print(self.pocket_particle.theta)
        print(self.pocket_particle.means)
        print(self.pocket_particle.weight)
        print(self.pocket_particle.sd)
    @pyqtSlot()
    def run(self):
        def distance(points, car_loc):
            if isinstance(points, sp.MultiPoint):
                min_dis = ((points[0].x - car_loc[0])**2 +
                           (points[0].y - car_loc[1])**2)**(1/2)
                min_point = (points[0].x, points[0].y)
                for i in range(1, len(points)):
                    temp = ((points[i].x - car_loc[0])**2 +
                            (points[i].y - car_loc[1])**2)**(1/2)
                    if(temp < min_dis):
                        min_dis = temp
                        min_point = (points[i].x, points[i].y)
                l = [min_dis, min_point]
                return l
            elif isinstance(points, sp.Point):
                l = []
                l.append(
                    ((points.x - car_loc[0])**2 + (points.y - car_loc[1])**2)**(1/2))
                min_point = (points.x, points.y)
                l.append(min_point)
                return l

        def rbfn_funct(input_vector, parameters):
            f_x = parameters.theta[0]  # theta
            for j in range(self.neurl_num):
                gaussian = gaussian_funct(
                    j, input_vector, parameters.means, parameters.sd[j])
                f_x = f_x + parameters.weight[j] * gaussian
            return f_x

        def gaussian_funct(jth_neurl, v_x, v_m, o):
            temp = 0
            means = np.array(
                v_m[len(v_x) * jth_neurl:len(v_x) * jth_neurl + self.dim_i])
            temp = (v_x - means).dot(v_x - means)
            return (math.exp(-temp / (2 * o ** 2)))
        def se_rbfn_funct(input_vector, parameters):
            f_x = parameters.theta[0]  # theta
            for j in range(6):
                gaussian = gaussian_funct(
                    j, input_vector, parameters.means, parameters.sd[j])
                f_x = f_x + parameters.weight[j] * gaussian
            return f_x

        def se_gaussian_funct(jth_neurl, v_x, v_m, o):
            temp = 0
            means = np.array(
                v_m[len(v_x) * jth_neurl:len(v_x) * jth_neurl + self.dim_i])
            temp = (v_x - means).dot(v_x - means)
            return (math.exp(-temp / (2 * o ** 2)))
        """
        Run this function 
        """
        # trace data [0] = car center x, [1] = car center y, [2] = direction length,
        # [3] = right length, [4] = left length, [5] = thita, [6] = direct point on map line
        # [7] = right point on map line, [8] left point on map line, [9] the angle between dir car and horizontal
        trace_10d = []
        for i in range(10):
            trace_10d.append([])
        # creat end area by shapely
        end_area = []
        end_area.append((self.data.x[0], self.data.y[0]))
        end_area.append((self.data.x[1], self.data.y[0]))
        end_area.append((self.data.x[1], self.data.y[1]))
        end_area.append((self.data.x[0], self.data.y[1]))
        end_area = sp.Polygon(end_area)

        # creat map line by shapely
        map_line = []
        for i in range(2, len(self.data.x)):
            map_line.append([self.data.x[i], self.data.y[i]])
        map_line = sp.LineString(map_line)

        car_center = (self.data.start[0], self.data.start[1])
        car = sp.Point(*car_center).buffer(3)
        # main loop for computing through fuzzy architecture
        while(not car.intersection(map_line)):

            # let data list[1] be 1 longer as signal here !!
            if(end_area.contains(sp.Point(car_center))):
                trace_10d[1].append(0)
                break

            # creat car circle polygon by shapely and initial it, r, x, y
            if (len(trace_10d[0]) == 0):
                # count the distance
                r = self.upbound_of_map
                # initial x y fai
                x = self.data.start[0]
                y = self.data.start[1]
                fai = self.data.start[2]
                output = 0
            else:
                """update new point for computing """
                car_center = (car_center[0] + math.cos(math.radians(fai + output)) + math.sin(math.radians(fai))*math.sin(math.radians(output)),
                              car_center[1] + math.sin(math.radians(fai + output)) - math.sin(math.radians(output))*math.cos(math.radians(fai)))
                car = sp.Point(*car_center).buffer(3)
                x = car_center[0]
                y = car_center[1]
                fai = fai - \
                    math.degrees(math.asin(2*math.sin(math.radians(output))/6))
            ##
            trace_10d[0].append(x)
            trace_10d[1].append(y)
            trace_10d[9].append(fai)
            # dir, l, r line for counting intersection

            dir_line = [
                [x, y], [x + r * math.cos(math.radians(fai)), y + r * math.sin(math.radians(fai))]]
            l_line = [[x, y], [
                x + r * math.cos(math.radians(fai + 45)), y + r * math.sin(math.radians(fai + 45))]]
            r_line = [[x, y], [
                x + r * math.cos(math.radians(fai - 45)), y + r * math.sin(math.radians(fai - 45))]]

            # First, computing the dir, l, and r distance between car and wall
            temp = sp.LineString(dir_line).intersection(map_line)
            temp = distance(temp, car_center)
            dir_dist = temp[0]
            trace_10d[6].append(temp[1])
            temp = sp.LineString(r_line).intersection(map_line)
            temp = distance(temp, car_center)
            r_dist = temp[0]
            trace_10d[7].append(temp[1])
            temp = sp.LineString(l_line).intersection(map_line)
            temp = distance(temp, car_center)
            l_dist = temp[0]
            trace_10d[8].append(temp[1])

            ### record distace set in trace6d ###
            trace_10d[2].append(dir_dist)
            trace_10d[3].append(r_dist)
            trace_10d[4].append(l_dist)
            list4d = np.array([dir_dist, r_dist, l_dist])
            list6d = np.array([x, y, dir_dist, r_dist, l_dist])
            output = rbfn_funct(np.array(list4d), self.pocket_particle)
            output = max(-40, min(output * 40, 40))
            ### record wheel angle in trace6d ###
            trace_10d[5].append(output)
        self.signals.plot.emit(trace_10d)
