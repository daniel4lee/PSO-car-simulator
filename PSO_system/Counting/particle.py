import random
import numpy as np
from copy import deepcopy
class Particle(object):
    def __init__(self, j_dim, i_dim, ranges, upbound_of_SD, v_max):
        self.vmax = v_max
        self.ranges = ranges
        self.upbound_of_SD = upbound_of_SD
        # individual = > 'thita', 'w', 'm', 'sd', 'adapt_value'
        self.theta = np.array([random.uniform(-1, 1)])
        self.weight = np.zeros(j_dim)
        self.means = np.zeros(j_dim*i_dim)
        self.sd = np.zeros(j_dim)
        self.fitness = None
        # w initialization
        for i in range(j_dim):
            self.weight[i] = random.uniform(-1, 1)
        # m initialization
        for i in range(j_dim*i_dim):
            self.means[i] = random.uniform(ranges[1], ranges[0])
        # sd initialization
        for i in range(j_dim):
            self.sd[i] = random.uniform(0.001, upbound_of_SD)
        
        # p-vector record the best lacation found by particle so far
        self.p_theta = deepcopy(self.theta)
        self.p_weight = deepcopy(self.weight)
        self.p_means = deepcopy(self.means)
        self.p_sd = deepcopy(self.sd)
        self.p_fitness = None

        # v-vector record the best lacation found by particle so far
        self.v_theta = np.array([random.uniform(-1, 1)])
        self.v_weight = np.zeros(j_dim)
        self.v_means = np.zeros(j_dim*i_dim)
        self.v_sd = np.zeros(j_dim)
        # w initialization
        for i in range(j_dim):
            self.v_weight[i] = random.uniform(-1, 1)
        # m initialization
        for i in range(j_dim*i_dim):
            self.v_means[i] = random.uniform(ranges[1], ranges[0])
        # sd initialization
        for i in range(j_dim):
            self.v_sd[i] = random.uniform(1/2 * upbound_of_SD, upbound_of_SD)

    def printmyself(self):
        print('theta', self.theta, 'p', self.p_theta)
        print('weight', self.weight, 'p', self.p_weight)
        print('means', self.means, 'p', self.p_means)
        print('sd', self.sd, 'p', self.p_sd)
        print('fitness', self.fitness, 'p', self.p_fitness)

    def update_p(self):
        self.p_theta = deepcopy(self.theta)
        self.p_weight = deepcopy(self.weight)
        self.p_means = deepcopy(self.means)
        self.p_sd = deepcopy(self.sd)
        self.p_fitness = deepcopy(self.fitness)

    def update_location(self):
        self.theta = deepcopy(self.theta + self.v_theta)
        self.weight = deepcopy(self.weight + self.v_weight)
        self.means = deepcopy(self.means + self.v_means)
        self.sd = deepcopy(self.sd + self.v_sd)
        self.fitness = None

    def limit_v(self):
        np.clip(self.v_theta, -1*self.vmax , 1*self.vmax, out=self.v_theta)
        np.clip(self.v_weight, -1*self.vmax, 1*self.vmax, out=self.v_weight)
        np.clip(self.v_means, -1*self.vmax, self.vmax, out=self.v_means)
        np.clip(self.v_sd, -1*self.vmax, self.vmax, out=self.v_sd)
    
    def limit_location_upbound(self):
        np.clip(self.theta, -1, 1, out=self.theta)
        np.clip(self.weight, -1, 1, out=self.weight)
        np.clip(self.means, self.ranges[1], self.ranges[0], out=self.means)
        np.clip(self.sd, 0.00000000000000000000001, self.upbound_of_SD, out=self.sd)
        

    

     
