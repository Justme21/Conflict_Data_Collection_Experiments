#!/usr/bin/env python
import sys
import getopt
#import world
import car
import utils
import theano as th
import theano.tensor as tt
import theano.tensor.nlinalg as tn
import pickle
import feature
import numpy as np

th.config.optimizer_verbose = True

import dynamics
import math
import os
import pdb
import re

def run_irl(car, reward, theta, data):
    #Compute Reward value when applied to trajectories in data
    #def gen():
    #    for point in data:
    #        for c, x0, u in zip(world.cars, point['x0'], point['u']):
    #            c.traj.x0.set_value(x0)
    #            for cu, uu in zip(c.traj.u, u):
    #                cu.set_value(uu)
    #        yield
    
    def gen():
        for point in data:
            car.traj.x0.set_value(point['x0'])
            for cu, uu in zip(car.traj.u, point['u']):
                cu.set_value(uu)
            yield

    r = car.traj.reward(reward)
    g = utils.grad(r, car.traj.u)
    H = utils.hessian(r, car.traj.u)
    I = tt.eye(utils.shape(H)[0])
    reg = utils.vector(1)
    reg.set_value([1e-1])
    H = H-reg[0]*I #ensuring Hessian is invertible I guess
    L = tt.dot(g, tt.dot(tn.MatrixInverse()(H), g))+tt.log(tn.Det()(-H))
    #Unnecessary
    #for _ in gen():
    #    pass
    optimizer = utils.Maximizer(L, [theta], gen=gen, method='gd', eps=0.0001, debug=True, iters=1000, inf_ignore=10000000)
    #print("Optimiser instantiated")
    #pdb.set_trace()
    optimizer.maximize()
    #print("Optimisation Complete")
    #pdb.set_trace()
    print(theta.get_value())


def extractData(filename,sample_length):
    results = open("{}".format(filename),"r")

    #Extract Parameters
    #lane_width
    line = results.readline()
    lane_width = float(line.split(":")[-1])

    #veh_length
    line = results.readline()
    veh_length = float(line.split(":")[-1])

    #veh_width
    line = results.readline()
    veh_width = float(line.split(":")[-1])

    #dt
    line = results.readline()
    dt = float(line.split(":")[-1])

    #speed_limit
    line = results.readline()
    speed_limit = float(line.split(":")[-1])

    #blank line
    results.readline()

    #X0
    #line = results.readline()
    #line = line.split(":")[-1]
    #x0 = np.array([float(x) for x in line.strip(" []\n").split(",")])

    #States
    line = results.readline()
    line = line.split(":")[-1]
    line = re.findall("[\[\(]+([\d\. ,-]+)[\)\]]+",line)
    states = np.array([tuple([float(x) for x in y.split(",")]) for y in line]) #num_observations x state dim

    #Actions
    line = results.readline()
    line = line.split(":")[-1]
    line = re.findall("[\[\(]+([\d\. ,-]+)[\)\]]+",line)
    actions = np.array([tuple([float(x) for x in y.split(",")]) for y in line]) #num_observations x action_dim
    #actions = np.array([tuple([float(x) for x in y.split(",")])[0] for y in line]).reshape(1,-1)

    #There's a better way to add a dimension, I'm just not savy to it.
    action_out = np.expand_dims(np.array(actions[:sample_length]),axis=0)
    X0_out = np.expand_dims(np.array(states[0]),axis=0)

    for i in range(sample_length,len(actions)-sample_length,int(sample_length/2)):
        action_out = np.concatenate((action_out,actions[np.newaxis,i:i+sample_length]),axis=0)
        X0_out = np.concatenate((X0_out,states[np.newaxis,i]),axis=0)

    #actions = actions[:100,:] #Need to make sure all trials contain the same number of actions

    return {"actions":action_out,"lane_width":lane_width,"veh_length":veh_length,"veh_width":veh_width,"dt":dt,"speed_limit":speed_limit,"x0":X0_out}


def horizGaussian(centre,lane_width,gauss_width=.5):
    @feature.feature #pass as a feature (I think)
    def f(t,x,u):
        return tt.exp(-.5*((x[1]-centre)**2)/((gauss_width**2)*(lane_width**2)/4))
    return f



if __name__ == '__main__':
    ###########Parameters############################
    dt = 0.1
    T = 20 #Length of trajectories (number of entries in a trajectory)

    ########Load Training Data for IRL###############
    file_list = [x for x in os.listdir() if re.match("irl_data_generator_results-.*.txt",x)]

    train = []
    data = None
    for fname in file_list:
        data = extractData(fname,T)
        for i in range(data["x0"].shape[0]):
            point = {'x0':data["x0"][i],"u":data["actions"][i]}
            train.append(point)

    #x0 = data["x0"]
    #u = data["actions"]
    #pdb.set_trace()

    lane_width = data["lane_width"]
    lane_y = data["x0"][0][1]+lane_width
    veh_width = data["veh_width"]
    veh_length = data["veh_length"]
    dt = data["dt"]
    speed_limit = data["speed_limit"]
    
    dyn = dynamics.CarDynamics(dt=dt)
    #car arguments: [dynamics model,initial state,color<default='yellow'>,T<default=5>]
    car = car.Car(dyn,[0,0,0,math.pi/2],T=T)
    
    #w = utils.vector(5) #reward weights
    #w.set_value(np.array([1., -50., 10., 10., -60.]))
    w = utils.vector(2) #reward weights
    w.set_value(np.array([.001,.001]))

    ######Define the (Linear) Reward function#######
    #All of the features of the reward are functions f(t,x,u); t:-time, x:-state value at time t, u:-action value at time t
    #x is a utility (theano) vector and u is a list of utility (theano) vectors
    #Limit the magnitude of controls
    r = 0.0001*feature.control()
    #Distance from middle of lane
    #for lane in the_world.lanes:
        #r = r + w[0]*lane.gaussian()
    for centre in [lane_width/2,3*lane_width/2]:
        r = r + w[0]*horizGaussian(centre,lane_width)
    #Distance from edge of lane
    #for fence in the_world.fences:
    #    r = r + w[1]*lane.gaussian()
    #for centre in [0,lane_width,lane_width,2*lane_width]:
    #    r = r + w[1]*horizGaussian(centre,lane_width)
    #Distance from centre of road
    #for road in the_world.roads:
    #    r = r + w[2]*road.gaussian(10.)
    #for centre in [lane_width]:
    #    r = r + w[2]*horizGaussian(centre,2*lane_width)
    #Distance from speed limit
    #r = r + w[3]*feature.speed(speed_limit)
    r = r + w[1]*feature.speed(speed_limit)
    #Distance from other cars
    #for car in the_world.cars:
    #    if car!=the_car:
    #        r = r + theta[4]*car.traj.gaussian()

    #############Perform IRL#######################
    run_irl(car, r, w, train)
