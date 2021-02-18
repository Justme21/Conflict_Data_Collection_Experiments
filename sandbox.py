import sys
sys.path.insert(0,'./libraries/driving_simulator')

import datetime
import linear_controller_classes as lcc
import math
import pdb
import pyautogui
import pygame
import simulator
import vehicle_classes
import random
import road_classes


def initialiseSimulator(cars,speed_limit,init_speeds=None,vehicle_spacing=3,lane_width=None,dt=.1,debug=False):
    """Takes in a list of cars and a boolean indicating whether to produce graphics.
       Outputs the standard straight road simulator environment with the input cars initialised
       on the map with the first car (presumed ego) ahead of the second"""
    #Construct the simulation environment
    if init_speeds is None:
        car_speeds = [speed_limit for _ in range(len(cars))] #both cars start going at the speed limit
    else:
        car_speeds = init_speeds

    num_junctions = 5
    num_roads = 4
    road_angles = [0,0,0,0]
    road_lengths = [5,45,5,45] #Shorter track for generating results
    junc_pairs = [(0,1),(1,2),(2,3),(3,4)]

    starts = [[(0,1),1],[(2,3),0]] #Follower car is initialised on the first road, leading car on the 3rd (assuring space between them)
    dests = [[(3,4),1],[(3,4),1]] #Simulation ends when either car passes the end of the 

    run_graphics = True
    draw_traj = False #trajectories are uninteresting by deafault

    runtime = 120.0 #max runtime; simulation will terminate if run exceeds this length of time

    #Initialise the simulator object, load vehicles into the simulation, then initialise the action simulation
    sim = simulator.Simulator(run_graphics,draw_traj,runtime,debug,dt=dt)
    sim.loadCars(cars)

    sim.initialiseSimulator(num_junctions,num_roads,road_angles,road_lengths,junc_pairs,\
                                                    car_speeds,starts,dests,lane_width=lane_width)
    return sim

###################################################################################################################

def onLaneTrigger(ego_car, trigger_car):
    #Triggered if trigger car is on the same lane as ego car
    def f():
        lanes = [x for x in ego_car.on if isinstance(x,road_classes.Lane)]
        return (True in [trigger_car in x.on for x in lanes]) and trigger_car.state["position"][0]>ego_car.state["position"][0]-ego_car.length/2

    return f


def computeDistance(pt1,pt2):
    """Compute the L2 distance between two points"""
    return math.sqrt((pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2)


def radiusTrigger(ego_car,trigger_car,radius):
    #Triggered if trigger car is too close to ego car
    def f():
        return computeDistance(ego_car.state["position"],trigger_car.state["position"])<radius

    return f

def headingTrigger(ego_car,radius):
    def f():
        return abs(ego_car.state["heading"])<radius or abs(ego_car.state["heading"])>360-radius

    return f


def andTrigger(triggers):
    def f():
        res = True
        for trigger in triggers:
            res = res and trigger()
        return res

    return f


def relativeYRadiusTrigger(ego_car,trigger_car,radius,rel='>'):
    def f():
        if rel == '=': return abs(ego_car.state["position"][1] - trigger_car.state["position"][1]) == radius
        elif rel == '<': return abs(ego_car.state["position"][1] - trigger_car.state["position"][1]) < radius
        else: return abs(ego_car.state["position"][1] - trigger_car.state["position"][1]) > radius

    return f


def changeController(car,tag):
    def f():
        old_controller = car.controller
        car.setController(tag=tag)
        #Transfer log from old controller to new so they seem like a single contiuous controller
        new_controller = car.controller
        new_controller.log = list(old_controller.log)

    return f

###################################################################################################################
#Direction Writing Stuff
def writeTextToScreen(screen,text_lines,start_position,font_size,space_size):
    white = (255,255,255)
    font = pygame.font.Font(None,font_size)
    text = [font.render(l,1,white) for l in text_lines]
    for i,line in enumerate(text):
        screen.blit(line,(start_position[0],start_position[1]+font_size*i+space_size*i))


def writeText(screen,text_list,start_position,font_size,space_size):
    def f():
        writeTextToScreen(screen,text_list,start_position,font_size,space_size)

    return f


#Trigger functions to fire writing tasks
def trueFunc():
    return True

#####################################################################################################################

def runExperiment():
    #debug mode
    debug = False

    #########################################################################
    #Setting Up Simulation
    #Vehicle Dimensions
    lane_width = 5
    veh_length = 4.6
    veh_width = 2

    #Dynamics Parameters
    accel_jerk = 3
    yaw_rate_jerk = 10
    dt = .1
    speed_limit = 5.5
    participant_accel_range = [-6,3]
    accel_range = [-3,3]
    yaw_rate_range = [-10,10] # degree per second^2
    
    #Define the car(s)
    #Human Controlled Car (this is the car being modelled)
    lane_changer = vehicle_classes.Car(None,True,1,timestep=dt,car_params={"length":veh_length,"width":veh_width},debug=debug)
    manual_controller = lcc.DrivingController(controller="manual",ego=lane_changer,speed_limit=speed_limit,yaw_rate_range=yaw_rate_range,accel_range=participant_accel_range,accel_jerk=accel_jerk,yaw_rate_jerk=yaw_rate_jerk)
    lane_changer.addControllers({"default":manual_controller})
    lane_changer.setController(tag="default")

    #Other Car
    lane_keeper = vehicle_classes.Car(None,False,2,timestep=dt,car_params={"length":veh_length,"width":veh_width},debug=debug)
    
    #################################################################################
    #Initialise Simulator here becayse need state definition
    sim = initialiseSimulator([lane_changer,lane_keeper],speed_limit,init_speeds=[5,0],lane_width=lane_width,dt=dt,debug=debug)

    lane_keeper.heading = (lane_keeper.heading+180)%360
    lane_keeper.initialisation_params["heading"] = lane_keeper.heading
    lane_keeper.sense()
    ##################################################################################
    #Write Instructions

    w,h = pyautogui.size() #height and width of screen

    pygame.init()
    g_sim = sim.g_sim

    screen = sim.g_sim.screen #This is messy, but the best way to get this I think
    font_size = 40
    space_size = 15

    instructions = ["-Press and hold UP arrow to accelerate","-Press and hold DOWN arrow to decelerate","-Press and hold the LEFT arrow to turn anti-clockwise","-Press and hold RIGHT arrow to turn clockwise","-Press SPACE to pause simulation"]
    write_instructions = writeText(screen,instructions,(0,int(h/5)),font_size,space_size)
   
    ###########################################################################################
    #Setting up Controllers for Lane Keeping Vehicle
    #Constant velocity controller
    constant_controller = lcc.DrivingController(controller="constant",ego=lane_keeper,other=lane_changer,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk,yaw_rate_range=yaw_rate_range,yaw_rate_jerk=yaw_rate_jerk)

    lane_keeper.addControllers({"default":constant_controller})
    lane_keeper.setController(tag="default",controller=None)
    
    #########################################################################################
    #Run Experiments
    while True:
         ######################################################################
        #Set Graphic Simulator triggers
        triggers = {trueFunc:write_instructions}
        g_sim.triggers = {}
        g_sim.addTriggers(triggers)

        ######################################################################
        #Run simulation
        sim.reinitialise()
        sim.runComplete() #will start from paused

    #Shut down the graphic screen
    sim.wrapUp()


if __name__ == "__main__":
    runExperiment()
