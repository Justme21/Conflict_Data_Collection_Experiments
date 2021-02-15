import sys
sys.path.insert(0,'../../libraries/driving_simulator')

import datetime
import linear_controller_classes as lcc
import math
import pdb
import simulator
import vehicle_classes
import road_classes

DATA_ADDRESS = "../../results/exp1/a"

def initialiseSimulator(cars,speed_limit,init_speeds=None,vehicle_spacing=3,lane_width=None,dt=.1,debug=False):
    """Takes in a list of cars and a boolean indicating whether to produce graphics.
       Outputs the standard straight road simulator environment with the input cars initialised
       on the map with the first car (presumed ego) ahead of the second"""
    #Construct the simulation environment
    if init_speeds is None:
        car_speeds = [speed_limit for _ in range(len(cars))] #both cars start going at the speed limit
    else:
        car_speeds = init_speeds

    num_junctions = 3
    num_roads = 2
    road_angles = [0,0]
    road_lengths = [5,120] #Shorter track for generating results
    junc_pairs = [(0,1),(1,2)]

    starts = [[(0,1),1],[(0,1),0]] #Follower car is initialised on the first road, leading car on the 3rd (assuring space between them)
    dests = [[(1,2),1],[(1,2),1]] #Simulation ends when either car passes the end of the 

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
        return (True in [trigger_car in x.on for x in lanes]) # and trigger_car.state["position"][0]>ego_car.state["position"][0]+ego_car.length/2

    return f


def computeDistance(pt1,pt2):
    """Compute the L2 distance between two points"""
    return math.sqrt((pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2)


def radiusTrigger(ego_car,trigger_car,radius):
    #Triggered if trigger car is too close to ego car
    def f():
        return computeDistance(ego_car.state["position"],trigger_car.state["position"])<radius and trigger_car.state["position"][0]>ego_car.state["position"][0]

    return f


def changeController(car,tag):
    def f():
        car.setController(tag=tag)

    return f

#####################################################################################################################

if __name__ == "__main__":
    #debug mode
    debug = False

    #Vehicle Dimensions
    lane_width = 5
    veh_length = 4.6
    veh_width = 2

    #Dynamics Parameters
    accel_jerk = 3
    yaw_rate_jerk = 10
    dt = .1
    speed_limit = 5.5
    accel_range = [-3,3]
    yaw_rate_range = [-10,10] # degree per second^2
    
    #Define the car(s)
    #Human Controlled Car (this is the car being modelled)
    lane_changer = vehicle_classes.Car(None,True,1,timestep=dt,car_params={"length":veh_length,"width":veh_width},debug=debug)
    manual_controller = lcc.DrivingController(controller="manual",ego=lane_changer,speed_limit=speed_limit,yaw_rate_range=yaw_rate_range,accel_range=accel_range,accel_jerk=accel_jerk,yaw_rate_jerk=yaw_rate_jerk)
    lane_changer.controller = manual_controller

    #Other Car
    lane_keeper = vehicle_classes.Car(None,False,2,timestep=dt,car_params={"length":veh_length,"width":veh_width},debug=debug)
    
    #Needs constant velocity controller AND IDM controller
    #IDM controller
    #idm_params = {"headway":1.6,"s0":2,"b":3} #passive
    idm_params = {"headway":0.0,"s0":0,"b":50} #aggressive
    idm_controller = lcc.DrivingController(controller="idm",ego=lane_keeper,other=lane_changer,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk,yaw_rate_range=yaw_rate_range,yaw_rate_jerk=yaw_rate_jerk,**idm_params)
    
    #Constant velocity controller
    constant_controller = lcc.DrivingController(controller="constant",ego=lane_keeper,other=lane_changer,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk,yaw_rate_range=yaw_rate_range,yaw_rate_jerk=yaw_rate_jerk)

    lane_keeper.addControllers({"default":constant_controller,"idm":idm_controller})
    lane_keeper.setController(tag="default",controller=None)

    trigger = onLaneTrigger(lane_keeper,lane_changer)
    consequent = changeController(lane_keeper,"idm")
    triggers = {trigger:consequent}

    lane_keeper.addTriggers(triggers)

    sim = initialiseSimulator([lane_changer,lane_keeper],speed_limit,init_speeds=[5,5],lane_width=lane_width,dt=dt,debug=debug)

    lane_keeper.heading = (lane_keeper.heading+180)%360
    lane_keeper.initialisation_params["heading"] = lane_keeper.heading
    lane_keeper.sense()


    sim.runComplete()

    #Extract log of behaviours from controller
    num_cars = 2
    constant_log_list = constant_controller.getLog()
    idm_log_list = idm_controller.getLog()
    lane_keeper_log_list = constant_log_list + idm_log_list

    lane_changer_log_list = manual_controller.getLog()

    #Behaviour being modelled/learnt put in first
    lane_keeper_state_list = [(x[0]["position"][0],x[0]["position"][1],x[0]["velocity"],math.radians(x[0]["heading"])) for x in lane_keeper_log_list]
    lane_keeper_act_list = [(x[1][0],math.radians(x[1][1])) for x in lane_keeper_log_list]

    #Behaviour of all other cars goes after
    lane_changer_state_list = [(x[0]["position"][0],x[0]["position"][1],x[0]["velocity"],math.radians(x[0]["heading"])) for x in lane_changer_log_list]
    lane_changer_act_list = [(x[1][0],math.radians(x[1][1])) for x in lane_changer_log_list]

    #import pdb
    #pdb.set_trace()


    results = open("{}/lane_change_results-{}.txt".format(DATA_ADDRESS,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),"w")
    results.write("num_cars: {}\n".format(num_cars))
    results.write("lane_width: {}\n".format(lane_width))
    results.write("veh_length: {}\n".format(veh_length))
    results.write("veh_width: {}\n".format(veh_width))
    results.write("dt: {}\n".format(dt))
    results.write("speed_limit: {}\n\n".format(speed_limit))

    #Behaviour being modelled/learnt put in first
    results.write("States: {}\n".format(lane_changer_state_list))
    results.write("Actions: {}\n".format(lane_changer_act_list))

    #Behaviour of all other cars goes after
    results.write("States: {}\n".format(lane_keeper_state_list))
    results.write("Actions: {}\n".format(lane_keeper_act_list))
    results.close()
