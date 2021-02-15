import sys
sys.path.insert(0,'../../libraries/driving_simulator')

import datetime
import linear_controller_classes as lcc
import math
import pdb
import simulator
import vehicle_classes
import road_classes

sys.path.insert(0,'../../libraries')

from trajectory_type_definitions import LaneChangeController

DATA_ADDRESS = "../../results/exp1/b"

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
        #if True in [trigger_car in x.on for x in lanes]:
        #    import pdb
        #    pdb.set_trace()
        return (True in [trigger_car in x.on for x in lanes])

    return f


def computeDistance(pt1,pt2):
    """Compute the L2 distance between two points"""
    return math.sqrt((pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2)


def radiusTrigger(ego_car,trigger_car,radius):
    #Triggered if trigger car is too close to ego car
    def f():
        return computeDistance(ego_car.state["position"],trigger_car.state["position"])<radius

    return f

def relativeXRadiusTrigger(ego_car,trigger_car,radius,rel='>'):
    def f():
        if rel == '=': return ego_car.state["position"][0] - trigger_car.state["position"][0] == radius
        elif rel == '<': return ego_car.state["position"][0] - trigger_car.state["position"][0] < radius
        else: return ego_car.state["position"][0] - trigger_car.state["position"][0] > radius

    return f

def andTrigger(triggers):
    def f():
        res = True
        for trigger in triggers: 
            res = res and trigger()
            #print("This trigger is {}".format(res))
        #import pdb
        #pdb.set_trace()
        return res

    return f

def changeController(car,tag):
    def f():
        car.setController(tag=tag)

    return f

def headingFixTrigger(car):
    def f():
        return car.accel == 0 and car.yaw_rate==0

    return f

def fixHeading(car):
    #Manually adjust heading due to discretisaion error between trajectory and execution
    def f():
        car.heading = 0

    return f

#####################################################################################################################

if __name__ == "__main__":
    #debug mode
    debug = False

    #Vehicle Dimensions
    lane_width = 5
    veh_length = 4.6
    veh_width = 2
    axle_length = 2.72 #default value

    #Dynamics Parameters
    accel_jerk = 3
    yaw_rate_jerk = 10
    dt = .1
    init_speed = 5
    speed_limit = 5.5
    accel_range = [-3,3]
    yaw_rate_range = [-10,10] # degree per second^2
    
    #Define the car(s)
    #Human Controlled Car (this is the car being modelled)
    lane_keeper = vehicle_classes.Car(None,True,1,timestep=dt,car_params={"length":veh_length,"width":veh_width},debug=debug)
    manual_controller = lcc.DrivingController(controller="manual",ego=lane_keeper,speed_limit=speed_limit,yaw_rate_range=yaw_rate_range,accel_range=accel_range,accel_jerk=accel_jerk,yaw_rate_jerk=yaw_rate_jerk)
    lane_keeper.controller = manual_controller

    #Other Car
    lane_changer = vehicle_classes.Car(None,False,2,timestep=dt,car_params={"length":veh_length,"width":veh_width},debug=debug)
    
    #Initialise Simulator here because need state definition
    sim = initialiseSimulator([lane_changer,lane_keeper],speed_limit,init_speeds=[init_speed,init_speed],lane_width=lane_width,dt=dt,debug=debug)

    lane_keeper.heading = (lane_keeper.heading+180)%360
    lane_keeper.initialisation_params["heading"] = lane_keeper.heading
    lane_keeper.sense()


    #First Lane Change Controller: Try lane change
    right_lane_state = lane_keeper.state.copy() #lane keeper is initially centred in the right lane

    right_lane_state["velocity"] = speed_limit
    lcT = 15
    lc_controller = lcc.DrivingController(controller="standard",ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk)
    lc_controller.controller = LaneChangeController(ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,dest_state=right_lane_state,axle_length=axle_length,T=lcT)

    lane_changer.addControllers({"default":lc_controller})
    lane_changer.setController(tag="default",controller=None)

    #Second Lane Change Controller: Lane Change Successful (shorten up the remaining trajectory)
    slcT = 5
    slc_controller = lcc.DrivingController(controller="standard",ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk)
    slc_controller.controller = LaneChangeController(ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,dest_state=right_lane_state,axle_length=axle_length,T=slcT)
    lane_changer.addControllers({"short_lane_change":slc_controller})

    short_lane_change_trigger = onLaneTrigger(lane_changer,lane_keeper) and relativeXRadiusTrigger(lane_changer,lane_keeper,veh_length,'>')
    short_lane_change_consequent = changeController(lane_changer,"short_lane_change")
    
    #Third Lane Change Controller: Lane Change Rejected
    left_lane_state = lane_changer.state.copy()
    left_lane_state["velocity"] = 4 #2*init_speed - speed_limit

    rlcT = 5
    rlc_controller = lcc.DrivingController(controller="standard",ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk)
    rlc_controller.controller = LaneChangeController(ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,dest_state=left_lane_state,axle_length=axle_length,T=rlcT)
    lane_changer.addControllers({"reverse_lane_change":rlc_controller})

    risk_radius = veh_width
    reverse_lane_change_trigger = andTrigger([onLaneTrigger(lane_changer,lane_keeper), relativeXRadiusTrigger(lane_changer,lane_keeper,veh_length,'<')]) #passive
    #reverse_lane_change_trigger = andTrigger([radiusTrigger(lane_changer,lane_keeper,risk_radius), relativeXRadiusTrigger(lane_changer,lane_keeper,veh_length,'<')]) #aggressive
    reverse_lane_change_consequent = changeController(lane_changer,"reverse_lane_change")

    #Fourth Lane Change Controller: Lane Change Behind
    lcbT = 5
    slow_right_lane_state = lane_keeper.state.copy() #lane keeper is initially centred in the right lane
    behind_controller = lcc.DrivingController(controller="standard",ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk)
    behind_controller.controller = LaneChangeController(ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,dest_state=slow_right_lane_state,axle_length=axle_length,T=lcbT)
    lane_changer.addControllers({"lane_change_behind":behind_controller})

    lane_change_behind_trigger = relativeXRadiusTrigger(lane_keeper,lane_changer,veh_length,rel='>')
    lane_change_behind_consequent = changeController(lane_changer,"lane_change_behind")
   
    heading_fix_trigger = headingFixTrigger(lane_changer)
    heading_fix_consequent = fixHeading(lane_changer)
 
    triggers = {short_lane_change_trigger:short_lane_change_consequent, reverse_lane_change_trigger:reverse_lane_change_consequent,lane_change_behind_trigger:lane_change_behind_consequent,\
                heading_fix_trigger:heading_fix_consequent}

    lane_changer.addTriggers(triggers)

    sim.runComplete()

    import pdb
    pdb.set_trace()


    #Here I am
    # To Do:
    #    i. Include final crash and on_road state to text output
    #   ii. Include passive vs. aggressive label for each car
    #  iii. Figure out how to record log from lane changing car in this setting. 

    #Extract log of behaviours from controller
    num_cars = 2
    constant_log_list = constant_controller.getLog()
    idm_log_list = idm_controller.getLog()
    lane_changer_log_list = constant_log_list + idm_log_list

    lane_keeper_log_list = manual_controller.getLog()

    #Behaviour being modelled/learnt put in first
    lane_changer_state_list = [(x[0]["position"][0],x[0]["position"][1],x[0]["velocity"],math.radians(x[0]["heading"])) for x in lane_changer_log_list]
    lane_changer_act_list = [(x[1][0],math.radians(x[1][1])) for x in lane_changer_log_list]

    #Behaviour of all other cars goes after
    lane_keeper_state_list = [(x[0]["position"][0],x[0]["position"][1],x[0]["velocity"],math.radians(x[0]["heading"])) for x in lane_keeper_log_list]
    lane_keeper_act_list = [(x[1][0],math.radians(x[1][1])) for x in lane_keeper_log_list]

    #import pdb
    #pdb.set_trace()


    results = open("{}/lane_keeping_results-{}.txt".format(DATA_ADDRESS,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),"w")
    results.write("num_cars: {}\n".format(num_cars))
    results.write("lane_width: {}\n".format(lane_width))
    results.write("veh_length: {}\n".format(veh_length))
    results.write("veh_width: {}\n".format(veh_width))
    results.write("dt: {}\n".format(dt))
    results.write("speed_limit: {}\n\n".format(speed_limit))

    #Behaviour being modelled/learnt put in first
    results.write("States: {}\n".format(lane_keeper_state_list))
    results.write("Actions: {}\n".format(lane_keeper_act_list))

    #Behaviour of all other cars goes after
    results.write("States: {}\n".format(lane_changer_state_list))
    results.write("Actions: {}\n".format(lane_changer_act_list))
    results.close()
