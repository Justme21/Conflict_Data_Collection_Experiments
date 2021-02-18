import sys
sys.path.insert(0,'../../libraries/driving_simulator')

import datetime
import linear_controller_classes as lcc
import math
import pdb
import pyautogui
import pygame
import random
import simulator
import vehicle_classes
import road_classes

sys.path.insert(0,'../../libraries')

from trajectory_type_definitions import LaneChangeController

DATA_ADDRESS = "../../results/exp1/b"

cur_score = None


def initialiseSimulator(cars,speed_limit,init_speeds=None,vehicle_spacing=3,lane_width=None,dt=.1,sim_position=None,sim_dimensions=None,debug=False):
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
    road_lengths = [5,100] #Shorter track for generating results
    junc_pairs = [(0,1),(1,2)]

    starts = [[(0,1),1],[(0,1),0]] #Follower car is initialised on the first road, leading car on the 3rd (assuring space between them)
    dests = [[(1,2),1],[(1,2),1]] #Simulation ends when either car passes the end of the 

    run_graphics = True
    draw_traj = False #trajectories are uninteresting by deafault

    runtime = 120.0 #max runtime; simulation will terminate if run exceeds this length of time

    #Initialise the simulator object, load vehicles into the simulation, then initialise the action simulation
    sim = simulator.Simulator(run_graphics,draw_traj,runtime,debug,dt=dt,graphic_position=sim_position,graphic_dimensions=sim_dimensions)
    sim.loadCars(cars)

    sim.initialiseSimulator(num_junctions,num_roads,road_angles,road_lengths,junc_pairs,\
                                                    car_speeds,starts,dests,lane_width=lane_width)
    return sim

###################################################################################################################
#Car controller triggers

def onLaneTrigger(ego_car, trigger_car):
    #Triggered if trigger car is on the same lane as ego car
    def f():
        lanes = [x for x in ego_car.on if isinstance(x,road_classes.Lane)]
        return (True in [trigger_car in x.on for x in lanes])

    return f


def radiusTrigger(ego_car,trigger_car,radius):
    #Triggered if trigger car is too close to ego car
    def f():
        return computeDistance(ego_car.state["position"],trigger_car.state["position"])<radius

    return f


def distanceTravelledTrigger(ego_car,threshold):
    def f():
        return ego_car.state["position"][0]>threshold

    return f


def relativeXRadiusTrigger(ego_car,trigger_car,radius,rel='>'):
    def f():
        if rel == '=': return ego_car.state["position"][0] - trigger_car.state["position"][0] == radius
        elif rel == '<': return ego_car.state["position"][0] - trigger_car.state["position"][0] < radius
        else: return ego_car.state["position"][0] - trigger_car.state["position"][0] > radius

    return f


def relativeYRadiusTrigger(ego_car,trigger_car,radius,rel='>'):
    def f():
        if rel == '=': return abs(ego_car.state["position"][1] - trigger_car.state["position"][1]) == radius
        elif rel == '<': return abs(ego_car.state["position"][1] - trigger_car.state["position"][1]) < radius
        else: return abs(ego_car.state["position"][1] - trigger_car.state["position"][1]) > radius

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


def changeController(car,tag):
    def f():
        old_controller = car.controller
        car.setController(tag=tag)
        #Transfer log from old controller to new so they seem like a single contiuous controller
        new_controller = car.controller
        new_controller.log = list(old_controller.log)

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
#Score functions

def computeDistance(pt1,pt2):
    """Compute the L2 distance between two points"""
    return math.sqrt((pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2)


def timeCost(car,dt,max_score):
    def f(): 
        return max_score - car.time/dt

    return f


def distanceCost(car1,car2,max_score,min_distance):
    def f():
        global cur_score
        cur_score =  min(cur_score,max_score*(computeDistance(car1.state["position"],car2.state["position"])/min_distance))
        return cur_score

    return f

###################################################################################################################
#Direction Writing Stuff
def writeTextToScreen(screen,text_lines,start_position,font_size,space_size):
    white = (255,255,255)
    font = pygame.font.Font(None,font_size)
    text = [font.render(l,1,white) for l in text_lines]
    for i,line in enumerate(text):
        screen.blit(line,(start_position[0],start_position[1]+font_size*i+space_size*i))


def writeScore(screen,scoreFunction,start_position,font_size,space_size):
    def f():
        line = ["Maximise your Score","Score: {}".format(round(scoreFunction(),1))]
        writeTextToScreen(screen,line,start_position,font_size,space_size)

    return f


def writeText(screen,text_list,start_position,font_size,space_size):
    def f():
        writeTextToScreen(screen,text_list,start_position,font_size,space_size)

    return f

#Trigger functions to fire writing tasks
def trueFunc():
    return True


def trueFunc1():
    return trueFunc()


def trueFunc2():
    return trueFunc()


def runExperiment(experiment_order):
    #debug mode
    debug = False
    exp_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ##########################################################################################
    #Setting Up Simulation
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
    participant_accel_range = [-6,3]
    accel_range = [-3,3]
    yaw_rate_range = [-10,10] # degree per second^2
    
    #Define the car(s)
    #Human Controlled Car (this is the car being modelled)
    lane_keeper = vehicle_classes.Car(None,True,1,timestep=dt,car_params={"length":veh_length,"width":veh_width},debug=debug)
    manual_controller = lcc.DrivingController(controller="manual",ego=lane_keeper,speed_limit=speed_limit,yaw_rate_range=yaw_rate_range,accel_range=participant_accel_range,accel_jerk=accel_jerk,yaw_rate_jerk=yaw_rate_jerk)
    lane_keeper.addControllers({"default":manual_controller})
    lane_keeper.setController(tag="default")

    #Other Car
    lane_changer = vehicle_classes.Car(None,False,2,timestep=dt,car_params={"length":veh_length,"width":veh_width},debug=debug)
   
    ###########################################################################################
    #Initialise Simulator here because need state definition

    sim_position = (0,0)
    w,h = pyautogui.size()
    sim_dimensions = (w,h)
    sim = initialiseSimulator([lane_changer,lane_keeper],speed_limit,init_speeds=[init_speed,init_speed],lane_width=lane_width,dt=dt,debug=debug,sim_position=sim_position,sim_dimensions=sim_dimensions)

    lane_keeper.heading = (lane_keeper.heading+180)%360
    lane_keeper.initialisation_params["heading"] = lane_keeper.heading
    lane_keeper.sense()

    ###########################################################################################
    #Write Instructions
    init_score = 200
    global cur_score
    cur_score  = init_score

    pygame.init()
    g_sim = sim.g_sim

    screen = sim.g_sim.screen #This is messy, but the best way to get this I think
    font_size = 40
    space_size = 15

    instructions = ["-Press and hold UP arrow to accelerate","-Press and hold DOWN arrow to decelerate","-Press and hold the LEFT arrow to turn anti-clockwise","-Press and hold RIGHT arrow to turn clockwise","-Press SPACE to pause simulation"]
    write_instructions = writeText(screen,instructions,(0,int(h/5)),font_size,space_size)
    
    ###########################################################################################
    #Setting up Controllers for Lane Changing Vehicle
    #First Lane Change Controller: Try lane change
    right_lane_state = lane_keeper.state.copy() #lane keeper is initially centred in the right lane

    right_lane_state["velocity"] = speed_limit
    lcT = 15
    lc_controller = lcc.DrivingController(controller="NA",ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk)
    lc_controller.controller = LaneChangeController(ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,dest_state=right_lane_state,axle_length=axle_length,T=lcT)

    lane_changer.addControllers({"default":lc_controller})
    lane_changer.setController(tag="default",controller=None)

    #Second Lane Change Controller: Lane Change Successful (shorten up the remaining trajectory)
    slcT = 5
    slc_controller = lcc.DrivingController(controller="NA",ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk)
    slc_controller.controller = LaneChangeController(ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,dest_state=right_lane_state,axle_length=axle_length,T=slcT)
    lane_changer.addControllers({"short_lane_change":slc_controller})

    short_lane_change_trigger = onLaneTrigger(lane_changer,lane_keeper) and relativeXRadiusTrigger(lane_changer,lane_keeper,veh_length,'>')
    short_lane_change_consequent = changeController(lane_changer,"short_lane_change")
    
    #Third Lane Change Controller: Lane Change Rejected
    left_lane_state = lane_changer.state.copy()
    left_lane_state["velocity"] = 4 #2*init_speed - speed_limit

    rlcT = 5
    rlc_controller = lcc.DrivingController(controller="NA",ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk)
    rlc_controller.controller = LaneChangeController(ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,dest_state=left_lane_state,axle_length=axle_length,T=rlcT)
    lane_changer.addControllers({"reverse_lane_change":rlc_controller})

    #Fourth Lane Change Controller: Lane Change Behind
    lcbT = 5
    slow_right_lane_state = lane_keeper.state.copy() #lane keeper is initially centred in the right lane
    behind_controller = lcc.DrivingController(controller="NA",ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,accel_jerk=accel_jerk)
    behind_controller.controller = LaneChangeController(ego=lane_changer,other=None,timestep=dt,speed_limit=speed_limit,accel_range=accel_range,dest_state=slow_right_lane_state,axle_length=axle_length,T=lcbT)
    lane_changer.addControllers({"lane_change_behind":behind_controller})

    lane_change_behind_trigger = relativeXRadiusTrigger(lane_keeper,lane_changer,veh_length,rel='>')
    lane_change_behind_consequent = changeController(lane_changer,"lane_change_behind")
   
    heading_fix_trigger = headingFixTrigger(lane_changer)
    heading_fix_consequent = fixHeading(lane_changer)

    #To end the simulation
    heading_radius = 2
    lane_changer_heading_trigger = headingTrigger(lane_changer,heading_radius)
    y_dist_trigger = relativeYRadiusTrigger(lane_changer,lane_keeper,lane_width/4,'<')
    sim_triggers = {andTrigger([distanceTravelledTrigger(lane_keeper,2*w/3),onLaneTrigger(lane_changer,lane_keeper),lane_changer_heading_trigger,y_dist_trigger]):sim.endSimulation}
    sim.addTriggers(sim_triggers)
 
    ##################################################################################################################
    #Run Experiments
    for i,(lane_keeper_type,lane_changer_type) in enumerate(experiment_order):
        ######################################################################
        #Set other car triggers
        lane_changer.triggers = {} #erase all previous triggers
        if lane_changer_type == "aggressive":
            risk_radius = veh_width
            reverse_lane_change_trigger = andTrigger([radiusTrigger(lane_changer,lane_keeper,risk_radius), relativeXRadiusTrigger(lane_changer,lane_keeper,veh_length,'<')]) #aggressive
        else:
            reverse_lane_change_trigger = andTrigger([onLaneTrigger(lane_changer,lane_keeper), relativeXRadiusTrigger(lane_changer,lane_keeper,veh_length,'<')]) #passive
        reverse_lane_change_consequent = changeController(lane_changer,"reverse_lane_change")
        
        triggers = {short_lane_change_trigger:short_lane_change_consequent, reverse_lane_change_trigger:reverse_lane_change_consequent,lane_change_behind_trigger:lane_change_behind_consequent,\
                    heading_fix_trigger:heading_fix_consequent}
        lane_changer.addTriggers(triggers)

        ######################################################################
        #Set Graphic Simulator triggers
        cur_score = init_score

        if lane_keeper_type == "aggressive":
            #idirective = "Get to the end of the lane as quickly as possible"
            directive = "Drive as if you are in a rush and stay in your lane"
            score_function = timeCost(lane_keeper,dt,init_score)
        else:
            directive = "Drive as if you were on a leisurely drive and stay in your lane"
            score_function = distanceCost(lane_keeper,lane_changer,init_score,veh_length)
        
        write_task  = writeText(screen,[directive],(int(w/2),int(h/5)),font_size,space_size)
        write_score = writeScore(screen,score_function,(int(w/2),int(h/5)+(font_size+space_size)),font_size,space_size)
        
        triggers = {trueFunc:write_instructions,trueFunc1:write_score,trueFunc2:write_task}
        g_sim.triggers = {}
        g_sim.addTriggers(triggers)
        
        ######################################################################
        #Run simulation
        sim.reinitialise()
        sim.runComplete() #will start from paused

        ######################################################################
        #Extract log of behaviours from controller
        num_cars = 2

        lane_changer_log_list = lane_changer.controller.getLog()
        lane_keeper_log_list = manual_controller.getLog()

        #Behaviour being modelled/learnt put in first
        lane_changer_state_list = [(x[0]["position"][0],x[0]["position"][1],x[0]["velocity"],math.radians(x[0]["heading"])) for x in lane_changer_log_list]
        lane_changer_act_list = [(x[1][0],math.radians(x[1][1])) for x in lane_changer_log_list]

        #Behaviour of all other cars goes after
        lane_keeper_state_list = [(x[0]["position"][0],x[0]["position"][1],x[0]["velocity"],math.radians(x[0]["heading"])) for x in lane_keeper_log_list]
        lane_keeper_act_list = [(x[1][0],math.radians(x[1][1])) for x in lane_keeper_log_list]

        results = open("{}/lane_keeping_results-{}-{}.txt".format(DATA_ADDRESS,exp_start_time,i),"w")
        results.write("num_cars: {}\n".format(num_cars))
        results.write("lane_width: {}\n".format(lane_width))
        results.write("veh_length: {}\n".format(veh_length))
        results.write("veh_width: {}\n".format(veh_width))
        results.write("dt: {}\n".format(dt))
        results.write("speed_limit: {}\n".format(speed_limit))

        #Behaviour being modelled/learnt put in first
        results.write("\nEgo\n")
        results.write("Type: {}\n".format(lane_keeper_type))
        results.write("On Road: {}\n".format(int(lane_keeper.on_road)))
        results.write("Crash: {}\n".format(int(lane_keeper.crashed)))
        results.write("States: {}\n".format(lane_keeper_state_list))
        results.write("Actions: {}\n".format(lane_keeper_act_list))

        #Behaviour of all other cars goes after
        results.write("\nOthers\n")
        results.write("Type: {}\n".format(lane_changer_type))
        results.write("On Road: {}\n".format(int(lane_changer.on_road)))
        results.write("Crash: {}\n".format(int(lane_changer.crashed)))
        results.write("States: {}\n".format(lane_changer_state_list))
        results.write("Actions: {}\n".format(lane_changer_act_list))
        results.close()

    #Shut down the graphic screen
    sim.wrapUp()


if __name__ == "__main__":
    num_observations = 1
    lane_keeper_type,lane_changer_type = [],[]
    for _ in range(num_observations):
        lane_keeper_type += ["aggressive","passive","aggressive","passive"] #(aggressive/passive) This dictates the instruction to be provided
        lane_changer_type += ["passive","passive","aggressive","aggressive"]#(aggressive/passive)

    experiment_types = list(zip(lane_keeper_type,lane_changer_type))

    experiment_order = random.sample(experiment_types,len(experiment_types))
    #experiment_order = [("passive","aggressive")] #(lane_keeper,lane_changer)

    runExperiment(experiment_order)
