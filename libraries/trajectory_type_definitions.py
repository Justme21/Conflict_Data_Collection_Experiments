import math
import numbers

class Line():
    def __init__(self,*args):
        args = list(args)
        args.reverse()
        self.coefs = [(x,i) for i,x in enumerate(args)]


    def dot(self,coefs=None):
        if coefs is None: coefs = list(self.coefs)
        else: coefs = list(coefs)

        for i in range(len(coefs)):
            if coefs[i][1] == 0:
                coefs[i] = (0,0)
            else:
                coefs[i] = (coefs[i][0]*coefs[i][1],coefs[i][1]-1)
        return coefs


class LaneChangeController():
    def __init__(self,ego=None,timestep=.1,accel_range=None,dest_state=None,axle_length=4.6,T=None,**kwargs):
        self.ego = ego
        self.accel_range = accel_range
        self.dt = timestep
        self.dest_state = dest_state

        self.trajectory = None
        self.index = 0
        self.axle_length = axle_length
        self.T = T


    def setup(self,ego=None,**kwargs):
        if ego is not None:
            self.ego = ego


    def selectAction(self,state,*args):
        if self.index*self.dt<self.T:
            self.trajectory = LaneChangeTrajectory(state,self.dest_state,self.T-self.index*self.dt)
            accel,yaw_rate = self.trajectory.action(0,self.axle_length)
            #accel,yaw_rate = self.trajectory.action(self.index*self.dt,self.axle_length)
            self.index += 1
        elif self.index*self.dt==self.T:
            accel,yaw_rate = self.trajectory.action(self.dt,self.axle_length)
            self.index += 1
        else:
            accel,yaw_rate = 0,0
        return accel,yaw_rate


class LaneChangeTrajectory():
    def __init__(self,init_state,dest_state,T):
        if init_state is None or dest_state is None:
            print("Error, default values for states are invalid")
            exit(-1)
        self.traj_len_t = T
        self.line_x,self.line_y = laneChange(init_state,dest_state,T)
        self.computeDerivatives()


    def computeDerivatives(self):
        self.x = self.line_x.coefs
        self.y = self.line_y.coefs
        self.x_dot = self.line_x.dot()
        self.y_dot = self.line_y.dot()
        self.x_dot_dot = self.line_x.dot(self.x_dot)
        self.y_dot_dot = self.line_y.dot(self.y_dot)


    def action(self,t,axle_length):
        if t>self.traj_len_t:
            return 0,0
        else:
            x = evaluate(t,self.x)
            y = evaluate(t,self.y)

            x_dot = evaluate(t,self.x_dot)
            y_dot = evaluate(t,self.y_dot)
            x_dot_dot = evaluate(t,self.x_dot_dot)
            y_dot_dot = evaluate(t,self.y_dot_dot)

            denom_a = math.sqrt(x_dot**2 + y_dot**2)
            denom_yaw = denom_a**3

            acceleration = ((x_dot*x_dot_dot)+(y_dot*y_dot_dot))/denom_a
            yaw_rate = math.degrees(math.atan(((x_dot*-y_dot_dot)-(-y_dot*x_dot_dot))*axle_length/denom_yaw))

        return acceleration,yaw_rate


    def position(self,t):
        return (evaluate(t,self.x),evaluate(t,self.y))


    def velocity(self,t):
        return math.sqrt(evaluate(t,self.x_dot)**2 + evaluate(t,self.y_dot)**2)


    def heading(self,t):
        x_dot = evaluate(t,self.x_dot)
        #minus here to capture the axis flip
        y_dot = -evaluate(t,self.y_dot)

        if x_dot == 0:
            if y_dot>0: heading = 90
            else: heading = 270

        else:
            heading = math.degrees(math.atan(y_dot/x_dot))
            if x_dot<0: heading = (heading+180)%360#atan has domain (-90,90) 

        heading%=360
        return heading


    def state(self,t,axle_length=None):
        """Returns the estimated state at a known timepoint along the trajectory.
           ACtion omitted as this would require vehicle axle length"""
        posit = self.position(t)
        vel = self.velocity(t)
        heading = self.heading(t)

        if axle_length is not None:
            acceleration,yaw_rate = self.action(t,axle_length)
        else:
            acceleration,yaw_rate = None,None

        state = {"position":posit,"velocity":vel,"heading":heading,"acceleration":acceleration,"yaw_rate":yaw_rate}
        return state


    def completePositionList(self,dt=.1):
        t = 0
        position_list = []
        while t<=self.traj_len_t+dt:
            position_list.append(self.position(t))
            t += dt

        return position_list

    def completeHeadingList(self,dt=.1):
        t = 0
        heading_list = []
        while t<= self.traj_len_t+dt:
            heading_list.append(self.heading(t))
            t += dt

        return heading_list


    def completeVelocityList(self,dt=.1):
        t = 0
        velocity_list = []
        while t<=self.traj_len_t+dt:
            velocity_list.append(self.velocity(t))
            t += dt

        return velocity_list


    def completeActionList(self,axle_length,dt=.1):
        t = 0
        action_list = []
        while t<=self.traj_len_t+dt:
            action_list.append(self.action(t,axle_length))
            t += dt

        return action_list


def laneChange(init_state,dest_state,T):
    init_pos = init_state["position"]
    init_vel = init_state["velocity"]
    init_heading = init_state["heading"]

    dest_pos = dest_state["position"]
    dest_vel = dest_state["velocity"]
    dest_heading = dest_state["heading"]

    #Translate to global coordinates
    init_vel = (init_vel*math.cos(math.radians(init_heading)),-init_vel*math.sin(math.radians(init_heading)))
    dest_vel = (dest_vel*math.cos(math.radians(dest_heading)),-dest_vel*math.sin(math.radians(dest_heading)))

    A_y = (2/T**3)*((T/2)*init_vel[1]+init_pos[1]-dest_pos[1])
    B_y = (1/(2*T))*(-3*A_y*T**2 - init_vel[1])
    C_y = init_vel[1]
    D_y= init_pos[1]

    line_y = Line(A_y,B_y,C_y,D_y)

    A_x = (dest_vel[0]-init_vel[0])/(2*T)
    B_x = init_vel[0]
    C_x = init_pos[0]
    line_x = Line(A_x,B_x,C_x)

    return line_x,line_y


def evaluate(t,coefs):
    return sum([entry[0]*(t**entry[1]) for entry in coefs])
