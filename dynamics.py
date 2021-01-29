import theano as th
import theano.tensor as tt

class Dynamics(object):
    def __init__(self, nx, nu, f, dt=None):
        self.nx = nx #number of state dimensions
        self.nu = nu #number of action dimensions
        self.dt = dt
        if dt is None:
            self.f = f
        else:
            self.f = lambda x, u: x+dt*f(x, u)
    def __call__(self, x, u):
        return self.f(x, u)

class CarDynamics(Dynamics):
    def __init__(self, dt=0.1, Lf=1.11, Lr=1.61, friction=0):
        def f(x, u):
            slip_angle = tt.arctan(tt.tan(u[1])*Lr/(Lr+Lf))
            return tt.stacklists([
                x[2]*tt.cos(x[3]+slip_angle),
                -x[2]*tt.sin(x[3]+slip_angle),
                u[0]-x[2]*friction,
                (x[2]/Lr)*tt.sin(slip_angle)
            ])
        #4 = dimension of state, 2 = dimension of action
        Dynamics.__init__(self, 4, 2, f, dt)

if __name__ == '__main__':
    dyn = CarDynamics(0.1)
    x = tt.vector()
    u = tt.vector()
    dyn(x, u)
