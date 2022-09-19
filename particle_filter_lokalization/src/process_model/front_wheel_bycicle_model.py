import numpy as np

class FrontWheelBycicleModel: 
    def __init__(self, vehicle_length, control_input_std, dt): 
        self.vehicle_length = vehicle_length
        self.control_input_std = control_input_std
        self.dt = dt
    # u = acc, steering 
    # z = acc_x, acc_y, rot
    # x = x, y, v, a, theta, delta
    #F(x=self.particles[i], u=u, step=config.dt, L=config.L, std=config.std, N=len(self.particles))
    def F(self, x,u): 
        
        # calculate new positions
        x_next = x[0] + (x[2]*np.cos(x[4] + x[5]) * self.dt)
        y_next = x[1] + (x[2]*np.sin(x[4] + x[5]) * self.dt)
        
         # calculate new velocity with new acceleration 
        v_next = x[2] + (x[3] * self.dt)
        # ensuring velocity gets not negative 
        if (v_next < 0): 
            v_next = 0
        # calculate new acceleration 
        a_next = u[0] + (np.random.randn() * self.control_input_std[0])
        # calculate new theta angle
        theta_next = x[4] + (((x[2]*np.sin(x[5]))/self.vehicle_length) * self.dt)
        # calculate new delta from control input
        delta_next = u[1] + (np.random.randn() * self.control_input_std[1])
        print(delta_next)

        return np.array([x_next, y_next,v_next, a_next, theta_next, delta_next], dtype=object)
    
    def get_current_state(self): 
        return np.array(self.current_state)


    def get_initial_state(self, x, y, v, a, theta, delta): 
        return np.array([x,y,v,a,theta,delta])