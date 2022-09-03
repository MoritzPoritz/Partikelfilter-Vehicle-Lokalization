import numpy as np

class FrontWheelBycicleModel: 
    def __init__(self, vehicle_length, control_input_std, dt): 
        self.vehicle_length = vehicle_length
        self.control_input_std = control_input_std
        self.current_state = []
        self.past_states = []
        self.dt = dt
    # u = acc, steering 
    # z = acc_x, acc_y, rot
    # x = x, y, v, a, theta, delta
    def F(self, u): 
        # get the last state
        x = self.past_states[-1]
        # calculate new acceleration 
        a_next = u[0] + (np.random.randn() * self.control_input_std[0])
        # calculate new velocity with new acceleration 
        v_next = x[2] + (a_next * self.dt)
        # calculate new delta from control input
        delta_next = u[1] + (np.random.randn() * self.control_input_std[1])
        # calculate new theta angle
        theta_next = x[4] + (((v_next*np.sin(delta_next))/self.vehicle_length) * self.dt)
        # calculate new positions
        x_next = x[0] + (v_next*np.cos(theta_next + delta_next) * self.dt)
        y_next = x[1] + (v_next*np.sin(theta_next + delta_next) * self.dt)
        # update current state
        self.current_state = np.array([x_next, y_next,v_next, a_next, theta_next, delta_next], dtype=object) 
        # append past states with current states
        self.past_states.append(self.current_state)
    def get_current_state(self): 
        return np.array(self.current_state)

    def get_past_states(self): 
        return np.array(self.past_states)

    def set_initial_state(self, x, y, v, a, theta, delta): 
        self.past_states.append(np.array([x,y,v,a,theta,delta]))