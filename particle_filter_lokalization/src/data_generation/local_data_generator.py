import numpy as np
import pandas as pd

import process_model.front_wheel_bycicle_model as fw_bycicle_model
import config.config as config
import utils.write_to_csv as write_to_csv
import utils.save_image as save_image
import cv2 as cv
class LocalDataGenerator: 
    def __init__(self):
        # control inputs
        self.car_ci_accelerations = []
        self.car_ci_steerings = []
    
        # ground truth values
        self.car_gt_positions_x = []
        self.car_gt_positions_y = []
        self.car_gt_velocities = []
        self.car_gt_timestamps = []

        # measurements
        self.car_m_accelerations = []
        self.car_m_orientations = []

        # distance transform
        self.map_shape = (0,0)
        self.map = []
        self.distance_map = []
        self.x_range = np.array([0,0])
        self.y_range = np.array([0,0])
        self.accounted_decimal_places = 2


    def drive_straight_in_x_direction(self):
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.std, dt=config.dt)
        model.set_initial_state(x=0, y=0, v=5, a=0, theta=0, delta=0)
        u = np.array([0,0])
        for i in range(1000): 
            model.F(u=u)
            x = model.get_current_state()
            self.car_ci_accelerations.append(u[0])
            self.car_ci_steerings.append(u[1])
            self.car_gt_positions_x.append(x[0])
            self.car_gt_positions_y.append(x[1])
            self.car_gt_velocities.append(x[2])
            self.car_gt_timestamps.append(config.dt*i)
            self.car_m_accelerations.append(x[3]+config.sensor_std[0]*np.random.randn()) 
            self.car_m_orientations.append(x[4]+config.sensor_std[1]*np.random.randn())   

        self.write_result_to_csv("straight_in_x_")
    def write_result_to_csv(self, type): 
        data = {
            'acceleration_input': self.car_ci_accelerations,
            'steering_input': self.car_ci_steerings, 
            'acceleration_measurement': self.car_m_accelerations, 
            'orientation_measurement': self.car_m_orientations, 
            'positions_x_ground_truth': self.car_gt_positions_x,
            'positions_y_ground_truth': self.car_gt_positions_y,
            'velocities_ground_truth': self.car_gt_velocities, 
            'timestamps': self.car_gt_timestamps
        }
        write_to_csv.write_to_csv(type + str("data_"), data)
        #save_image.save_array_as_image(np.stack([self.car_gt_positions_x, self.car_gt_positions_y],axis=1),"map_image_straight_line" )
        self.create_map_image("map_image_straight_line")


    def positions_to_image_coordinates(self):
        

    def create_map_image(self, name): 
        self.x_range = (np.array(self.car_gt_positions_x).min(), np.array(self.car_gt_positions_x).max())
        self.y_range = (np.array(self.car_gt_positions_y).min(), np.array(self.car_gt_positions_y).max())
        
        
        self.map_shape = (self.y_range[1] - self.y_range[0]+1, self.x_range[1] - self.x_range[0]+1)
        print(self.map_shape)
        self.map = np.array(np.zeros(self.map_shape))  
        
        position_vectors = np.stack([self.car_gt_positions_x, self.car_gt_positions_y],axis=1)
        for pos in position_vectors: 
            self.map[int(pos[0]*10**self.accounted_decimal_places-self.x_range[0]*10**self.accounted_decimal_places), int(pos[1]*10**self.accounted_decimal_places-self.y_range[0]*10**self.accounted_decimal_places)] = 1

        save_image.save_array_as_image(self.map,"map_image_straight_line" )

