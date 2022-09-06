import json
from pickletools import uint8
from turtle import title
import numpy as np
import pandas as pd

import process_model.front_wheel_bycicle_model as fw_bycicle_model
import config.config as config
import utils.csv_handler as csv_handler
import utils.json_handler as json_handler
import utils.image_handler as image_handler
import cv2 as cv
import matplotlib.pyplot as plt
class LocalDataGeneratorLIDAR: 
    def __init__(self):
        # control inputs
        self.car_ci_accelerations = []
        self.car_ci_steerings = []
    
        # ground truth values
        self.car_gt_positions_x = []
        self.car_gt_positions_y = []
        self.car_gt_timestamps = []

        # measurements
        self.xs = []
        # point cloud environment
        self.pc_env_x = []
        self.pc_env_y = []
        self.pc_creation_noise = .2
        # point cloud measurement
        self.pc_measures = []
        self.pc_measures_noise = 0.3

    def generate_data(self, data_type): 
        if (data_type == config.straight_x_line_name): 
            self.drive_straight_in_x_direction()
        elif(data_type == config.curve_line_name): 
            self.drive_a_long_curve()
        elif(data_type == config.s_curve_name_constant_velocity): 
            self.drive_s_curve_with_constant_velocity()
        elif(data_type == config.s_curve_name_variable_velocity): 
            self.drive_s_curve_with_variable_velocity()
        elif(data_type == "all"): 
            self.generate_all_data()

    def generate_all_data(self): 
        self.drive_straight_in_x_direction()
        self.drive_a_long_curve()
        self.drive_s_curve_with_constant_velocity()
        self.drive_s_curve_with_variable_velocity()

 
    def drive_straight_in_x_direction(self):
        self.reset_lists()

        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.std, dt=config.dt)
        u = np.array([0,0])
        self.xs.append(model.get_initial_state(x=0, y=0, v=5, a=0, theta=0, delta=0))
        
        for i in range(1,1000): 
            self.xs.append(model.F(x=self.xs[i-1],u=u))          
            self.car_ci_accelerations.append(u[0])
            self.car_ci_steerings.append(u[1])
            self.car_gt_positions_x.append(self.xs[i-1][0])
            self.car_gt_positions_y.append(self.xs[i-1][1])
            self.car_gt_velocities.append(self.xs[i-1][2])
            self.car_gt_timestamps.append(config.dt*i)
            points = self.create_points_from_pos(np.array([self.xs[i-1][0], self.xs[i-1][1]]), self.xs[i-1][4])
            print(points)
            for p in points:
                print(p)
                self.pc_env_x.append(p[0]) + np.random.randn() * self.pc_measures_noise
                self.pc_env_y.append(p[1]) + np.random.randn() * self.pc_measures_noise
                
            
        self.write_result_to_csv(config.straight_x_line_name)
        self.save_point_cloud(config.straight_x_line_name)

    def create_points_from_pos(self, pos, theta): 
        forward_vec = np.array(np.cos(theta), np.sin(theta)) + pos
        perpendicular_vec1 = np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])  + pos
        perpendicular_vec2 = np.array([np.cos(theta-np.pi/2), np.sin(theta-np.pi/2)])  + pos
        return np.array([perpendicular_vec1, perpendicular_vec2])
    
    def write_result_to_csv(self, type): 
        type = type+config.lidar_data_appendix
        data = {
            'acceleration_input': self.car_ci_accelerations,
            'steering_input': self.car_ci_steerings,              
            'positions_x_ground_truth': self.car_gt_positions_x,
            'positions_y_ground_truth': self.car_gt_positions_y,
            'velocities_ground_truth': self.car_gt_velocities, 
            'timestamps': self.car_gt_timestamps
        }
        csv_handler.write_to_csv(config.paths['data_path']+type + config.data_suffix, data)

        return True

    def save_point_cloud(self, type): 
        data = {
            'pc_x': self.pc_env_x,
            'pc_y': self.pc_env_y
        }
        csv_handler.write_to_csv(config.paths['data_path']+type+config.point_cloud_appendix, data)

    def reset_lists(self): 
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

        # pc
        self.pc_env_x = []
        self.pc_env_y = []
