import csv
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
from scipy import stats


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
        self.pc_creation_noise = 0.3
        self.pc_measure_noise = 0.6
        self.measurement_distances_mode = []

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
        self.reset_lists()
        self.drive_a_long_curve()
        self.reset_lists()
        self.drive_s_curve_with_constant_velocity()
        self.reset_lists()
        self.drive_s_curve_with_variable_velocity()

    def drive_s_curve_with_variable_velocity(self):
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.imu_std, dt=config.dt)
        u = np.array([0,0], dtype=float)
        self.xs.append(model.get_initial_state(x=1, y=1, v=5, a=0, theta=0, delta=0))
        
        for i in range(1,2000): 
            # set steering
            if (i < 200 and u[1] < config.max_steering_angle): 
                u[1] += 0.001
            elif (i> 200 and i < 600 and u[1] > -config.max_steering_angle): 
                u[1] -= 0.001
            elif (i > 600 and i < 1000 and u[1] < config.max_steering_angle): 
                u[1] += 0.001
            elif (i > 1000 and i < 1400 and u[1] > -config.max_steering_angle): 
                u[1] -= 0.001
            elif (i > 1400 and i < 1800 and u[1] < config.max_steering_angle): 
                u[1] += 0.001
            elif (i > 1800 and i < 2000 and u[1] > 0): 
                u[1] -= 0.001

            # set acceleration
            if (i < 100 and u[0] < 15): 
                u[0] += 0.01
            elif (i > 100 and i < 150 and u[0] > 2): 
                u[0] -= 0.01
            elif (i > 150 and i < 200 and u[0] > -1): 
                if (self.xs[i-1][2] >= 0):
                    u[0] -= 0.1
            elif (i > 200 and i < 600 and u[0] < 20): 
                u[0] += 0.001
            elif(i > 600 and i < 1200): 
                u[0] = 0
            self.xs.append(model.F(x=self.xs[i-1],u=u))          
            self.car_ci_accelerations.append(u[0])
            self.car_ci_steerings.append(u[1])
            self.car_gt_positions_x.append(self.xs[i-1][0])
            self.car_gt_positions_y.append(self.xs[i-1][1])
            self.car_gt_velocities.append(self.xs[i-1][2])
            self.car_gt_timestamps.append(config.dt*i)
            points = self.create_points_from_pos(np.array([self.xs[i-1][0], self.xs[i-1][1]]), self.xs[i-1][4])

            for p in points:
                self.pc_env_x.append(p[0] + np.random.randn() * self.pc_creation_noise)
                self.pc_env_y.append(p[1] + np.random.randn() * self.pc_creation_noise)
        
        pc = np.stack([self.pc_env_x, self.pc_env_y], axis=1)        
        positions = np.stack([self.car_gt_positions_x, self.car_gt_positions_y], axis=1)
        # create measurements
        for p in positions:
            subs = p - pc
            dists = np.linalg.norm(subs, axis=1)
            in_range = dists[dists < config.lidar_range]
            self.measurement_distances_mode.append(stats.mode(in_range)[0][0])
            
        self.write_result_to_csv(config.s_curve_name_variable_velocity)
        self.save_point_cloud(config.s_curve_name_variable_velocity)

    def drive_s_curve_with_constant_velocity(self):
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.imu_std, dt=config.dt)
        u = np.array([0,0], dtype=float)
        self.xs.append(model.get_initial_state(x=1, y=1, v=5, a=0, theta=0, delta=0))
        
        for i in range(1,2000): 
            if (i < 200 and u[1] < config.max_steering_angle): 
                u[1] += 0.001
            elif (i> 200 and i < 600 and u[1] > -config.max_steering_angle): 
                u[1] -= 0.001
            elif (i > 600 and i < 1000 and u[1] < config.max_steering_angle): 
                u[1] += 0.001
            elif (i > 1000 and i < 1400 and u[1] > -config.max_steering_angle): 
                u[1] -= 0.001
            elif (i > 1400 and i < 1800 and u[1] < config.max_steering_angle): 
                u[1] += 0.001
            elif (i > 1800 and i < 2000 and u[1] > 0): 
                u[1] -= 0.001
            self.xs.append(model.F(x=self.xs[i-1],u=u))          
            self.car_ci_accelerations.append(u[0])
            self.car_ci_steerings.append(u[1])
            self.car_gt_positions_x.append(self.xs[i-1][0])
            self.car_gt_positions_y.append(self.xs[i-1][1])
            self.car_gt_velocities.append(self.xs[i-1][2])
            self.car_gt_timestamps.append(config.dt*i)
            current_position = np.array([self.xs[i-1][0], self.xs[i-1][1]])
            points = self.create_points_from_pos(current_position, self.xs[i-1][4])

            for p in points:
                self.pc_env_x.append(p[0] + np.random.randn() * self.pc_creation_noise)
                self.pc_env_y.append(p[1] + np.random.randn() * self.pc_creation_noise)
        pc = np.stack([self.pc_env_x, self.pc_env_y], axis=1)        
        positions = np.stack([self.car_gt_positions_x, self.car_gt_positions_y], axis=1)
        # create measurements
        for p in positions:
            subs = p - pc
            dists = np.linalg.norm(subs, axis=1)
            in_range = dists[dists < config.lidar_range]
            self.measurement_distances_mode.append(stats.mode(in_range)[0][0])
        self.write_result_to_csv(config.s_curve_name_constant_velocity)
        self.save_point_cloud(config.s_curve_name_constant_velocity)

        # create a circular measurement for each positions  

    def drive_a_long_curve(self):
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.imu_std, dt=config.dt)
        u = np.array([0,0], dtype=float)
        self.xs.append(model.get_initial_state(x=1, y=1, v=5, a=0, theta=0, delta=0))
        
        for i in range(1,1000): 
            u[1] += 0.0001
            if (u[0] < 10): 
                u[0] += 0.00001
            elif(u[0]>10):
                u[0] = 0
            self.xs.append(model.F(x=self.xs[i-1],u=u))          
            self.car_ci_accelerations.append(u[0])
            self.car_ci_steerings.append(u[1])
            self.car_gt_positions_x.append(self.xs[i-1][0])
            self.car_gt_positions_y.append(self.xs[i-1][1])
            self.car_gt_velocities.append(self.xs[i-1][2])
            self.car_gt_timestamps.append(config.dt*i)
            points = self.create_points_from_pos(np.array([self.xs[i-1][0], self.xs[i-1][1]]), self.xs[i-1][4])

            for p in points:
                self.pc_env_x.append(p[0] + np.random.randn() * self.pc_creation_noise)
                self.pc_env_y.append(p[1] + np.random.randn() * self.pc_creation_noise)

        pc = np.stack([self.pc_env_x, self.pc_env_y], axis=1)        
        positions = np.stack([self.car_gt_positions_x, self.car_gt_positions_y], axis=1)
        # create measurements
        for p in positions:
            subs = p - pc
            dists = np.linalg.norm(subs, axis=1)
            in_range = dists[dists < config.lidar_range]
            self.measurement_distances_mode.append(stats.mode(in_range)[0][0])
                    
        self.write_result_to_csv(config.curve_line_name)
        self.save_point_cloud(config.curve_line_name)
 
    def drive_straight_in_x_direction(self):
        self.reset_lists()

        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.imu_std, dt=config.dt)
        u = np.array([0,0], dtype=float)
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
            for p in points:
                self.pc_env_x.append(p[0] + np.random.randn() * self.pc_creation_noise)
                self.pc_env_y.append(p[1] + np.random.randn() * self.pc_creation_noise)

        pc = np.stack([self.pc_env_x, self.pc_env_y], axis=1)        
        positions = np.stack([self.car_gt_positions_x, self.car_gt_positions_y], axis=1)
        # create measurements
        for p in positions:
            subs = p - pc
            dists = np.linalg.norm(subs, axis=1)
            in_range = dists[dists < config.lidar_range]
            self.measurement_distances_mode.append(stats.mode(in_range)[0][0])
                
            
        self.write_result_to_csv(config.straight_x_line_name)
        self.save_point_cloud(config.straight_x_line_name)

    
    def create_points_from_pos(self, pos, theta): 
        forward_vec = np.array(np.cos(theta), np.sin(theta)) + pos
        perpendicular_vec1 = np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])*10  + pos
        perpendicular_vec2 = np.array([np.cos(theta-np.pi/2), np.sin(theta-np.pi/2)])*10  + pos
        return np.array([perpendicular_vec1, perpendicular_vec2])
    
    def write_result_to_csv(self, type): 
        type = type+config.lidar_data_appendix
        basic_data = {
            'acceleration_input': self.car_ci_accelerations,
            'steering_input': self.car_ci_steerings,              
            'positions_x_ground_truth': self.car_gt_positions_x,
            'positions_y_ground_truth': self.car_gt_positions_y,
            'velocities_ground_truth': self.car_gt_velocities,
            'measurements': self.measurement_distances_mode, 
            'timestamps': self.car_gt_timestamps
        }
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+type + config.data_suffix, basic_data)

        
    def save_point_cloud(self, type): 
        data = {
            'pc_x': self.pc_env_x,
            'pc_y': self.pc_env_y
        }
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+type+config.point_cloud_appendix, data)

        #xyz = np.stack([self.pc_env_x, self.pc_env_y, np.full((len(self.pc_env_x),), 0)],axis=1)
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(xyz)
        #o3d.io.write_point_cloud(config.paths['data_path']+type+config.point_cloud_appendix, pcd)

 


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
        self.measurement_distances_mode = []

        # pc
        self.pc_env_x = []
        self.pc_env_y = []

        self.xs = []
