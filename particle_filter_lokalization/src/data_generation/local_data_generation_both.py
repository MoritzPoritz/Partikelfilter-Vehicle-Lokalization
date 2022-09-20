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

import rain_simulation.rain_simulation as rs

from scipy import stats

class LocalDataGenerator: 
    def __init__(self):
        # control inputs
        self.car_ci_accelerations = []
        self.car_ci_steerings = []
        # ground truth
        self.car_gt_positions_x = []
        self.car_gt_positions_y = []
        self.car_gt_velocities = []
        self.car_gt_timestamps = []

        # imu measurements
        self.car_m_accelerations = []
        self.car_m_orientations = []

        #lidar measurements
        # point cloud environment
        self.pc_env_x = []
        self.pc_env_y = []
        self.reflectivities = []
        # point cloud measurement
        self.pc_creation_noise = 0.3
        self.pc_measure_noise = 0
        self.lidar_distance = config.lidar_range
        self.object_distance = 20
        self.measurement_distances_mode = []
        self.measurement_intensities_mode = []
        # distance transform
        self.map_shape = (0,0)
        self.map = []
        self.distance_map = []
        self.x_range = np.array([0,0])
        self.y_range = np.array([0,0])
        self.accounted_decimal_places = 0
        self.position_vectors_in_image_coordinates = []

        # values for rain and fog
        self.rain_rate = 0
        self.fog_rate = 0
        self.p_min = 0.9/(np.pi * config.lidar_range**2)
        
        self.xs = []
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
    
    def add_to_data(self, x, u,i):
        self.car_ci_accelerations.append(u[0])
        self.car_ci_steerings.append(u[1])
        self.car_gt_positions_x.append(x[0])
        self.car_gt_positions_y.append(x[1])
        self.car_gt_velocities.append(x[2])
        self.car_gt_timestamps.append(config.dt*i)

        #save imu measurements
        self.car_m_accelerations.append(x[3]+config.imu_sensor_std[0]*np.random.randn()) 
        self.car_m_orientations.append(x[4]+config.imu_sensor_std[1]*np.random.randn())   
        # save lidar measurements
        points = self.create_points_from_pos(np.array([x[0], x[1]]), x[4],i)
        for p in points:
            self.pc_env_x.append(p[0])
            self.pc_env_y.append(p[1])
            self.reflectivities.append(p[2])



    def create_lidar_measurement(self): 
        pc = np.stack([self.pc_env_x, self.pc_env_y], axis=1)        
        positions = np.stack([self.car_gt_positions_x, self.car_gt_positions_y], axis=1)
        # create measurements
        for p in positions:
            #position, rain_rate, pc_array, p_min, lidar_range
            ranges, intensities = rs.apply_rain(p,self.rain_rate, pc, self.p_min, config.lidar_range)
            self.measurement_distances_mode.append(stats.mode(ranges)[0][0])
            self.measurement_intensities_mode.append(stats.mode(intensities)[0][0])
            #subs = (p - pc) #+ np.random.randn(p.shape)*self.pc_measure_noise
            #dists = np.linalg.norm(subs, axis=1)
            #in_range = dists[dists < config.lidar_range]
            #self.measurement_distances_mode.append(stats.mode(in_range)[0][0])

    def drive_a_long_curve(self): 
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.imu_std, dt=config.dt)
        self.xs.append(model.get_initial_state(x=1, y=1, v=5, a=0, theta=0, delta=0))

        u = np.array([0,0], dtype=float)
        
        for i in range(1,1000): 
            u[1] = u[1] + 0.0001
            if (u[0] < 10): 
                u[0] += 0.00001
            elif(u[0]>10):
                u[0] = 0
            self.xs.append(model.F(x=self.xs[i-1],u=u)) 
            self.add_to_data(self.xs[i-1], u, i)
           
        
        self.create_lidar_measurement()
        # save the results
        self.save_point_cloud(config.curve_line_name)
        self.write_lidar_result_to_csv(config.curve_line_name)
        self.write_imu_result_to_csv(config.curve_line_name)

    def create_points_from_pos(self, pos, theta, i): 
        forward_vec = np.array(np.cos(theta), np.sin(theta)) + pos
        point_1 = (np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])+ (np.random.randn() * self.pc_creation_noise) * self.object_distance* np.sin(i))  + pos
        point_2 = (np.array([np.cos(theta-np.pi/2), np.sin(theta-np.pi/2)])+ (np.random.randn() * self.pc_creation_noise) * self.object_distance* np.sin(i))  + pos
        reflectivity_1 = np.random.random()
        reflectivity_2 = np.random.random()
        point_1 = np.append(point_1, reflectivity_1)
        point_2 = np.append(point_2, reflectivity_2)
        return np.array([point_1, point_2])

    def drive_straight_in_x_direction(self):
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.imu_std, dt=config.dt)
        u = np.array([0,0])
        self.xs.append(model.get_initial_state(x=0, y=0, v=5, a=0, theta=0, delta=0))
        for i in range(1,1000): 
            self.xs.append(model.F(x=self.xs[i-1],u=u)) 
            self.add_to_data(self.xs[i-1], u, i)
           
        
        self.create_lidar_measurement()
        # save the results
        self.save_point_cloud(config.straight_x_line_name)
        self.write_lidar_result_to_csv(config.straight_x_line_name)
        self.write_imu_result_to_csv(config.straight_x_line_name)   
            

    def drive_s_curve_with_constant_velocity(self): 
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.imu_std, dt=config.dt)
        u = np.array([0,0], dtype=float)
        self.xs.append(model.get_initial_state(x=0, y=0, v=5, a=0, theta=0, delta=0))
        
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
            self.xs.append(model.F(x=self.xs[i-1],u=u))
            self.add_to_data(self.xs[i-1], u, i)
           
        
        self.create_lidar_measurement()
        # save the results
        self.save_point_cloud(config.s_curve_name_constant_velocity)
        self.write_lidar_result_to_csv(config.s_curve_name_constant_velocity)
        self.write_imu_result_to_csv(config.s_curve_name_constant_velocity)  

    def drive_s_curve_with_variable_velocity(self): 
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.imu_std, dt=config.dt)
        u = np.array([0,0], dtype=float)
        self.xs.append(model.get_initial_state(x=0, y=0, v=5, a=0, theta=0, delta=0))
        
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
            self.add_to_data(self.xs[i-1], u, i)
           
        
        self.create_lidar_measurement()
        # save the results
        self.save_point_cloud(config.s_curve_name_variable_velocity)
        self.write_lidar_result_to_csv(config.s_curve_name_variable_velocity)
        self.write_imu_result_to_csv(config.s_curve_name_variable_velocity)



    def save_point_cloud(self, type): 
        data = {
            'pc_x': self.pc_env_x,
            'pc_y': self.pc_env_y, 
            'reflect' : self.reflectivities
        }
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+type+config.point_cloud_appendix, data)

    def write_lidar_result_to_csv(self, type): 
        type = type+config.lidar_data_appendix
        basic_data = {
            'acceleration_input': self.car_ci_accelerations,
            'steering_input': self.car_ci_steerings,              
            'positions_x_ground_truth': self.car_gt_positions_x,
            'positions_y_ground_truth': self.car_gt_positions_y,
            'velocities_ground_truth': self.car_gt_velocities,
            'measurements_distances': self.measurement_distances_mode,
            'measurements_intensities': self.measurement_intensities_mode, 
            'timestamps': self.car_gt_timestamps
        }
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+type + config.data_suffix, basic_data)

    def write_imu_result_to_csv(self, type): 
        type = type+config.imu_data_appendix
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
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+type + config.data_suffix, data)
        self.create_map_image(config.image_and_image_data_prefix+type)

    def floating_numbers_to_whole_numbers(self, floating_number):
        return int(floating_number * 10**self.accounted_decimal_places)

    def create_map_image(self, name): 
        self.x_range = (
            self.floating_numbers_to_whole_numbers(np.array(self.car_gt_positions_x).min()) - self.floating_numbers_to_whole_numbers(config.map_border), 
            self.floating_numbers_to_whole_numbers(np.array(self.car_gt_positions_x).max()) + self.floating_numbers_to_whole_numbers(config.map_border)
        )
        self.y_range = (
            self.floating_numbers_to_whole_numbers(np.array(self.car_gt_positions_y).min()) - self.floating_numbers_to_whole_numbers(config.map_border),
            self.floating_numbers_to_whole_numbers(np.array(self.car_gt_positions_y).max()) + self.floating_numbers_to_whole_numbers(config.map_border)
        )
        self.map_shape = (
            self.y_range[1] - self.y_range[0]+self.floating_numbers_to_whole_numbers(1), # is the +1 really needed?
            self.x_range[1] - self.x_range[0]+self.floating_numbers_to_whole_numbers(1)
        )
        self.map = np.array(np.zeros(self.map_shape))  
        
        position_vectors = np.stack([self.car_gt_positions_x, self.car_gt_positions_y],axis=1)
        for pos in position_vectors: 

            x_in_image_coordinates = int(self.floating_numbers_to_whole_numbers(pos[0]))
            y_in_image_coordinates = int(self.floating_numbers_to_whole_numbers(pos[1]))
            
            index = [
                x_in_image_coordinates-self.x_range[0], 
                y_in_image_coordinates-self.y_range[0]
            ]
            self.position_vectors_in_image_coordinates.append(np.array([x_in_image_coordinates, y_in_image_coordinates]))
            self.map[index[1],index[0]] = 1
        self.position_vectors_in_image_coordinates = np.array(self.position_vectors_in_image_coordinates)
        # save image and transformationdata
        image_handler.save_array_as_image(self.map*255,config.paths['data_path']+name+config.image_suffix)
        trans_data = {
            "decimal_multiplier": 10**self.accounted_decimal_places,
            "x_min": self.x_range[0],
            "y_min": self.y_range[0],
            "x_max": self.x_range[1],
            "y_max": self.y_range[1],
        }
        json_handler.write_to_json(config.paths['data_path']+name+config.image_data_suffix, trans_data)
        self.create_distance_map(name)

    def create_distance_map(self,name):
        self.distance_map = cv.GaussianBlur(self.map,(5,5),2)

        for i in range(6):
            self.distance_map = cv.GaussianBlur(self.distance_map,(5,5),2)
        to_one = 1/self.distance_map.max()
        self.distance_map = self.distance_map * to_one
        image_handler.save_array_as_image(self.distance_map*255,config.paths['data_path']+name+config.distance_map_suffix)
    
    def reset_lists(self): 
        print("Resetting_lists")
        # control inputs
        self.car_ci_accelerations = []
        self.car_ci_steerings = []
        # ground truth
        self.car_gt_positions_x = []
        self.car_gt_positions_y = []
        self.car_gt_velocities = []
        self.car_gt_timestamps = []

        # imu measurements
        self.car_m_accelerations = []
        self.car_m_orientations = []

        #lidar measurements
        # point cloud environment
        self.pc_env_x = []
        self.pc_env_y = []
        self.reflectivities = []
        # point cloud measurement
        self.measurement_distances_mode = []
        self.measurement_intensities_mode = []

        # distance transform
        self.map_shape = (0,0)
        self.map = []
        self.distance_map = []
        self.x_range = np.array([0,0])
        self.y_range = np.array([0,0])
        self.accounted_decimal_places = 0
        self.position_vectors_in_image_coordinates = []
        
        self.xs = []
