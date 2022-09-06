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
class LocalDataGeneratorIMU: 
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
        self.accounted_decimal_places = 0
        self.position_vectors_in_image_coordinates = []
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

    def drive_a_long_curve(self): 
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.std, dt=config.dt)
        self.xs.append(model.get_initial_state(x=1, y=1, v=5, a=0, theta=0, delta=0))

        u = np.array([0,0], dtype=float)
        
        for i in range(1,1000): 
            u[1] = u[1] + 0.0001
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
            self.car_m_accelerations.append(self.xs[i-1][3]+config.sensor_std[0]*np.random.randn()) 
            self.car_m_orientations.append(self.xs[i-1][4]+config.sensor_std[1]*np.random.randn())   
            
        self.write_result_to_csv(config.curve_line_name)

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
            self.car_m_accelerations.append(self.xs[i-1][3]+config.sensor_std[0]*np.random.randn()) 
            self.car_m_orientations.append(self.xs[i-1][4]+config.sensor_std[1]*np.random.randn())   
            
        self.write_result_to_csv(config.straight_x_line_name)

    def drive_s_curve_with_constant_velocity(self): 
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.std, dt=config.dt)
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
            self.car_ci_accelerations.append(u[0])
            self.car_ci_steerings.append(u[1])
            self.car_gt_positions_x.append(self.xs[i-1][0])
            self.car_gt_positions_y.append(self.xs[i-1][1])
            self.car_gt_velocities.append(self.xs[i-1][2])
            self.car_gt_timestamps.append(config.dt*i)
            self.car_m_accelerations.append(self.xs[i-1][3]+config.sensor_std[0]*np.random.randn()) 
            self.car_m_orientations.append(self.xs[i-1][4]+config.sensor_std[1]*np.random.randn())
        
        self.write_result_to_csv(config.s_curve_name_constant_velocity)

    def drive_s_curve_with_variable_velocity(self): 
        self.reset_lists()
        model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.std, dt=config.dt)
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
            self.car_ci_accelerations.append(u[0])
            self.car_ci_steerings.append(u[1])
            self.car_gt_positions_x.append(self.xs[i-1][0])
            self.car_gt_positions_y.append(self.xs[i-1][1])
            self.car_gt_velocities.append(self.xs[i-1][2])
            self.car_gt_timestamps.append(config.dt*i)
            self.car_m_accelerations.append(self.xs[i-1][3]+config.sensor_std[0]*np.random.randn()) 
            self.car_m_orientations.append(self.xs[i-1][4]+config.sensor_std[1]*np.random.randn())
        
        self.write_result_to_csv(config.s_curve_name_variable_velocity)

    def write_result_to_csv(self, type): 
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
        csv_handler.write_to_csv(config.paths['data_path']+type + config.data_suffix, data)
        #save_image.save_array_as_image(np.stack([self.car_gt_positions_x, self.car_gt_positions_y],axis=1),"map_image_straight_line" )
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
            print(index[1])
            self.map[index[1],index[0]] = 1
        self.position_vectors_in_image_coordinates = np.array(self.position_vectors_in_image_coordinates)
        # save image and transformationdata
        image_handler.save_array_as_image(config.paths['data_path']+self.map*255,name+config.image_suffix)
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
        image_handler.save_array_as_image(config.paths['data_path']+self.distance_map*255,name+config.distance_map_suffix)
    
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

        # distance transform
        self.map_shape = (0,0)
        self.map = []
        self.distance_map = []
        self.x_range = np.array([0,0])
        self.y_range = np.array([0,0])
        self.accounted_decimal_places = 0
        self.position_vectors_in_image_coordinates = []
        self.xs = []
