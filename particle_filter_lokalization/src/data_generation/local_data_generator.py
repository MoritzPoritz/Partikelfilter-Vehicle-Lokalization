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
        self.accounted_decimal_places = 0
        self.position_vectors_in_image_coordinates = []


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
        self.write_result_to_csv(config.straight_x_line_name)


    
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
        csv_handler.write_to_csv(type + config.data_suffix, data)
        #save_image.save_array_as_image(np.stack([self.car_gt_positions_x, self.car_gt_positions_y],axis=1),"map_image_straight_line" )
        self.create_map_image(config.image_and_image_data_prefix+type)


    def floating_numbers_to_whole_numbers(self, floating_number):
        return int(floating_number * 10**self.accounted_decimal_places)

    def create_map_image(self, name): 
        self.x_range = (
            self.floating_numbers_to_whole_numbers(np.array(self.car_gt_positions_x).min()),
            self.floating_numbers_to_whole_numbers(np.array(self.car_gt_positions_x).max())
        )
        self.y_range = (
            
            self.floating_numbers_to_whole_numbers(np.array(self.car_gt_positions_y).min()),
            self.floating_numbers_to_whole_numbers(np.array(self.car_gt_positions_y).max())
        )

        if (abs(self.y_range[1]-self.y_range[0])< 200):
            self.y_range = (-100,100)   
        
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
        image_handler.save_array_as_image(self.map*255,name+config.image_suffix)
        trans_data = {
            "decimal_multiplier": 10**self.accounted_decimal_places,
            "x_min": self.x_range[0],
            "y_min": self.y_range[0],
            "x_max": self.x_range[1],
            "y_max": self.y_range[1],
        }
        json_handler.write_to_json(name+config.image_data_suffix, trans_data)
        self.create_distance_map(name)

    def create_distance_map(self,name):
        self.distance_map = cv.GaussianBlur(self.map,(5,5),2)
        self.distance_map = cv.GaussianBlur(self.distance_map,(5,5),2)
        self.distance_map = cv.GaussianBlur(self.distance_map,(5,5),2)
        to_one = 1/self.distance_map.max()
        self.distance_map = self.distance_map * to_one
        image_handler.save_array_as_image(self.distance_map*255,name+config.distance_map_suffix)
        '''
        
        inverted_map = 1-self.map
        print(type(self.map))
        self.distance_map = cv.distanceTransform(inverted_map, cv.DIST_L2, 3, cv.CV_8U)
        cv.normalize(self.distance_map, self.distance_map, 0, 1.0, cv.NORM_MINMAX)
        self.distance_map = 1- self.distance_map
        '''
