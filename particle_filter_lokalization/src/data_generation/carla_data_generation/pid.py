from __future__ import print_function
import glob
import os
import numpy as np
import sys
import cv2 as cv

try:
    sys.path.append(glob.glob('..\\carla\\dist\\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname('..\\src'), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import config.config as config
import utils.csv_handler as csv_handler
import utils.image_handler as image_handler
import utils.json_handler as json_handler
from scipy import stats

import keyboard


class PIDTester: 
    def __init__(self,client, args):
        # carla related stuff
        
        self.client = client
        self.args = args
        self.world = self.client.get_world()
        self.set_settings()

        self.traffic_manager = self.client.get_trafficmanager()
        self.tm_port = self.traffic_manager.get_port()
        self.traffic_manager.set_synchronous_mode(True)

        self.car = None
        self.spectator = self.world.get_spectator()
        self.debug = self.world.debug

        # Stuff for creation of the map data
        self.positions = []
        self.x_range = None
        self.y_range = None

       

        # carla stuff
        self.actor_list = []
        self.spawn_point_index = 10
        self.blueprint_library = self.world.get_blueprint_library()

    def tick(self): 
        self.world.tick() 

    def set_settings(self): 
        print("Set settings")
        self.settings = self.world.get_settings()
        if not self.settings.synchronous_mode:
            self.settings.synchronous_mode = True
        print("Synchronous mode on: ", self.settings.synchronous_mode)
        # fixed_delta_seconds need  <= max_substep_delta_time * max_substeps
        self.settings.fixed_delta_seconds = .01
        self.settings.max_substep_delta_time = 0.01
        self.settings.max_substeps = 10
        self.world.apply_settings(self.settings)


    def spawn_vehicle(self): 
        blueprint_library = self.world.get_blueprint_library()
        bp_vehicle = blueprint_library.filter('mustang')[0]
        transform = self.world.get_map().get_spawn_points()[self.spawn_point_index]    
        self.car = self.world.spawn_actor(bp_vehicle, transform)
        #self.spectator.set_transform(self.car.get_transform())
        self.actor_list.append(self.car)
        print('created %s' % self.car.type_id) 
        self.car.set_autopilot(True, self.tm_port)
        print("Sat Trafficmanager to synchronous")
        
    def destroy(self):
  
        if self.car is not None:
            self.car.destroy()

      
    def run(self): 
        print("Started data retrievement")
        self.spawn_vehicle()
        print("spawned vehicle")
        self.add_imu_sensor()
        #self.add_lidar_sensor()
        while True: 
            self.tick()
            if keyboard.is_pressed('q'): 
                print("user stopped the loop")
                break
        
        print("End sensor retrievment")

        self.destroy()
        self.write_imu_data_to_csv()
        self.write_lidar_data_to_csv()
        #self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                
        
        self.traffic_manager.set_synchronous_mode(False)
        self.settings.synchronous_mode = False
        print('done.')

    def write_lidar_data_to_csv(self): 
        print("Write lidar data")
        data = {
            'acceleration_input': self.acceleration_input_lidar,
            'steering_input': self.steering_input_lidar, 
            'measurements': self.lidar_measurements, 
            'positions_x_ground_truth': self.positions_x_ground_truth_lidar,
            'positions_y_ground_truth': self.positions_y_ground_truth_lidar,
            'velocities_ground_truth': self.velocities_ground_truth_lidar, 
            'timestamps': self.timestamps_lidar
        }
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+ config.overall_data_file + config.lidar_data_appendix + config.data_suffix, data)
        print("Wrote LIDAR data to: ", config.paths['data_path']+ config.overall_data_file + config.lidar_data_appendix + config.data_suffix)

    def write_imu_data_to_csv(self): 
        print("Write imu data")
        
        data = {
            'acceleration_input': self.acceleration_input_imu,
            'steering_input': self.steering_input_imu, 
            'acceleration_measurement': self.acceleration_measurement, 
            'orientation_measurement': self.orientation_measurement, 
            'positions_x_ground_truth': self.positions_x_ground_truth_imu,
            'positions_y_ground_truth': self.positions_y_ground_truth_imu,
            'velocities_ground_truth': self.velocities_ground_truth_imu, 
            'timestamps': self.timestamps_imu
        }
        csv_handler.write_structured_data_to_csv(config.paths['data_path']+ config.overall_data_file + config.imu_data_appendix + config.data_suffix, data)
        print("Wrote IMU data to: ", config.paths['data_path']+ config.overall_data_file + config.imu_data_appendix + config.data_suffix)