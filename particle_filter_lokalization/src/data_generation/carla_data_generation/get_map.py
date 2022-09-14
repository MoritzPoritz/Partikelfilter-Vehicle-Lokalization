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

import keyboard


class MapRetriever: 
    def __init__(self,client, args):
        # carla related stuff
        
        self.client = client
        self.args = args
        self.world = self.client.get_world()
        self.car = None
        self.spectator = self.world.get_spectator()
        self.debug = self.world.debug

        # Stuff for creation of the map data
        self.positions = []
        self.x_range = None
        self.y_range = None

        self.map_shape = ()
        self.map = []

        self.distance_map = []

        self.position_vectors_in_image_coordinates = []

        self.accounted_decimal_places = 0
    def tick(self): 
        self.world.tick() 
        if (self.car != None): 
            current_location = self.car.get_location() 
            self.positions.append(np.array([current_location.x, current_location.y]))

    def map_retrievement(self): 

        actor_list = []
    
        spawn_point_index = 124
        # set world settings, synchronous mode and delta seconds
        settings = self.world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
        # fixed_delta_seconds need  <= max_substep_delta_time * max_substeps
        settings.fixed_delta_seconds = .05
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        print("Settings", settings)
        self.world.apply_settings(settings)


        # setup vehicle 
        blueprint_library = self.world.get_blueprint_library()

        bp_vehicle = blueprint_library.filter('mustang')[0]
        transform = self.world.get_map().get_spawn_points()[spawn_point_index]
        spawn_points = self.world.get_map().get_spawn_points()
        # draw reference points
        self.debug.draw_point(spawn_points[spawn_point_index].location, 1, carla.Color(255, 0,0), 0)
        self.debug.draw_string(spawn_points[spawn_point_index].location, str(spawn_points[spawn_point_index].location),True, carla.Color(255,0,0), 99999999999999990)
        
        self.debug.draw_point(spawn_points[0].location, 1, carla.Color(255, 0,0), 0)
        self.debug.draw_string(spawn_points[0].location, str(spawn_points[0].location),True, carla.Color(255,0,0), 99999999999999990)
        
        self.debug.draw_point(spawn_points[50].location, 1, carla.Color(255, 0,0), 0)
        self.debug.draw_string(spawn_points[50].location, str(spawn_points[50].location),True, carla.Color(255,0,0), 99999999999999990)
        
        self.debug.draw_point(spawn_points[100].location, 1, carla.Color(255, 0,0), 0)
        self.debug.draw_string(spawn_points[100].location, str(spawn_points[100].location),True, carla.Color(255,0,0), 99999999999999990)

        #for i in range(len(spawn_points)): 
            #self.debug.draw_box(carla.BoundingBox(spawn_points[i].location,carla.Vector3D(1,2,2)),spawn_points[i].rotation, 0.05, carla.Color(255,0,0,0),0)
        #    self.debug.draw_string(spawn_points[i].location, "Index: " + str(i),True, carla.Color(255,0,0), 99999999999999990)
        self.car = self.world.spawn_actor(bp_vehicle, transform)
        #self.spectator.set_transform(self.car.get_transform())
        actor_list.append(self.car)
        print('created %s' % self.car.type_id)
        
        tm = self.client.get_trafficmanager()
        tm_port = tm.get_port()
        tm.set_synchronous_mode(True)
        self.car.set_autopilot(True, tm_port)
        

        while True: 
            self.tick()
            
            if keyboard.is_pressed('q'): 
                print("user stopped the loop")
                break
        
        self.write_results_to_csv()
        self.create_map_image('carla')
        print("End sensor retrievment")
        #lidar_bp.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

    def floating_numbers_to_whole_numbers(self, floating_number):
        return int(floating_number * 10**self.accounted_decimal_places)

    def create_map_image(self, name): 
        self.x_range = (
            self.floating_numbers_to_whole_numbers(np.array(self.positions[:,0]).min()) - self.floating_numbers_to_whole_numbers(config.map_border), 
            self.floating_numbers_to_whole_numbers(np.array(self.positions[:,0]).max()) + self.floating_numbers_to_whole_numbers(config.map_border)
        )
        self.y_range = (
            self.floating_numbers_to_whole_numbers(np.array(self.positions[:,1]).min()) - self.floating_numbers_to_whole_numbers(config.map_border),
            self.floating_numbers_to_whole_numbers(np.array(self.positions[:,1]).max()) + self.floating_numbers_to_whole_numbers(config.map_border)
        )

       
        
        self.map_shape = (
            self.y_range[1] - self.y_range[0]+self.floating_numbers_to_whole_numbers(1), # is the +1 really needed?
            self.x_range[1] - self.x_range[0]+self.floating_numbers_to_whole_numbers(1)
        )
        self.map = np.array(np.zeros(self.map_shape))  
        
        for pos in self.positions: 

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
        image_handler.save_array_as_image(self.map*255,config.paths['map_path']+name+config.image_suffix)
        trans_data = {
            "decimal_multiplier": 10**self.accounted_decimal_places,
            "x_min": self.x_range[0],
            "y_min": self.y_range[0],
            "x_max": self.x_range[1],
            "y_max": self.y_range[1],
        }
        json_handler.write_to_json(config.paths['map_path']+name+config.image_data_suffix, trans_data)
        self.create_distance_map(name)

    def create_distance_map(self,name):
        self.distance_map = cv.GaussianBlur(self.map,(5,5),2)

        for i in range(6):
            self.distance_map = cv.GaussianBlur(self.distance_map,(5,5),2)
        to_one = 1/self.distance_map.max()
        self.distance_map = self.distance_map * to_one
        image_handler.save_array_as_image(self.distance_map*255,config.paths['map_path']+name+config.distance_map_suffix)

    def write_results_to_csv(self): 

        self.positions = np.array(self.positions)

        data = {
            'point_x': self.positions[:,0],
            'point_y': self.positions[:,1]
        }
        csv_handler.write_structured_data_to_csv(config.paths['map_path']+'carla_map' + config.map_data_appendix, data)
