import config.config as config
import utils.json_handler as json_handler
import utils.image_handler as image_handler
import numpy as np
class DistanceMap: 
    def __init__(self): 
        self.map = np.array(image_handler.image_to_array(config.paths['map_path']+config.carla_map_name))
        self.distance_map = np.array(image_handler.image_to_array(config.paths['map_path']+config.carla_distance_map_name))
        self.prepare_distance_map()
        self.scale = 0.63
        self.translate_x = 149
        self.translate_y = 61

    def prepare_distance_map(self): 
        self.distance_map = self.distance_map/255
        to_one = 1/self.distance_map.max()
        self.distance_map = self.distance_map * to_one


    def world_coordinates_to_image(self,world_point):
        
        return (world_point*self.scale + np.array([self.translate_x, self.translate_y])).astype(int)   
    
    #def world_coordinates_to_image(self, coordinates):
    #    x_image = int(coordinates[0] * self.image_data['decimal_multiplier'] - self.image_data['x_min'])
    #    y_image = int(coordinates[1] * self.image_data['decimal_multiplier'] - self.image_data['y_min'])

        return np.array([x_image, y_image])

    def image_to_world_coordinates(self, coordinates): 
        x_world = (coordinates[0] + self.image_data['x_min']) / self.image_data['decimal_multiplier']
        y_world = (coordinates[1] + self.image_data['y_min']) / self.image_data['decimal_multiplier']
        return np.array([x_world, y_world])

    def get_distance_from_worlds(self, coordinates):
        distance = self.distance_map[coordinates[1], coordinates[0]]
        return distance

