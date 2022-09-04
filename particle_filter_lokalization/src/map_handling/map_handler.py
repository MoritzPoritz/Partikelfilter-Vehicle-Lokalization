import config.config as config
import utils.json_handler as json_handler
import utils.image_handler as image_handler
import numpy as np
class DistanceMap: 
    def __init__(self, map_name): 
        self.map_name = map_name
        self.image_data = json_handler.json_to_dict(config.image_and_image_data_prefix+map_name+config.image_data_suffix)
        self.distance_map = np.array(image_handler.image_to_array(config.image_and_image_data_prefix+map_name+config.distance_map_suffix))

    def world_coordinates_to_image(self, coordinates):
        x_image = int(coordinates[0] * self.image_data['decimal_multiplier'] - self.image_data['x_min'])
        y_image = int(coordinates[1] * self.image_data['decimal_multiplier'] - self.image_data['y_min'])

        return np.array([x_image, y_image])

    def image_to_world_coordinates(self, coordinates): 
        x_world = (coordinates[0] + self.image_data['x_min']) / self.image_data['decimal_multiplier']
        y_world = (coordinates[1] + self.image_data['y_min']) / self.image_data['decimal_multiplier']
        return np.array([x_world, y_world])

    def get_distance_from_worlds(self, coordinates):
        distance = self.distance_map[coordinates[1], coordinates[0]]
        return distance

