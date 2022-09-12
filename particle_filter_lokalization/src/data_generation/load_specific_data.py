import pandas as pd
import numpy as np
import utils.csv_handler as csv_handler
import config.config as config
def load_simulation_data(name):
    path = config.paths['data_path']+name 
    return csv_handler.load_csv(path)


def load_point_cloud(name): 
    path = config.paths['data_path']+name 
    point_cloud_df = csv_handler.load_csv(path)
    return np.stack([point_cloud_df['pc_x'],point_cloud_df['pc_y']], axis=1)

def load_lidar_measurements(name): 
    path = config.paths['pc_measurements_path']+name 
    return csv_handler.load_csv(path)
    
    
