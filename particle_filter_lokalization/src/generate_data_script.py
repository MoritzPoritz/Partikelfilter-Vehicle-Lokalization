from pdb import main
import data_generation.local_data_generator_imu as ldg_imu
import data_generation.local_data_generator_lidar as ldg_lidar
import data_generation.local_data_generation_both as ldg
import argparse
import config.config as config
import numpy as np
def main(): 
    argparser = argparse.ArgumentParser(
        description='Script for generating vehicle movement data')
    argparser.add_argument(
        '--raodtype','-rt',
        metavar='R',
        dest='road_type',
        default=config.straight_x_constant_velocity_line_name,
        help='road type u wish to generate data for')
    argparser.add_argument(
        '--filtertype','-ft',
        metavar='F',
        dest='filter_type',
        default=config.straight_x_constant_velocity_line_name,
        help='filter u want to generate data for')
    
    args = argparser.parse_args()

    if (args.filter_type == "imu"):
        data_generator = ldg_imu.LocalDataGeneratorIMU()
        data_generator.generate_data(args.road_type)
    elif(args.filter_type == "lidar"): 
        data_generator = ldg_lidar.LocalDataGeneratorLIDAR()
        data_generator.generate_data(args.road_type)

    elif(args.filter_type== "all"): 
        for i in range(config.sample_size):
            data_generator = ldg.LocalDataGenerator()
            data_generator.initial_velocity = np.random.uniform(config.v_range[0],config.v_range[1] )
            data_generator.initial_acceleration = np.random.uniform(config.a_range[0],config.a_range[1] )
            data_generator.initial_theta = np.random.uniform(config.theta_range[0],config.theta_range[1] )
            data_generator.initial_delta = np.random.uniform(config.delta_range[0],config.delta_range[1] )
            data_generator.sample_id = i
            road_type = np.random.choice(config.all_road_types)
            for rain_rate in config.rain_rates:
                data_generator.rain_rate = rain_rate
                data_generator.generate_data(road_type)
                data_generator.reset_lists()
    


if __name__ == "__main__": 
    main()