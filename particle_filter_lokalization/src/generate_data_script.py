from pdb import main
import data_generation.local_data_generator_imu as ldg_imu
import data_generation.local_data_generator_lidar as ldg_lidar
import data_generation.local_data_generation_both as ldg
import argparse
import config.config as config
def main(): 
    argparser = argparse.ArgumentParser(
        description='Script for generating vehicle movement data')
    argparser.add_argument(
        '--raodtype','-rt',
        metavar='R',
        dest='road_type',
        default=config.straight_x_line_name,
        help='road type u wish to generate data for')
    argparser.add_argument(
        '--filtertype','-ft',
        metavar='F',
        dest='filter_type',
        default=config.straight_x_line_name,
        help='filter u want to generate data for')
    
    args = argparser.parse_args()

    if (args.filter_type == "imu"):
        data_generator = ldg_imu.LocalDataGeneratorIMU()
        data_generator.generate_data(args.road_type)
    elif(args.filter_type == "lidar"): 
        data_generator = ldg_lidar.LocalDataGeneratorLIDAR()
        data_generator.generate_data(args.road_type)

    elif(args.filter_type== "all"): 
        data_generator = ldg.LocalDataGenerator()
        data_generator.generate_data(args.road_type)
  


if __name__ == "__main__": 
    main()