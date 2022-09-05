from pdb import main
import data_generation.local_data_generator as ldg
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

    args = argparser.parse_args()

    data_generator = ldg.LocalDataGenerator()
    data_generator.generate_specific_data(args.road_type)
    #data_generator.drive_straight_in_x_direction()
    #data_generator.drive_a_long_curve()
    #data_generator.drive_s_curve_with_constant_velocity()


if __name__ == "__main__": 
    main()