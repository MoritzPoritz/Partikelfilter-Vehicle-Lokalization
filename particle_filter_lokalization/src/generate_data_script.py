from pdb import main
import data_generation.local_data_generator as ldg


def main(): 
    data_generator = ldg.LocalDataGenerator()
    #data_generator.drive_straight_in_x_direction()
    data_generator.drive_a_long_curve()



if __name__ == "__main__": 
    main()