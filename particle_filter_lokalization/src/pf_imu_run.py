import particle_filter_imu.particle_filter_imu as pf_imu
import config.config as config
import plotting.plot_animated_filter as animated
import argparse
import os
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
    if(args.road_type == 'all'): 
        filter_results_path = os.path.abspath(os.path.join(
                  os.path.dirname('filter_results'), 
                  os.pardir)
        ) + "\\data\\"
        all_data = os.listdir(filter_results_path)
        for file in all_data: 
            
            if (file.endswith(".csv")): 
                
                road_type_name = file.split("__")[0]
                pf = pf_imu.ParticleFilterIMU(config.N, road_type_name)
                pf.run_pf_imu()
                pf.evaluate()
                pf.save_result()

    else:
        if (args.road_type == config.straight_x_line_name): 
            pf = pf_imu.ParticleFilterIMU(config.N, config.straight_x_line_name)
        elif(args.road_type == config.curve_line_name): 
            pf = pf_imu.ParticleFilterIMU(config.N, config.curve_line_name)        
        elif(args.road_type == config.s_curve_name_constant_velocity): 
            pf = pf_imu.ParticleFilterIMU(config.N, config.s_curve_name_constant_velocity)    
        elif(args.road_type == config.s_curve_name_variable_velocity): 
            pf = pf_imu.ParticleFilterIMU(config.N, config.s_curve_name_variable_velocity)        
            

        pf.run_pf_imu()
        pf.evaluate()
        pf.save_result()
        animated.plot_results_animated(pf.particles_at_t, pf.weights_at_t, pf.xs, pf.ground_truth,pf.dm, pf.Ts, pf.mse, pf.mse_db)


if __name__ == "__main__": 
    main()