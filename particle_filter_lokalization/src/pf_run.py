import particle_filter_imu.particle_filter_imu as pf_imu
import particle_filter_lidar.particle_filter_lidar as pf_lidar
import config.config as config
import plotting.plot_animated_filter as animated
import argparse
import os
import shutil

import time

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
    argparser.add_argument(
        '--specific_dataset','-sd',
        metavar='F',
        dest='specific_dataset',
        default="sample_id_0__straight_in_x_constant_velocity__rain_rate_0",
        help='data u want to the filter to run on')
    
    args = argparser.parse_args()

   

    if (args.filter_type == "imu"):
        if (args.road_type == "specific"):
            pf = pf_imu.ParticleFilterIMU(config.N, args.specific_dataset)
            start = time.time()
            pf.run_pf_imu()
            print("IMU-particle-filter took: ", time.time() - start, " seconds")
            pf.evaluate()
            pf.save_result()
            animated.plot_results_animated_imu(pf.particles_at_t, pf.weights_at_t, pf.xs, pf.ground_truth, pf.dm, pf.Ts, pf.mse, pf.mse_db, pf.rmse)
    elif (args.filter_type == "lidar"):
        print("Run LIDAR-FIlter")
        if (args.road_type == "specific"):

            pf = pf_lidar.ParticleFilterLIDAR(config.N,  args.specific_dataset)
            start = time.time()
            pf.run_pf_lidar()
            print("LIDAR-particle-filter took: ", time.time() - start, " seconds")
            pf.evaluate()
            pf.save_result()
            animated.plot_results_animated_lidar(pf.particles_at_t, pf.weights_at_t, pf.xs, pf.ground_truth,pf.dm, pf.Ts, pf.mse, pf.mse_db, pf.rmse, pf.point_cloud)
    
    elif (args.filter_type == "all" and args.road_type == "all"): 
        all_data = os.listdir(config.paths["data_path"])
        for file in all_data: 
            if (file.endswith(".csv")): 
                if ("imu__data" in file):
                    file_name_array = file.split("__")
                    sample_id = file_name_array[0]
                    road_type_name = file_name_array[1]
                    rain_rate = file_name_array[2]
                    pf = pf_imu.ParticleFilterIMU(config.N, sample_id+"__"+road_type_name+"__"+rain_rate)
                    pf.run_pf_imu()
                    pf.evaluate()
                    pf.save_result()
                    original = config.paths['data_path']+file
                    to = config.paths['already_filtered']+file
                    print("Finished imu filter on " + sample_id+"__"+road_type_name+"__"+rain_rate)
                    print("Moved ", original, " to ", to)
                    shutil.move(original, to)

                elif ("lidar__data" in file):
                    file_name_array = file.split("__")
                    sample_id = file_name_array[0]
                    road_type_name = file_name_array[1]
                    rain_rate = file_name_array[2]
                    pf = pf_lidar.ParticleFilterLIDAR(config.N, sample_id+"__"+road_type_name+"__"+rain_rate)
                    pf.run_pf_lidar()
                    pf.evaluate()
                    pf.save_result()
                    
                    original = config.paths['data_path']+file
                    to = config.paths['already_filtered']+file
                    print("Finished lidar filter on " + sample_id+"__"+road_type_name+"__"+rain_rate)
                    print("Moved ", original, " to ", to)
                    shutil.move(original, to)
    
    '''
        if(args.road_type == 'all'):   
            all_data = os.listdir(config.paths["data_path"])
            for file in all_data: 
                if (file.endswith(".csv")): 
                    if ("imu__data" in file):
                        road_type_name = file.split("__")[0]
                        pf = pf_imu.ParticleFilterIMU(config.N, road_type_name)
                        pf.run_pf_imu()
                        pf.evaluate()
                        pf.save_result()
        else:
            if (args.road_type == config.straight_x_constant_velocity_line_name): 
                pf = pf_imu.ParticleFilterIMU(config.N, config.straight_x_constant_velocity_line_name)
            elif(args.road_type == config.curve_line_name): 
                pf = pf_imu.ParticleFilterIMU(config.N, config.curve_line_name)        
            elif(args.road_type == config.s_curve_name_constant_velocity): 
                pf = pf_imu.ParticleFilterIMU(config.N, config.s_curve_name_constant_velocity)    
            elif(args.road_type == config.s_curve_name_variable_velocity): 
                pf = pf_imu.ParticleFilterIMU(config.N, config.s_curve_name_variable_velocity)        
                

            pf.run_pf_imu()
            pf.evaluate()
            pf.save_result()
            animated.plot_results_animated_imu(pf.particles_at_t, pf.weights_at_t, pf.xs, pf.ground_truth,pf.dm, pf.Ts, pf.mse, pf.mse_db)
    elif (args.filter_type == "lidar"):
        if(args.road_type == 'all'): 
            
            all_data = os.listdir(config.paths['data_path'])
            for file in all_data: 
                if (file.endswith(".csv")): 
                    if ("lidar__data" in file):
                        road_type_name = file.split("__")[0]
                        pf = pf_lidar.ParticleFilterLIDAR(config.N, road_type_name)
                        pf.run_pf_lidar()
                        pf.evaluate()
                        pf.save_result()
        else:
            if (args.road_type == config.straight_x_constant_velocity_line_name): 
                pf = pf_lidar.ParticleFilterLIDAR(config.N, config.straight_x_constant_velocity_line_name)
            elif(args.road_type == config.curve_line_name): 
                pf = pf_lidar.ParticleFilterLIDAR(config.N, config.curve_line_name)        
            elif(args.road_type == config.s_curve_name_constant_velocity): 
                pf = pf_lidar.ParticleFilterLIDAR(config.N, config.s_curve_name_constant_velocity)    
            elif(args.road_type == config.s_curve_name_variable_velocity): 
                pf = pf_lidar.ParticleFilterLIDAR(config.N, config.s_curve_name_variable_velocity)        
                

            pf.run_pf_lidar()
            pf.evaluate()
            pf.save_result()
            animated.plot_results_animated_lidar(pf.particles_at_t, pf.weights_at_t, pf.xs, pf.ground_truth, pf.Ts, pf.mse, pf.mse_db, pf.point_cloud)
    else: 

        sample_id_0__straight_in_x_constant_velocity__rain_rate_0__imu__data
        sample_id_0__straight_in_x_variable_velocity__rain_rate_0__imu__data
    '''

if __name__ == "__main__": 
    main()