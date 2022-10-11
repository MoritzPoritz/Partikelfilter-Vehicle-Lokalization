import csv
import evaluation.evaluation as evaluation
import config.config as config
import argparse
import sys
import os
import numpy as np
import utils.csv_handler as csv_handler
import datetime
def main(): 
    argparser = argparse.ArgumentParser(
        description='Script for generating vehicle movement data')
    argparser.add_argument(
        '--raodtype','-rt',
        metavar='R',
        dest='road_type',
        default=config.straight_x_constant_velocity_line_name,
        help='road type u wish to generate data for')
    args = argparser.parse_args()

 
    
    if (args.road_type == "all"):
        evaluation_results = []
        results = os.listdir(config.paths['filter_results_path'])
        samples = {}
        for r in results: 
            if r.endswith(".csv"):               
                file_name = r.split(".")[0]
                print(file_name)
                splitted_file_name = file_name.split("__")
                sample_id = splitted_file_name[0]
                road_type = splitted_file_name[1]
                rain_rate = splitted_file_name[2]
                filter_type = splitted_file_name[3]


                evaluator = evaluation.ParticleFilterEvaluator(config.paths['filter_results_path'] + file_name)
                evaluator.evaluate_filter_performance()
                evaluator.calculate_se_over_time()
                evaluation_results.append(np.array([filter_type, sample_id, rain_rate, road_type, evaluator.mse, evaluator.mse_db, evaluator.rmse]))
        evaluation_results = np.array(evaluation_results)
        data = {
            'filter': evaluation_results[:,0],
            'sample': evaluation_results[:,1],
            'rain': evaluation_results[:,2], 
            'road': evaluation_results[:,3],
            'mse': evaluation_results[:,4],
            'mse_db': evaluation_results[:,5],
            'rmse': evaluation_results[:,6]
        }    
        timestamp = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        csv_handler.write_structured_data_to_csv(config.paths['evaluation_results_path'] + 'results_'+timestamp, data)
    else: 
        evaluator = evaluation.ParticleFilterEvaluator(args.road_type)
        evaluator.evaluate_filter_performance()
        evaluator.calculate_se_over_time()
        evaluator.plot_se_over_time()
        #evaluator.plot_se_over_time()
if __name__ == "__main__": 
    main()