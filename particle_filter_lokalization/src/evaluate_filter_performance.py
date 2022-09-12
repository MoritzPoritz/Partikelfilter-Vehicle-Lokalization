import csv
import evaluation.evaluation as evaluation
import config.config as config
import argparse
import sys
import os
import numpy as np
import utils.csv_handler as csv_handler
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

    filter_results_path = os.path.abspath(os.path.join(
                  os.path.dirname('filter_results'), 
                  os.pardir)
    ) + "\\data\\filter_results\\"
     
    
    if (args.road_type == "all"):
        evaluation_results = []
        results = os.listdir(filter_results_path)
        for r in results: 
            name = r.split(".")[0]
            print(name)
            evaluator = evaluation.ParticleFilterEvaluator("\\filter_results\\" + name)
            evaluator.evaluate_filter_performance()
            evaluator.calculate_se_over_time()
            evaluation_results.append(np.array([name, evaluator.mse, evaluator.mse_db]))
        evaluation_results = np.array(evaluation_results)
        data = {
            'road': evaluation_results[:,0],
            'mse': evaluation_results[:,1],
            'mse_db': evaluation_results[:,2]
        }    
        csv_handler.write_structured_data_to_csv('evaluation_results\\results', data)
    else: 
        evaluator = evaluation.ParticleFilterEvaluator(args.road_type)
        evaluator.evaluate_filter_performance()
        evaluator.calculate_se_over_time()
        evaluator.plot_se_over_time()
        #evaluator.plot_se_over_time()
if __name__ == "__main__": 
    main()