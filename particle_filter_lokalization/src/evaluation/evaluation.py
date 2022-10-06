import numpy as np
import utils.csv_handler as csv_handler
import matplotlib.pyplot as plt
import config.config as config
class ParticleFilterEvaluator: 
    def __init__(self, dataset_name): 
        self.dataset_name = dataset_name
        self.dataset = csv_handler.load_csv(dataset_name)

        self.mse = 0
        self.mse_db = 0
        self.se_over_time = 0
        self.rmse = 0
        


    def evaluate_filter_performance(self): 
        rx = self.dataset['gt_x'] - self.dataset['xs_x']
        ry = self.dataset['gt_y'] - self.dataset['xs_y']
        self.mse = (rx**2 + ry**2).mean()
        self.mse_db = np.log10(self.mse)*10
        self.rmse = np.sqrt(self.mse)

    def calculate_se_over_time(self): 
        rx = self.dataset['gt_x'] - self.dataset['xs_x']
        ry = self.dataset['gt_y'] - self.dataset['xs_y']
        self.se_over_time = rx**2 + ry**2

    def plot_se_over_time(self): 
        plt.plot(self.dataset['Ts'], self.se_over_time)
        plt.show()


