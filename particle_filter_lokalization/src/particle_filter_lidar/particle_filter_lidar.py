from ctypes.wintypes import PCHAR
import numpy as np
import process_model.front_wheel_bycicle_model as fw_bycicle_model
import config.config as config
import data_generation.load_specific_data as load_specific_data
import copy
from filterpy.monte_carlo import systematic_resample
import utils.csv_handler as csv_handler
from scipy.spatial.distance import directed_hausdorff
import math
from scipy import stats


class ParticleFilterLIDAR: 
    def __init__(self, N,dataset_name):
        # particle filte related stuff
        self.N = N
      
        # process model related stuff
        self.process_model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.lidar_std, dt=config.dt)
        # data related stuff
        self.simulation_data = load_specific_data.load_simulation_data(dataset_name+config.lidar_data_appendix+config.data_suffix)
        #self.lidar_measurements = load_specific_data.load_lidar_measurements(dataset_name+config.lidar_data_appendix+config.point_cloud_measured_appendix)

        self.Ts=self.simulation_data['timestamps'].values
        self.point_cloud = load_specific_data.load_point_cloud(dataset_name+config.point_cloud_appendix)
        #self.particles = self.create_uniform_particles()
        self.particles = self.create_gaussian_particles(
            np.array([
                self.simulation_data['positions_x_ground_truth'][0], 
                self.simulation_data['positions_y_ground_truth'][0],
                self.simulation_data['velocities_ground_truth'][0], 
                self.simulation_data['acceleration_input'][0],
                np.random.uniform(0, np.pi*2, 1)[0],
                self.simulation_data['steering_input'][0]
            ]), 
            np.array([config.initial_pos_radius, config.initial_pos_radius, config.imu_sensor_std[0], config.imu_sensor_std[1], config.imu_sensor_std[1], np.deg2rad(70)])
        )
        self.weights = np.full((self.particles.shape[0],), 1/self.particles.shape[0], dtype=float)

        self.particles_at_t = []
        self.weights_at_t = []
        self.ground_truth = np.stack([self.simulation_data['positions_x_ground_truth'], self.simulation_data['positions_y_ground_truth']], axis=1)
        self.xs = []
        self.mse = 0
        self.mse_db = 0

        #result related
        self.dataset_name = dataset_name

    def predict(self, u): 
        for i in range(self.N): 
            self.particles[i] = self.process_model.F(x=self.particles[i], u=u)
        
    
    def create_gaussian_particles(self,mean, std):
        particles = np.empty((self.N, 6))
        particles[:, 0] = mean[0] + (np.random.rand(self.N) * std[0])
        particles[:, 1] = mean[1] + (np.random.rand(self.N) * std[1])
        particles[:, 2] = mean[2] + (np.random.rand(self.N) * std[2])
        particles[:, 3] = mean[3] + (np.random.rand(self.N) * std[3])
        particles[:, 4] = mean[4] + (np.random.rand(self.N) * std[0])
        particles[:, 5] = mean[5] + (np.random.rand(self.N) * std[0])
        particles[:, 4] %= 2 * np.pi
        particles[:, 5] %= 2 * np.pi
        return particles

    
    def update(self, z, R):
        # find the lidar pointcloud for each particle than calculate its distances to the measurement

        weights = []
        

        for p in self.particles[:,:2]: 
            # preparing particle measurements
            p_subs = p - self.point_cloud
            p_dists = np.linalg.norm(p_subs, axis=1)
            p_in_range = p_dists[p_dists < config.lidar_range]
            if (len(p_in_range > 0)):
                p_mode = stats.mode(p_in_range)[0][0]
                weight = stats.norm(p_mode, config.lidar_sensor_std).pdf(z)
                weights.append(weight)
            else: 
                weights.append(0)
        '''
        distances = []
        for p in self.particles[:,:2]: 
            diff = p-self.point_cloud
            pc_in_range = self.point_cloud[np.linalg.norm(diff, axis=1) < config.lidar_range]
            
            if (len(pc_in_range) != len(z) and len(pc_in_range) == 0 or len(z) == 0): 
                distance = 100
            elif (len(pc_in_range) == 0 and len(z) == 0): 
                distance = 1
            else: 
                distance = directed_hausdorff(z, pc_in_range)[0]
            if (distance == math.inf): 
                print(z, pc_in_range)
            distances.append(distance)
        distances = np.array(distances, dtype=float)

        distances = distances.max() - distances
        '''

        self.weights = self.weights * weights
        self.weights += 1.e-300       
        self.weights /= sum(self.weights) # normalize
        #return weights


    '''
    Estimate creates an estimation of all particles
    '''
    def estimate(self): 
        pos = self.particles[:,0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var  = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var

    '''
    Resample function resamples all particles
    '''
    def resample_from_index(self,indexes):
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))
        self.weights =  np.array(self.weights, dtype=float)
        return self.weights, self.particles
    '''
    Measures the number of particles contributing to the propability distribution
    '''
    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    '''
    main function running the pf
    '''
    def run_pf_lidar(self):
    
        zs = self.simulation_data['measurements'].values
        us = np.stack([self.simulation_data['acceleration_input'], self.simulation_data['steering_input']], axis=1)
        
        
        resample_counter = 0
        
        for i in range(len(self.Ts)): 
            self.particles_at_t.append(copy.copy(self.particles))
            self.weights_at_t.append(copy.copy(self.weights))
            self.predict(u=us[i])

            
            self.update(z=zs[i], R=config.lidar_sensor_std)
            if (self.neff() < self.N/config.lidar_neff_threshold): 

                resample_counter += 1
                indexes = systematic_resample(self.weights)
                self.weights, self.particles = self.resample_from_index(indexes)
                assert np.allclose(self.weights, 1/self.N)

            if (i % 100 == 0): 
                print(i, " iterations done, ", resample_counter, " resamples")
                resample_counter = 0
            mu, var = self.estimate()
            self.xs.append(mu)
        self.xs = np.array(self.xs, dtype=float)
        self.particles_at_t = np.array(self.particles_at_t, dtype=float)
        self.weights_at_t = np.array(self.weights_at_t, dtype=float)
        
        # save data for plotting and evaluation

        #save_particle_filter_data(particles_at_t, weights_at_t, xs, ground_truth_at_t, Ts)
    def save_result(self):
        data = {
            'gt_x': self.ground_truth[:,0], 
            'gt_y': self.ground_truth[:,1], 
            'xs_x': self.xs[:,0], 
            'xs_y': self.xs[:,1],
            'Ts': self.Ts
        }
        csv_handler.write_structured_data_to_csv(config.paths['filter_results_path']+self.dataset_name+config.lidar_data_appendix, data)
    
    def evaluate(self): 
        rx = self.xs[:,0] - self.ground_truth[:,0]
        ry = self.xs[:,1] - self.ground_truth[:,1]
        self.mse = (rx**2+ry**2).mean()
        self.mse_db = np.log10(self.mse)*10
