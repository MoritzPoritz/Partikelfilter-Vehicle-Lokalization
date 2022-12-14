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
import map_handling.map_handler_imu as map_handler



class ParticleFilterLIDAR: 
    def __init__(self, N,dataset_name):
        # particle filte related stuff
        self.N = N
      
        # process model related stuff
        self.process_model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.lidar_std, dt=config.dt)
        # data related stuff
        self.simulation_data = load_specific_data.load_simulation_data(dataset_name+config.lidar_data_appendix+config.data_suffix)
        #self.lidar_measurements = load_specific_data.load_lidar_measurements(dataset_name+config.lidar_data_appendix+config.point_cloud_measured_appendix)
        self.dm = map_handler.DistanceMap(dataset_name)
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

        lidar_distance_likelihoods = []
        lidar_intensity_likelihoods = []
        map_distances = []

        for p in self.particles[:,:2]: 
            # getting lidar measurement
            p_subs = p - self.point_cloud[:,:2]
            p_dists = np.linalg.norm(p_subs[:,:2], axis=1)
            p_intens = self.point_cloud[:,1][p_dists < config.lidar_range]
            p_in_range = p_dists[p_dists < config.lidar_range]
            if (len(p_in_range > 0)):
                p_mode_dist = stats.mode(p_in_range)[0][0]
                p_mode_int = stats.mode(p_intens)[0][0]
                lidar_distance_likelihood = stats.norm(p_mode_dist, config.lidar_sensor_std).pdf(z[0])
                lidar_distance_likelihoods.append(lidar_distance_likelihood)
                lidar_intensity_likelihood = stats.norm(p_mode_int, config.lidar_sensor_std).pdf(z[1])
                lidar_intensity_likelihoods.append(lidar_intensity_likelihood)
                
            else: 
                lidar_distance_likelihoods.append(0)
                lidar_intensity_likelihoods.append(0)

            image_point = self.dm.world_coordinates_to_image(p)
            if (image_point[0] < self.dm.distance_map.shape[1] and image_point[0] > 0 and image_point[1] < self.dm.distance_map.shape[0] and image_point[1] > 0): 
                value = self.dm.get_distance_from_worlds(image_point)
             
                map_distances.append(value)
            else:
                map_distances.append(0)


        lidar_distance_likelihoods = np.array(lidar_distance_likelihoods)
        lidar_intensity_likelihoods = np.array(lidar_intensity_likelihoods)
        map_distances = np.array(map_distances)
        average_likelihoods = np.array((map_distances + lidar_distance_likelihoods + lidar_intensity_likelihoods) / 3)
        self.weights = self.weights * average_likelihoods
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
    
        zs = np.stack([self.simulation_data['measurements_distances'].values, self.simulation_data['measurements_intensities'].values], axis=1)
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
