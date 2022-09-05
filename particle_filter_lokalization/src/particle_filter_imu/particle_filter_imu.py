import numpy as np
from scipy import stats
import process_model.front_wheel_bycicle_model as fw_bycicle_model
import config.config as config
import data_generation.load_specific_data as load_specific_data
import map_handling.map_handler as map_handler
import copy
from filterpy.monte_carlo import systematic_resample
import utils.csv_handler as csv_handler


class ParticleFilterIMU: 


    def __init__(self, N,dataset_name):
        # particle filte related stuff
        self.N = N
      
        # process model related stuff
        self.process_model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.std, dt=config.dt)
        # data related stuff
        self.simulation_data = load_specific_data.load_simulation_data(dataset_name+config.data_suffix)
        self.dm = map_handler.DistanceMap(dataset_name)
        self.Ts=self.simulation_data['timestamps'].values
        
        # create ranges
        x_min = self.dm.image_data['x_min']
        x_max = self.dm.image_data['x_max']
        
        y_min = self.dm.image_data['y_min']
        y_max = self.dm.image_data['y_max']
        
        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]
    
        self.v_range = [0, 10]
        self.a_range = [0, 10]

        self.theta_range = [0,2*np.pi]
        self.delta_range = [self.simulation_data['steering_input'].min(), self.simulation_data['steering_input'].max()]

        #self.particles = self.create_uniform_particles()
        self.particles = self.create_gaussian_particles(
            np.array([
                self.simulation_data['positions_x_ground_truth'][0], 
                self.simulation_data['positions_y_ground_truth'][0],
                self.simulation_data['velocities_ground_truth'][0], 
                self.simulation_data['acceleration_input'][0],
                self.simulation_data['orientation_measurement'][0],
                self.simulation_data['steering_input'][0]
            ]), 
            np.array([config.initial_pos_radius, config.initial_pos_radius, config.sensor_std[0], config.sensor_std[1], config.sensor_std[1], np.deg2rad(70)])
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
        
    '''
    creates uniformly distributed particles
    '''
    def create_uniform_particles(self):
        particles = np.empty((self.N, 6))
        particles[:, 0] = np.random.uniform(self.x_range[0], self.x_range[1], size=self.N)
        particles[:, 1] = np.random.uniform(self.y_range[0], self.y_range[1], size=self.N)
        particles[:, 2] = np.random.uniform(self.v_range[0], self.v_range[1], size=self.N)
        particles[:, 3] = np.random.uniform(self.a_range[0], self.a_range[1], size=self.N)
        particles[:, 4] = np.random.uniform(self.theta_range[0], self.theta_range[1], size=self.N)
        particles[:, 5] = np.random.uniform(self.delta_range[0], self.delta_range[1], size=self.N)
        particles[:, 4] %= 2 * np.pi
        particles[:, 5] %= 2 * np.pi
        return particles
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
        # acceleration likelihood
        acceleration_likelihoods = stats.norm(self.particles[:,3], R[0]).pdf(z[0])
        # orientation likelihood --> Check if this should be in filter
        rotation_likelihoods = stats.norm(self.particles[:,4], R[2]).pdf(z[1])
        particles_image_coords = np.array(list(map(self.dm.world_coordinates_to_image, self.particles[:, 0:2])))

        distances = []
        for p in particles_image_coords: 
            if (p[0] < self.dm.distance_map.shape[1] and p[0] > 0 and p[1] < self.dm.distance_map.shape[0] and p[1] > 0): 
                value = self.dm.get_distance_from_worlds(p)
             
                distances.append(value)
            else:
                distances.append(0)
        distances = np.array(distances, dtype=object)
        average = np.array((acceleration_likelihoods + rotation_likelihoods + distances) / 3)
        #average = np.array((rotation_likelihoods + distances) / 2)

        self.weights = self.weights * average 
        
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
    def run_pf_imu(self):
    
        zs = np.stack([self.simulation_data['acceleration_measurement'], self.simulation_data['orientation_measurement']], axis=1)
        us = np.stack([self.simulation_data['acceleration_input'], self.simulation_data['steering_input']], axis=1)
        
        
        resample_counter = 0
        
        for i,u in enumerate(self.Ts): 
            self.particles_at_t.append(copy.copy(self.particles))
            self.weights_at_t.append(copy.copy(self.weights))
            self.predict(u=us[i])
            self.update(z=zs[i], R=config.sensor_std)
        
            if (self.neff() < self.N/config.neff_threshold): 

                resample_counter += 1
                indexes = systematic_resample(self.weights)
                self.weights, particles = self.resample_from_index(indexes)
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
        csv_handler.write_to_csv('filter_results/'+self.dataset_name, data)
    
    def evaluate(self): 
        rx = self.xs[:,0] - self.ground_truth[:,0]
        ry = self.xs[:,1] - self.ground_truth[:,1]
        self.mse = (rx**2+ry**2).mean()
        self.mse_db = np.log10(self.mse)*10
