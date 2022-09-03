import numpy as np
from scipy import stats
import process_model.front_wheel_bycicle_model as fw_bycicle_model
import config
class ParticleFilterIMU: 


    def __init__(self, N, x_range, y_range, v_range, a_range, theta_range, delta_range):
        # particle filte related stuff
        self.N = N
        self.x_range = x_range
        self.y_range = y_range
        self.v_range = v_range
        self.a_range = a_range
        self.theta_range = theta_range
        self.delta_range = delta_range
        self.particles = self.create_uniform_particles()
        # process model related stuff
        self.process_model = fw_bycicle_model.FrontWheelBycicleModel(vehicle_length=config.L, control_input_std=config.std, dt=config.std)
        # data related stuff
        

    def predict(self): 
        '''
        Calls F for every particle
        '''
        for i in range(self.N): 
            self.particles[i] = self.process_model.F(x=self.particles[i], u=u, step=dt, L=L, std=std, N=len(particles))
        return particles
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

    def update(self, particles, weights,z, R, dm):
        # acceleration likelihood
        acceleration_likelihoods = stats.norm(particles[:,3], R[0]).pdf(z[0])
        # orientation likelihood --> Check if this should be in filter
        rotation_likelihoods = stats.norm(particles[:,4], R[2]).pdf(z[1])
        particles_image_coords = dm.coord_to_image(particles[:, 0:2])
        distances = []
        for p in particles_image_coords: 
            if (p[0] < dm.distance_map.shape[1] and p[0] > 0 and p[1] < dm.distance_map.shape[0] and p[1] > 0): 
                value = dm.distance_map[p[1], p[0]]
                if (value <= 0.7): 
                    value /= 50
                else: 
                    value = 1
                distances.append(value)
            else:
                distances.append(0)
        distances = np.array(distances, dtype=object)
        average = np.array((acceleration_likelihoods +  distances) / 2)

        weights = weights * average 
        weights += 1.e-300       
        weights /= sum(weights) # normalize
        return weights


    '''
    Estimate creates an estimation of all particles
    '''
    def estimate(self,particles, weights): 
        pos = particles[:,0:2]
        mean = np.average(pos, weights=weights, axis=0)
        var  = np.average((pos - mean)**2, weights=weights, axis=0)
        return mean, var

    '''
    Resample function resamples all particles
    '''
    def resample_from_index(particles, weights, indexes):
        particles[:] = particles[indexes]
        weights.resize(len(particles))
        weights.fill(1.0 / len(weights))
        weights =  np.array(weights, dtype=float)
        return weights, particles
    '''
    Measures the number of particles contributing to the propability distribution
    '''
    def neff(weights):
        return 1. / np.sum(np.square(weights))
