import numpy as np
from scipy import stats
class ParticleFilterIMU: 
    def update(particles, weights,z, R, dm):
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