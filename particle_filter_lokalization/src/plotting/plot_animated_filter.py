import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def plot_results_animated(particles, weights, xs, ground_truth, dm, Ts): 
        
    print("Particles: ",particles.dtype)
    print("weights: ",weights.dtype)
    print("xs: ",xs.dtype)
    print("ground_truth: ",ground_truth.dtype)
    fig, ax = plt.subplots()
    def animate(i):
        # First convert data to image coordinates
        
        particles_image = []
        xs_image = []
        ground_truth_image = []
        particles_image = np.array(list(map(dm.world_coordinates_to_image, particles[i][:, 0:2])))
        xs_image = dm.world_coordinates_to_image(np.array(xs[i]))
        ground_truth_image = dm.world_coordinates_to_image(np.array(ground_truth[i]))        
        # than plot the data
        ax.clear()
        plt.imshow(dm.distance_map, cmap="gray",alpha=0.2)
        ax.scatter(particles_image[:,0], particles_image[:,1], color="b", label="particles", s = weights[i] * 100)
        #ax.plot(weights[i][:,0], weights[i][:,1], c = "yellow")
        ax.scatter(xs_image[0], xs_image[1], color="red", label="estimation")
        ax.scatter(ground_truth_image[0], ground_truth_image[1], color="green", label="ground truth")
        #ax.set_ylim([2500, 0])
        #ax.set_xlim([0, 2500])
        plt.title("At: " + str(Ts[i]))
        plt.legend()


    anim = FuncAnimation(fig, animate, frames=len(xs), interval=1/10, repeat=True)
    plt.show()
