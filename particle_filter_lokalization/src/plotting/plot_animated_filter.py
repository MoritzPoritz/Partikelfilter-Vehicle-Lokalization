import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def plot_results_animated_imu(particles, weights, xs, ground_truth, dm, Ts, mse, mse_db, rmse): 
        
    print("Particles: ",particles.dtype)
    print("weights: ",weights.dtype)
    print("xs: ",xs.dtype)
    print("ground_truth: ",ground_truth.dtype)
    fig, ax1 = plt.subplots()
    mses = (xs[:,0] - ground_truth[:,0])**2 + (xs[:,0] - ground_truth[:,0])**2
    image_xs_list = []
    image_gt_list = []

    def animate(i):
        # First convert data to image coordinates
        
        particles_image = []
        xs_image = []
        ground_truth_image = []
        particles_image = np.array(list(map(dm.world_coordinates_to_image, particles[i][:, 0:2])))
        xs_image = dm.world_coordinates_to_image(np.array(xs[i]))
        ground_truth_image = dm.world_coordinates_to_image(np.array(ground_truth[i]))     

        image_xs_list.append(xs_image)
        image_gt_list.append(ground_truth_image)   
        # than plot the data
        ax1.clear()
        ax1.set_title("Partikel filter animation")
        
        plt.imshow(dm.distance_map, cmap="gray",alpha=0.2)
        ax1.scatter(particles_image[:,0], particles_image[:,1], color="b", label="particles", s = weights[i] * 1000)
        #ax1.plot(weights[i][:,0], weights[i][:,1], c = "yellow")
        ax1.scatter(xs_image[0], xs_image[1], color="red", label="estimation")
        ax1.scatter(ground_truth_image[0], ground_truth_image[1], color="green", label="ground truth")
        #plotlines
        ax1.plot(np.array(image_xs_list)[:,0], np.array(image_xs_list)[:,1], color="red", label="estimation")
        ax1.plot(np.array(image_gt_list)[:,0], np.array(image_gt_list)[:,1], color="green", label="ground truth")
        ax1.set_xlabel("x in Pixel")
        ax1.set_ylabel("y in Pixel")
        ax1.legend()
      
        plt.title("RMSE: " + '{0:.3g}'.format(rmse)+"m")
        plt.legend()


    anim = FuncAnimation(fig, animate, frames=len(xs), interval=1/10, repeat=True)
    plt.show()



def plot_results_animated_lidar(particles, weights, xs, ground_truth, dm, Ts, mse, mse_db, rmse, point_cloud): 
        
    print("Particles: ",particles.dtype)
    print("weights: ",weights.dtype)
    print("xs: ",xs.dtype)
    print("ground_truth: ",ground_truth.dtype)
    fig, ax1 = plt.subplots()
    mses = (xs[:,0] - ground_truth[:,0])**2 + (xs[:,0] - ground_truth[:,0])**2
    
    image_xs_list = []
    image_gt_list = []
    pc_image = np.array(list(map(dm.world_coordinates_to_image, point_cloud[:,:2])))

    def animate(i):
        # First convert data to image coordinates
        
        particles_image = []
        xs_image = []
        ground_truth_image = []
        particles_image = np.array(list(map(dm.world_coordinates_to_image, particles[i][:, 0:2])))
        xs_image = dm.world_coordinates_to_image(np.array(xs[i]))
        ground_truth_image = dm.world_coordinates_to_image(np.array(ground_truth[i]))     


        image_xs_list.append(xs_image)
        image_gt_list.append(ground_truth_image)   
        # than plot the data
        ax1.clear()
        ax1.set_title("Partikel filter animation")
        
        ax1.scatter(pc_image[:,0], pc_image[:,1], s = 0.2, c = "black")
        plt.imshow(dm.distance_map, cmap="gray",alpha=0.2)
        ax1.scatter(particles_image[:,0], particles_image[:,1], color="b", label="particles", s = weights[i] * 1000)
        #ax1.plot(weights[i][:,0], weights[i][:,1], c = "yellow")
        ax1.scatter(xs_image[0], xs_image[1], color="red", label="estimation")
        ax1.scatter(ground_truth_image[0], ground_truth_image[1], color="green", label="ground truth")
        #plotlines
        ax1.plot(np.array(image_xs_list)[:,0], np.array(image_xs_list)[:,1], color="red", label="estimation")
        ax1.plot(np.array(image_gt_list)[:,0], np.array(image_gt_list)[:,1], color="green", label="ground truth")
        ax1.set_xlabel("x in Pixel")
        ax1.set_ylabel("y in Pixel")
        ax1.legend()
      
        plt.title("RMSE: " + '{0:.3g}'.format(rmse) + "m")
        plt.legend()


    anim = FuncAnimation(fig, animate, frames=len(xs), interval=1/10, repeat=True)
    plt.show()
