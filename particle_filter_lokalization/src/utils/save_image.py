import numpy as np
import matplotlib.pyplot as plt
import os
def save_array_as_image(array, name): 
    data_path = os.path.abspath(os.path.join(
        os.path.dirname('data'), 
        os.pardir)
    ) + '\\data\\images\\'

    #plt.plot(array[:,0], array[:,1])
    plt.imsave(data_path+name+".png", array)