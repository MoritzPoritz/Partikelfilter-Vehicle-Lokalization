import numpy as np 
import pandas as pd
import sys
import os
def write_structured_data_to_csv(path, data): 
    path = path+'.csv'
    df = pd.DataFrame(data)
    df.to_csv(path)
    print("Wrote " + path)

def write_numpy_array_to_csv(path, data): 
    path = path+'.csv'
    np.savetxt(path, data, delimiter=",")

def load_numpy_array_from_csv(path): 
    path = path+'.csv'
    return np.loadtxt(path)

def load_csv(path): 
  
    path = path+'.csv'
    return pd.read_csv(path)

