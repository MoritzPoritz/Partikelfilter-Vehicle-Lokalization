import numpy as np 
import pandas as pd
import sys
import os
def write_to_csv(path, data): 
    
  
    path = path+'.csv'
    df = pd.DataFrame(data)
    df.to_csv(path)
    print("Wrote " + path)


def load_csv(path): 
  
    path = path+'.csv'
    return pd.read_csv(path)

