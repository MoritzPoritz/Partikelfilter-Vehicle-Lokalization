import numpy as np 
import pandas as pd
import sys
import os
def write_to_csv(name, data): 

    data_path = os.path.abspath(os.path.join(
                  os.path.dirname('data'), 
                  os.pardir)
    ) + '\\data\\'

    path = data_path+str(name)+'.csv'
    df = pd.DataFrame(data)
    df.to_csv(path)
    print("Wrote " + path)



