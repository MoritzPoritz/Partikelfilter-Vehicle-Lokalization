import json
import os

def write_to_json(name,data): 

    data_path = os.path.abspath(os.path.join(
                  os.path.dirname('images'), 
                  os.pardir)
    ) + '\\data\\'
    
    path = data_path+str(name)+'.json'

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4) 

def json_to_dict(name): 
    data_path = os.path.abspath(os.path.join(
                  os.path.dirname('images'), 
                  os.pardir)
    ) + '\\data\\'
    
    path = data_path+str(name)+'.json'
    with open(path) as f: 
        data = json.load(f)
        return data


