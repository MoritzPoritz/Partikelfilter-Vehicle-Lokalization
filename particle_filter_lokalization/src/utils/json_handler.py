import json
import os
import config.config as config
def write_to_json(name,data): 


    path = name+'.json'

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4) 

def json_to_dict(name): 
    
    path = name+'.json'
    with open(path) as f: 
        data = json.load(f)
        return data


