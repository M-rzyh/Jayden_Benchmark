import yaml
import os
from typing import Optional

def get_eval_params(name: Optional[str] = None):
    config_file = os.path.join(os.path.dirname(__file__), "eval_params.yaml")
    with open(config_file) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    if name and name in params:
        return params[name]
    elif name and name not in params:
        return None
    
    return params

if __name__ == "__main__":
    print(get_eval_params("MOHumanoidDR-v5"))