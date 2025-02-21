import os
import yaml

def load_paths_config():
    with open('paths.yaml', 'r') as f:
        paths_config = yaml.safe_load(f)
    return paths_config

def get_hf_home():
    # Try to get from environment first
    hf_home = os.getenv('HF_HOME')
    if hf_home:
        return hf_home
        
    # Fall back to config file
    try:
        paths_config = load_paths_config()
        return paths_config.get('cluster', {}).get('HF_HOME', '')
    except:
        return '' 