"""
Batch-specific configuration parameters

Local variables are overwritten by the contents of config.yml

"""

import os
import yaml

configfile = os.path.join(os.path.dirname(__file__), 'config.yml')

if os.path.isfile(configfile):
	config = yaml.load(open(configfile, 'rb'), Loader=yaml.FullLoader)

elif not os.path.exists(configfile):
	config = {}
	config['dataset_directory'] = None
	config['rbp_performance_dir'] = None
	with open(configfile, 'w') as f:
		yaml.dump(config, f, default_flow_style=False)


for k, v in config.items():
	locals()[k] = v




