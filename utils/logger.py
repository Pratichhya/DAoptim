import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='')

# from ignite import contrib.handlers.wandb_logger.*

# # load the 'config.json' file
# logging_params = json.load("config.json")["logger"]

# if logging_params is not None and logging_params["activate"]:
#     os.environ["WANDB_API_KEY"] = logging_params["usertoken"]
#     LOGGER = WandbLogger(project=logging_params["project"],entity=logging_params["entity"])
# else:
#     LOGGER = None


class Logger:
    def __init__(self):
        self.entries = {}
    
    def add_entry(self, entry):
        self.entries[len(self.entries)+1] = entry
        
    def __str__(self):
        return json.dumps(self.entries, sort_keys =True, indent =4)
    
    