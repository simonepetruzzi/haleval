import wandb
import numpy as np

class WandbLogger:
    """
    Logger for logging to wandb.
    """
    def __init__(self, wandb_entity, wandb_project, settings, dir=None, wandb_run=None):
        self.entity = wandb_entity
        self.project = wandb_project
        self.wandb_run = wandb_run
        self.config = settings if isinstance(settings, dict) else eval(settings)
        
                
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            config=self.config,
            name=wandb_run,
            dir=dir,
        )
    
    def log(self, dict_to_log):
        wandb.log(dict_to_log)
    
    def close(self):
        wandb.finish()