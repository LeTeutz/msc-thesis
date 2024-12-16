import json
from datetime import datetime
import os

class ExperimentLogger:
    def __init__(self, log_file: str = "results.json"):
        self.log_file = log_file

        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                json.dump([], f)


    def log_experiment(self, experiment_name: str, metrics: dict, config: dict = None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment": experiment_name,
            "metrics": metrics
        }
        if config:
            log_entry["config"] = config

        try:
            with open(self.log_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        except FileNotFoundError:
            data = []
        
        data.append(log_entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=4)
      
            
    def log_metrics(self, experiment_name: str,
                   accuracy: float,
                   avg_confidence: float,
                   adaptation_time: float = None):
        metrics = {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence
        }
        if adaptation_time is not None:
            metrics["adaptation_time"] = adaptation_time
            
        self.log_experiment(experiment_name, metrics)