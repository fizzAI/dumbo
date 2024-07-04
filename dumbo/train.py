from .config import Config, Tasks, Optimizers, Schedulers, Loggers, Metrics
from fire import Fire
from composer import Trainer

def train(config_path: str):
    pass

if __name__ == "__main__":
    Fire(train)