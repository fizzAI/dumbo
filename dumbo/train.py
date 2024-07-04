from .config import Config, Tasks, Optimizers, Schedulers, Loggers, Metrics
from . import utils
from fire import Fire
from composer import Trainer
import torch

def train(config_path: str):
    config = Config.from_yaml_file(config_path)
    print(f"""                       
   _           _        | Dumbo v{utils.VERSION}
 _| |_ _ _____| |_ ___  | ---------------------
| . | | |     | . | . | | Cuda enabled: {torch.cuda.is_available()}
|___|___|_|_|_|___|___| | 
                        | Axolotl has one of these so I made one too
""")
    
    match config.task.name:
        case Tasks.TEXT_CLASSIFICATION:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification(config.model.model_name_or_path, num_labels=config.task.num_labels)
        case _:
            raise ValueError(f"Task {config.task} not supported")

    from transformers import AutoTokenizer
    if config.model.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)    

if __name__ == "__main__":
    Fire(train)