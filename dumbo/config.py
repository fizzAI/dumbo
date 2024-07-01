from dataclasses import dataclass, field
from composer import Time, TimeUnit
from typing import List, Tuple
from enum import Enum
from .utils import PreservingEnum

class Tasks(Enum):
    """Supported tasks"""

    TEXT_CLASSIFICATION = "text_classification" # lm for sequence modelling
    """Text classification task"""

    TEXT_GENERATION = "text_generation" # causal lm
    """Text generation task"""

class Optimizers(PreservingEnum):
    MISSING = 0
    """Missing optimizer (attempt to load from file if found)"""

    AdamW = "AdamW"
    """AdamW optimizer"""

    DecoupledAdamW = "DecoupledAdamW"
    """Decoupled AdamW optimizer"""

    SGD = "SGD"
    """Stochastic Gradient Descent optimizer"""

    DecoupledSGD = "DecoupledSGD"
    """Decoupled Stochastic Gradient Descent optimizer"""

@dataclass
class Model:
    """Model configuration"""

    name_or_path: str
    """required, huggingface name or path of the model"""

    tokenizer_name_or_path: str = None
    """huggingface name or path of the tokenizer, otherwise the model name_or_path is used"""

@dataclass
class DatasetSplits:
    """Dataset splits configuration"""

    train: str = "train"
    """name of the training split"""

    eval: str = "eval"
    """name of the evaluation split"""

@dataclass
class DatasetColumns:
    """Dataset columns configuration"""

    text: str = "text"
    """name of the text column"""

    label: str = "label"
    """name of the label column (only for text classification)"""

@dataclass
class Dataset:
    """Dataset configuration"""

    name: str
    """required, name of the dataset"""

    splits: DatasetSplits = field(default_factory=DatasetSplits)
    """dataset splits configuration"""

    columns: DatasetColumns = field(default_factory=DatasetColumns)
    """dataset columns configuration"""

@dataclass
class Task:
    """Task configuration"""

    name: Tasks

    num_labels: int = None
    """number of labels (for text classification)"""

@dataclass
class Optimizer:
    """Optimizer configuration"""

    name: Optimizers
    """required, name of the optimizer"""

    lr: float = 5e-5
    """learning rate"""

    betas: Tuple[float, float]
    """betas"""

    weight_decay: float = 0.0
    """weight decay"""

    epsilon: float = 1e-8
    """epsilon"""

@dataclass
class TrainingConfig:
    """Training configuration"""

    model: Model

    datasets: List[Dataset]
    """list of datasets to use for training"""

    task: Task