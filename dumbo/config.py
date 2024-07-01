from dataclasses import dataclass, field
from composer import Time, TimeUnit
from typing import List, Tuple, Type
from enum import Enum
from dataclass_wizard import YAMLWizard, DumpMixin, LoadMixin
from dataclass_wizard.type_def import T
from dataclass_wizard.parsers import SingleArgParser
from dataclass_wizard.models import Extras
from dataclass_wizard.abstractions import AbstractParser
from dataclass_wizard.utils.typing_compat import eval_forward_ref_if_needed
from fire import Fire
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

class Schedulers(PreservingEnum):
    MISSING = 0
    """Missing optimizer (attempt to load from file if found)"""

    CosineAnnealingWithWarmupScheduler = "CosineAnnealingWithWarmupScheduler"
    """CosineAnnealingWithWarmupScheduler scheduler"""

class Loggers(PreservingEnum):
    MISSING = 0
    """Missing logger (attempt to load from file if found)"""

    FileLogger = "FileLogger"
    """File logger"""

    WandBLogger = "WandBLogger"
    """Weights & Biases logger"""

    MLFlowLogger = "MLFlowLogger"
    """MLFlow logger"""

    CometMLLogger = "CometMLLogger"
    """CometML logger"""

    NeptuneLogger = "NeptuneLogger"
    """Neptune logger"""

    ProgressBarLogger = "ProgressBarLogger"
    """Progress bar logger"""

    TensorBoardLogger = "TensorBoardLogger"
    """TensorBoard logger"""

    InMemoryLogger = "InMemoryLogger"
    """In-memory logger"""

    RemoteUploaderDownloader = "RemoteUploaderDownloader"
    """Remote uploader/downloader"""

class Precisions(Enum):
    FP32 = "fp32"
    """Single precision"""

    FP16 = "amp_fp16"
    """Mixed precision (fp16)"""

    BF16 = "amp_bf16"
    """Mixed precision (bf16)"""

    FP8 = "amp_fp8"
    """Mixed precision (fp8, requires a Hopper+ Nvidia GPU and Transformer Engine to be installed)"""

class Metrics(PreservingEnum):
    CrossEntropy = "CrossEntropy"
    """Cross entropy loss"""

    MultiClassAccuracy = "MultiClassAccuracy"

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

    betas: Tuple[float, float] = (0.9, 0.95)
    """betas"""

    weight_decay: float = 0.0
    """weight decay"""

    epsilon: float = 1e-8
    """epsilon"""

@dataclass
class Scheduler(LoadMixin):
    """Scheduler configuration"""

    name: Schedulers
    """required, name of the scheduler"""

    warmup_time: Time = field(default_factory=lambda: Time(100, TimeUnit.BATCH))
    """warmup time"""

    @classmethod
    def get_parser_for_annotation(cls, ann_type: Type[T], base_cls: Type = None, extras: Extras = None) -> AbstractParser:
        class Parser(AbstractParser):
            def __call__(self, o) -> Type[ann_type]:
                return ann_type.from_input(o)

        ann_type = eval_forward_ref_if_needed(ann_type, base_cls)
        
        if issubclass(ann_type, Time):
            return Parser(base_cls, extras, ann_type)
        
        return super().get_parser_for_annotation(ann_type, base_cls, extras)

@dataclass
class Logger:
    """Logger configuration"""

    name: Loggers
    """required, name of the logger"""

@dataclass
class Metric:
    """Metric configuration"""

    name: Metrics
    """required, name of the metric"""

@dataclass
class Trainer(LoadMixin):
    """Trainer configuration"""

    max_duration: Time = field(default_factory=lambda: Time(1, TimeUnit.EPOCH))
    """maximum duration"""

    eval_interval: Time = field(default_factory=lambda: Time(1, TimeUnit.EPOCH))
    """evaluate metrics every N time units"""

    batch_size: int = 8
    """batch size"""

    max_length: int = 512
    """maximum length"""

    precision: Precisions = Precisions.FP32

    @classmethod
    def get_parser_for_annotation(cls, ann_type: Type[T], base_cls: Type = None, extras: Extras = None) -> AbstractParser:
        class Parser(AbstractParser):
            def __call__(self, o) -> Type[ann_type]:
                return ann_type.from_input(o)

        ann_type = eval_forward_ref_if_needed(ann_type, base_cls)
        
        if issubclass(ann_type, Time):
            return Parser(base_cls, extras, ann_type)
        
        return super().get_parser_for_annotation(ann_type, base_cls, extras)

@dataclass
class Config(YAMLWizard, LoadMixin):
    """Configuration"""

    model: Model

    datasets: List[Dataset]
    """list of datasets to use for training"""

    task: Task

    optimizer: Optimizer

    scheduler: Scheduler

    loggers: List[Logger]

    metrics: List[Metric]

    trainer: Trainer

def check_config_validity(path):
    config = Config.from_yaml_file(path)
    print(repr(config))
    print("Seems okay lol")

class Cli:
    @staticmethod
    def check(path):
        check_config_validity(path)

if __name__ == "__main__":
    Fire(Cli)