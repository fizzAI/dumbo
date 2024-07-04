from dataclasses import dataclass, field
from composer import Time, TimeUnit
from typing import List, Tuple, Type
from enum import Enum
from dataclass_wizard import YAMLWizard, LoadMixin
from dataclass_wizard.type_def import T
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

# FIXME: composer doesn't support quantized optimizers ootb yet, but we should
class Optimizers(PreservingEnum):
    MISSING = 0
    """Missing optimizer (attempt to load from file if found)"""

    AdamW = "AdamW"
    """AdamW optimizer"""

    DecoupledAdamW = "DecoupledAdamW"
    """Decoupled AdamW optimizer"""

    SGD = "SGD"
    """Stochastic Gradient Descent optimizer"""

    DecoupledSGDW = "DecoupledSGD"
    """Decoupled Stochastic Gradient Descent optimizer"""

    ScheduleFreeAdamW = "ScheduleFreeAdamW"
    """## EXPERIMENTAL
    Schedule-free AdamW optimizer (refer to [Defazio et. al, 2024](https://arxiv.org/abs/2405.15682); use a constant LR scheduler with this)"""

    ScheduleFreeSGD = "ScheduleFreeSGD"
    """## EXPERIMENTAL
    Schedule-free Stochastic Gradient Descent optimizer (refer to [Defazio et. al, 2024](https://arxiv.org/abs/2405.15682); use a constant LR scheduler with this)"""

    CAME = "CAME"
    """## EXPERIMENTAL
    CAME optimizer (refer to [Luo et. al, 2023](https://arxiv.org/abs/2307.02047))"""

    def to_optimizer(self):
        match self:
            case Optimizers.AdamW:
                from torch.optim import AdamW
                return AdamW
            case Optimizers.DecoupledAdamW:
                from composer.optim import DecoupledAdamW
                return DecoupledAdamW
            case Optimizers.SGD:
                from torch.optim import SGD
                return SGD
            case Optimizers.DecoupledSGDW:
                from composer.optim import DecoupledSGDW
                return DecoupledSGDW
            case Optimizers.ScheduleFreeAdamW:
                from schedulefree.adamw_schedulefree import AdamWScheduleFree
                return AdamWScheduleFree
            case Optimizers.ScheduleFreeSGD:
                from schedulefree.sgd_schedulefree import SGDScheduleFree
                return SGDScheduleFree
            case Optimizers.CAME:
                from came_pytorch import CAME
                return CAME
            case _:
                # FIXME: allow loading from files via generic mechanism
                raise ValueError(f"Unsupported optimizer: {self.value}")

class Schedulers(PreservingEnum):
    MISSING = 0
    """Missing optimizer (attempt to load from file if found)"""

    ConstantWithWarmupScheduler = "ConstantWithWarmupScheduler"
    """ConstantWithWarmupScheduler scheduler"""

    CosineAnnealingWithWarmupScheduler = "CosineAnnealingWithWarmupScheduler"
    """CosineAnnealingWithWarmupScheduler scheduler"""

    LinearWithWarmupScheduler = "LinearWithWarmupScheduler"
    """LinearWithWarmupScheduler scheduler"""

    def to_scheduler(self):
        match self:
            case Schedulers.ConstantWithWarmupScheduler:
                from composer.optim import ConstantWithWarmupScheduler
                return ConstantWithWarmupScheduler
            case Schedulers.CosineAnnealingWithWarmupScheduler:
                from composer.optim import CosineAnnealingWithWarmupScheduler
                return CosineAnnealingWithWarmupScheduler
            case Schedulers.LinearWithWarmupScheduler:
                from composer.optim import LinearWithWarmupScheduler
                return LinearWithWarmupScheduler
            case _:
                raise ValueError(f"Unsupported scheduler: {self.value}")

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

    def to_logger(self):
        match self:
            case Loggers.FileLogger:
                from composer.loggers import FileLogger
                return FileLogger
            case Loggers.WandBLogger:
                from composer.loggers import WandBLogger
                return WandBLogger
            case Loggers.MLFlowLogger:
                from composer.loggers import MLFlowLogger
                return MLFlowLogger
            case Loggers.CometMLLogger:
                from composer.loggers import CometMLLogger
                return CometMLLogger
            case Loggers.NeptuneLogger:
                from composer.loggers import NeptuneLogger
                return NeptuneLogger
            case Loggers.ProgressBarLogger:
                from composer.loggers import ProgressBarLogger
                return ProgressBarLogger
            case Loggers.TensorBoardLogger:
                from composer.loggers import TensorBoardLogger
                return TensorBoardLogger
            case Loggers.InMemoryLogger:
                from composer.loggers import InMemoryLogger
                return InMemoryLogger
            case Loggers.RemoteUploaderDownloader:
                from composer.loggers import RemoteUploaderDownloader
                return RemoteUploaderDownloader
            case _:
                raise ValueError(f"Unsupported logger: {self.value}")

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

def create_default_config(path, task):
    match Tasks(task):
        case Tasks.TEXT_CLASSIFICATION:
            config = Config(
                model=Model(
                    name_or_path="distilbert-base-uncased"
                ),
                datasets=[
                    Dataset(
                        name="imdb"
                    )
                ],
                task=Task(
                    name=Tasks.TEXT_CLASSIFICATION,
                    num_labels=2
                ),
                optimizer=Optimizer(
                    name=Optimizers.AdamW
                ),
                scheduler=Scheduler(
                    name=Schedulers.CosineAnnealingWithWarmupScheduler
                ),
                loggers=[
                    Logger(
                        name=Loggers.ProgressBarLogger
                    )
                ],
                metrics=[
                    Metric(
                        name=Metrics.CrossEntropy
                    ),
                    Metric(
                        name=Metrics.MultiClassAccuracy
                    )
                ],
                trainer=Trainer()
            )
        case Tasks.TEXT_GENERATION:
            config = Config(
                model=Model(
                    name_or_path="gpt2"
                ),
                datasets=[
                    Dataset(
                        name="wikitext"
                    )
                ],
                task=Task(
                    name=Tasks.TEXT_GENERATION
                ),
                optimizer=Optimizer(
                    name=Optimizers.AdamW
                ),
                scheduler=Scheduler(
                    name=Schedulers.CosineAnnealingWithWarmupScheduler
                ),
                loggers=[
                    Logger(
                        name=Loggers.ProgressBarLogger
                    )
                ],
                metrics=[
                    Metric(
                        name=Metrics.CrossEntropy
                    )
                ],
                trainer=Trainer()
            )
        case _:
            raise ValueError(f"Unsupported task for default config: {task}")

    config.to_yaml_file(path)

class Cli:
    @staticmethod
    def check(path):
        check_config_validity(path)
    
    @staticmethod
    def create(path, task="text_classification"):
        """Creates a default configuration file with basic settings."""
        create_default_config(path, task)

if __name__ == "__main__":
    Fire(Cli)