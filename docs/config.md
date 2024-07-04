# Configuration documentation
Told you we were cuter than Axolotl!

## Notes
- For any configuration options that need to specify an amount of time, please refer to [Composer's documentation on their `Time` class](https://docs.mosaicml.com/projects/composer/en/stable/trainer/time.html).

## Configuration file
Dumbo configuration files are simple YAML files. Basic examples can be found in the `examples` directory; alternatively, use `python -m dumbo.config create <config_file>` to generate a template configuration file.

## Configuration options
The following settings are available in the configuration file:

### `model`
Specifies the model to use, as well other related things. It contains the following fields:

- `name_or_path`: Either the name of the model on Huggingface, or the path to a local model directory. It should be able to be loaded by `transformers.AutoModel.from_pretrained`.
- `tokenizer_name_or_path` (optional): Either the name of the tokenizer's repo on Huggingface, or the path to a local tokenizer directory. It should be able to be loaded by `transformers.AutoTokenizer.from_pretrained`. (defaults to `name_or_path`)

### `datasets`
A list of datasets to use for training. Each dataset should have the following fields:

- `name`: The name of the dataset on Huggingface.
- `splits` (optional): A dict of splits to use, containing the following entries:
  - `train`: The name of the training split. (defaults to `"train"`)
  - `eval`: The name of the evaluation split. (defaults to `"test"`)
- `columns` (optional): A dict of columns to use, containing the following entries:
  - `text`: The name of the text column. (defaults to `"text"`)
  - `label`: The name of the label column. (defaults to `"label"`)

Specify at least one dataset.

### `task`
Specifies the task to perform, as well as its settings. It contains the following fields:

- `name`: The type of task to perform. Currently, only `"text_classification"` and `"text_generation"` are supported.
- `num_labels` (optional): The number of labels in the dataset. Required for text classification tasks.

### `optimizer`
Specifies the optimizer to use, as well as its parameters. It contains the following fields:

- `name`: The name of the optimizer to use. You can select from any of the following:
    - `AdamW`
    - `DecoupledAdamW` (refer to [Composer's documentation](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.optim.DecoupledAdamW.html) for more details on this method)
    - `SGD`
    - `DecoupledSGDW` (refer to [Composer's documentation](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.optim.DecoupledSGDW.html) for more details on this method)
    - `ScheduleFreeAdamW` (refer to [Defazio et. al, 2024](https://arxiv.org/abs/2405.15682); use a constant LR scheduler with this)
    - `ScheduleFreeSGD` (refer to [Defazio et. al, 2024](https://arxiv.org/abs/2405.15682); use a constant LR scheduler with this)
    - `CAME` (refer to [Luo et. al, 2020](https://arxiv.org/abs/2307.02047))
- `learning_rate` (optional): The learning rate to use. (defaults to `5e-5`)
- `weight_decay` (optional): The weight decay to use. (defaults to `0.0`)
- `betas` (optional): The betas to use. (defaults to `(0.9, 0.95)`; only applicable to AdamW-like optimizers)
- `eps` (optional): The epsilon value to use. (defaults to `1e-8`; only applicable to AdamW-like optimizers)

### `scheduler`
Specifies the scheduler to use, as well as its parameters. It contains the following fields:

- `name`: The name of the scheduler to use. You can select from any of the following:
    - `ConstantWithWarmupScheduler`
    - `CosineAnnealingWithWarmupScheduler`
    - `LinearWithWarmupScheduler`
- `warmup_time` (optional): The amount of time to warm up for. (defaults to `100ba`)

### `loggers`
A list of loggers to pass into the trainer. Each logger should have the following fields:

- `name`: The name of the logger to use. You can select from any of the following:
    - `FileLogger`
    - `WandBLogger`
    - `MLFlowLogger`
    - `CometMLLogger`
    - `NeptuneLogger`
    - `ProgressBarLogger`
    - `TensorBoardLogger`
    - `InMemoryLogger`
    - `RemoteUploaderDownloader`

Additionally, any extra fields will be passed into the logger's constructor.

### `metrics`
A list of metrics to use for evaluation. Each metric should have the following fields:

- `name`: The name of the metric to use. You can select from any of the following:
    - `CrossEntropy` - cross-entropy loss, as implemented by Composer
    - `MultiClassAccuracy` - multi-class accuracy, as implemented by Pytorch

### `trainer`
Specifies the configuration for the trainer. It contains the following fields:

- `max_duration` (optional): The maximum amount of time to train for. (defaults to `1ep`)
- `eval_interval` (optional): The interval at which to evaluate the model. (defaults to `1ep`)
- `batch_size` (optional): The batch size to use. (defaults to `8`; note that [Composer automatically handles gradient accumulation steps](https://docs.mosaicml.com/projects/composer/en/stable/notes/auto_microbatching.html), so specify this to be whatever you want the effective batch size to be)
- `max_length` (optional): The maximum length of the input sequences. (defaults to `512`)
- `precision` (optional): The precision to use for training. Must be one of the following:
    - `fp32` (default)
    - `amp_fp16`
    - `amp_bf16`
    - `amp_fp8` (note that this requires an Nvidia GPU from Hopper or later, as well as installation of [Transformer Engine](https://github.com/NVIDIA/TransformerEngine))