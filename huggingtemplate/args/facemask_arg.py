from typing import List
from dataclasses import dataclass, field
from .base import (
    DataArguments,
    TrainingArguments,
    ModelArguments,
    CollateArguments,
    MetricArguments,
    AlarmArguments,
)
from .argparse import lambda_field


@dataclass
class FaceMaskDataArguments(DataArguments):
    train_data_dir: str = field(default="new_standard.csv")
    augmented_data_dir: List[str] = lambda_field(default=[])
    test_data_dir: str = field(default="../input/data/eval/images/")
    return_image: bool = field(default=False)
    level: int = field(default=1)
    augmentation: bool = field(default=True)
    is_valid: bool = field(default=False)


@dataclass
class FaceMaskTrainingArguments(TrainingArguments):
    wandb_project: str = field(default='ai-stage-face-mask')
    report_to: str = field(default='wandb')
    run_name: str = field(default='test')
    load_best_model_at_end: bool = field(default=True)

@dataclass
class FaceMaskModelArguments(ModelArguments):
    model_name_or_path: str = field(default='facebook/deit-base-distilled-patch16-224')
    cache_dir: str = field(default='cache')
    architectures: str = field(default='DeiTForSingleHeadClassification')
    num_labels: int = field(default=18)
    num_mask_labels: int = field(default=3)
    num_gender_labels: int = field(default=2)
    num_age_labels: int = field(default=3)
    transformers_version: str = field(default='4.10.0.dev0')
    epsilon: float = field(default=0.1)

    def __post_init__(self):
        self.label2id = {str(i): str(i) for i in range(self.num_labels)}
        self.id2label = {str(i): str(i) for i in range(self.num_labels)}


@dataclass
class FaceMaskCollateArguments(CollateArguments):
    input_size: int = field(default=224)
    color_jitter: float = field(default=0.4)
    aa: str = field(default='rand-m9-mstd0.5-inc1')
    train_interpolation: str = field(default='bicubic')
    reprob: float = field(default=0.25)
    remode: str = field(default='pixel')
    recount: int = field(default=1)
    rand_ratio: float = field(default=0.45)


@dataclass
class FaceMaskMetricArguments(MetricArguments):
    metric_fn: str = field(default="compute_metrics_classification")


@dataclass
class FaceMaskAlarmArguments(AlarmArguments):
    alarm_type: str = field(default=None)
