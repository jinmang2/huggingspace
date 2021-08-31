from dataclasses import dataclass, field


@dataclass
class DataArguments:
    pass


from transformers import TrainingArguments


@dataclass
class ModelArguments:
    pass


@dataclass
class CollateArguments:
    pass


@dataclass
class MetricArguments:
    pass


@dataclass
class AlarmArguments:
    pass
