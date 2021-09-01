import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.file_utils import ModelOutput


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MultiLabelSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mask_logits: torch.FloatTensor = None
    gender_logits: torch.FloatTensor = None
    age_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
