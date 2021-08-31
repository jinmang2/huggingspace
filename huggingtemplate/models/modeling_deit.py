import torch
import torch.nn as nn

from typing import Optional, List, Tuple, Dict
from transformers import DeiTModel, DeiTPreTrainedModel

from .modeling_outputs import (
    SequenceClassifierOutput,
    MultiLabelSequenceClassifierOutput
)


class DeiTWithLabelSmoothing(DeiTPreTrainedModel):
    _label_smoothing_factor: float = 0.0

    def __init__(self, config):
        super().__init__(config)
        self.epsilon = config.epsilon

    @property
    def epsilon(self):
        return self._label_smoothing_factor

    @epsilon.setter
    def epsilon(self, value):
        if isinstance(value, float) and value >= 0.0 and value < 1.0:
            self._label_smoothing_factor = value

    def label_smoothing(self, logits, labels):
        if self.epsilon == 0.0:
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits, labels)
        batch_size = logits.size(0)
        n_classes = logits.size(-1)

        log_probs = -torch.log_softmax(logits, dim=-1)

        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss = nll_loss.sum() / batch_size
        smoothed_loss = smoothed_loss.sum() / (batch_size * n_classes)
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class DeiTForSingleHeadClassification(DeiTWithLabelSmoothing):
    _label_smoothing_factor: float = 0.0

    def __init__(self, config):
        super().__init__(config)

        self.deit = DeiTModel(config, add_pooling_layer=False)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        # we don't use the distillation token

        loss = None
        if labels is not None:
            loss = self.label_smoothing(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DeiTForMultiHeadClassification(DeiTWithLabelSmoothing):
    _label_smoothing_factor: float = 0.0

    def __init__(self, config):
        super().__init__(config)

        self.deit = DeiTModel(config, add_pooling_layer=False)

        self.mask_classifier = nn.Linear(config.hidden_size, config.num_mask_labels)
        self.gender_classifier = nn.Linear(config.hidden_size, config.num_gender_labels)
        self.age_classifier = nn.Linear(config.hidden_size, config.num_age_labels)

        self.init_weights()

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        mask_logits = self.mask_classifier(sequence_output[:, 0, :])
        gender_logits = self.gender_classifier(sequence_output[:, 0, :])
        age_logits = self.age_classifier(sequence_output[:, 0, :])
        # we don't use the distillation token

        loss = None
        if labels is not None:
            # label 받기
            mask_labels = labels[:, 0].to(torch.int64)
            gender_labels = labels[:, 1].to(torch.int64)
            age_labels = labels[:, 2].to(torch.int64)
            # 각각 loss 계산
            mask_loss = self.label_smoothing(mask_logits, mask_labels)
            gender_loss = self.label_smoothing(gender_logits, gender_labels)
            age_loss = self.label_smoothing(age_logits, age_labels)
            # 세 loss의 합을 최소화!
            loss = mask_loss + gender_loss + age_loss

        return MultiLabelSequenceClassifierOutput(
            loss=loss,
            mask_logits=mask_logits,
            gender_logits=gender_logits,
            age_logits=age_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
