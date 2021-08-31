from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics_mask_gender_age(eval_pred: EvalPrediction):
    predictions, labels = eval_pred

    mask = predictions[0].argmax(axis=-1)
    gender = predictions[1].argmax(axis=-1)
    age = predictions[2].argmax(axis=-1)

    preds = mask * 6 + gender * 3 + age
    labels = labels[:, 0] * 6 + labels[:, 1] * 3 + labels[:, 2]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_metrics_mask_gender(eval_pred: EvalPrediction):
    predictions, labels = eval_pred

    mask = predictions[0].argmax(axis=-1)
    gender = predictions[1].argmax(axis=-1)

    preds = mask * 2 + gender
    labels = labels[:, 0] * 2 + labels[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_metrics_age(eval_pred: EvalPrediction):
    predictions, labels = eval_pred

    age = predictions.argmax(axis=-1)

    preds = age
    labels = labels[:, 0]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
