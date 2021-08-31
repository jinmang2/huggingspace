from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics_classification(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
