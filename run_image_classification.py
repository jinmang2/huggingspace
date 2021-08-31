import os
import sys

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig

from dataclasses import dataclass, field, asdict

from huggingtemplate.args import (
    FaceMaskDataArguments,
    FaceMaskTrainingArguments,
    FaceMaskModelArguments,
    FaceMaskCollateArguments,
    FaceMaskMetricArguments,
    FaceMaskAlarmArguments,
)
from huggingtemplate import collate_funcs
from huggingtemplate import datasets
from huggingtemplate import metrics
from huggingtemplate import models
from huggingtemplate import trainers


def main():
    parser = HfArgumentParser(
        (FaceMaskDataArguments,
         FaceMaskTrainingArguments,
         FaceMaskModelArguments,
         FaceMaskCollateArguments,
         FaceMaskMetricArguments,
         FaceMaskAlarmArguments,)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # read args from json file
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        # read args from shell script or real arguments
        args = parser.parse_args_into_dataclasses()

    data_args, training_args, model_args, collate_args, metric_args, alarm_args = args

    if training_args.report_to == "wandb":
        import wandb
        wandb.login()
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    dataset_cls = getattr(datasets, data_args.dataset_class)
    trainer_cls = getattr(trainers, training_args.trainer_class)
    collate_fn = getattr(collate_funcs, collate_args.collatn_fn)
    metric_fn = getattr(metrics, metric_args.metric_fn)

    train_transform, eval_transform = collate_fn(transform_args)
    if not data_args.augmentation:
        train_transform = eval_transform

    # @TODO: argument parsing
    dataset = data_cls.load(
        data_args.train_data_dir,
        is_train=True,
        transform=train_transform,
        return_image=data_args.return_image,
        level=data_args.level,
        is_valid=data_args.is_valid
    )
    if data_args.is_valid:
        train_dataset, eval_dataset = dataset
        eval_dataset.transform = eval_transform
    else:
        train_dataset =  dataset
        eval_dataset = None

    for aug_data_path in data_args.augmented_data_dir:
        aug_dataset = data_cls.load(
            aug_data_path,
            is_train=True,
            transform=train_transform,
            return_image=data_args.return_image,
            level=data_args.level,
            is_valid=False,
        )
        train_dataset += aug_dataset

    def model_init():
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        for key, value in asdict(model_args).items():
            setattr(config, key, value)
        model_class = getattr(models, model_args.architectures)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
        )
        return model

    trainer = trainer_class(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric_fn,
        model_init=model_init,
    )

    trainer.train()
    if training_args.report_to == "wandb":
        wandb.finish()

    # @TODO: 자동화
    trainer.model.save_pretrained(
        save_directory=os.path.join(training_args.output_dir, training_args.run_name),
    )

    test_dataset = dataset_cls.load(
        data_args.test_data_dir,
        is_train=False,
        transform=eval_transform,
        return_image=data_args.return_image,
        level=data_args.level,
    )

    predictions = trainer.predict(test_dataset=test_dataset)
    preds = predictions.predictions
    preds = preds.argmax(axis=-1)

    test_dir = data_args.submit_file_dir
    submission = pd.read_csv(os.path.join(test_dir, data_args.submit_file_name))

    file2label = dict(zip(test_dataset.total_imgs, preds.tolist()))
    submission['ans'] = submission.ImageID.map(file2label)

    submission.to_csv(
        os.path.join(test_dir, "submission-" + training_args.run_name + ".csv"),
        index=False
    )


if __name__ == "__main__":
    main()
