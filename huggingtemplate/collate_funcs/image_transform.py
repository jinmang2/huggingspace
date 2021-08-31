import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import _pil_interp


def build_transform_for_beit(args):
    resize_im = args.input_size > 32

    # this should always dispatch to transforms_imagenet_train
    transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
    )
    size = int((256 / 224) * args.input_size)
    transform.transforms.insert(0, transforms.Resize(size, interpolation=3))
    aa_params = dict(
        translate_const=int(args.input_size * args.rand_ratio),
        img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
    )
    if args.train_interpolation and args.train_interpolation != 'random':
        aa_params['interpolation'] = _pil_interp(args.train_interpolation)
    # @TODO: 위치 자동으로 서치
    if args.aa.startswith('rand'):
        transform.transforms[3] = rand_augment_transform(args.aa, aa_params)
    elif args.aa.startswith('augmix'):
        aa_params['translate_pct'] = 0.3
        transform.transforms[3] = augment_and_mix_transform(args.aa, aa_params)
    else:
        transform.transforms[3] = auto_augment_transform(args.aa, aa_params)
    if not resize_im:
        # replace RandomResizedCropAndInterpolation with
        # RandomCrop
        transform.transforms[1] = transforms.RandomCrop(
            args.input_size, padding=4)

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    train_transform = transform
    eval_transform = transforms.Compose(t)

    return train_transform, eval_transform
