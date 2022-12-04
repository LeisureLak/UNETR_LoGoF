# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    # data_dir = args.data_dir
    # datalist_json = os.path.join(data_dir, args.json_list)
    # train_transform = transforms.Compose( # 定义了训练集的transform并在dataloader中采用
    #     [
    #         transforms.LoadImaged(keys=["image", "label"]),
    #         transforms.EnsureChannelFirstd(keys=["image", "label"]),
    #         # transforms.AddChanneld(keys=["image", "label"]),
    #         transforms.Orientationd(keys=["image", "label"], axcodes="LPI"),
    #         transforms.Spacingd(
    #             keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
    #         ),
    #         # 对整个矩阵在给定范围内采用特定强度的值缩放
    #         transforms.ScaleIntensityRanged(
    #             keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
    #         ),
    #         # 裁剪出前景目标（默认的算法select_fn是取大于0的像素部分）
    #         transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
    #
    #         # 根据正负像素比例来裁剪图像
    #         transforms.RandCropByPosNegLabeld(
    #             keys=["image", "label"],
    #             label_key="label",
    #             spatial_size=(args.roi_x, args.roi_y, args.roi_z), # 裁剪出图像的大小
    #             # 这种情况正负像素比例为1:1
    #             pos=1,
    #             neg=1,
    #             num_samples=2, # 要在这张图上裁剪几个spatial_size大小的图像
    #             image_key="image",
    #             image_threshold=0, # 裁剪出的区域，image_key对应的像素必须大于image_threshold
    #         ),
    #         # 三个轴向上的随机flip
    #         transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
    #         transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
    #         transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
    #         # 随机旋转
    #         transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
    #         # 随机缩放像素值
    #         transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
    #         # 随机移动像素值
    #         transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
    #         transforms.ToTensord(keys=["image", "label"]),
    #     ]
    # )
    # val_transform = transforms.Compose( # 定义了验证集的transform并在dataloader中采用
    #     [
    #         transforms.LoadImaged(keys=["image", "label"]),
    #         transforms.EnsureChannelFirstd(keys=["image", "label"]),
    #         # transforms.AddChanneld(keys=["image", "label"]),
    #         transforms.Orientationd(keys=["image", "label"], axcodes="LPI"),
    #         transforms.Spacingd(
    #             keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
    #         ),
    #         transforms.ScaleIntensityRanged(
    #             keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
    #         ),
    #         transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
    #         transforms.ToTensord(keys=["image", "label"]),
    #     ]
    # )
    #
    # if args.test_mode:
    #     test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    #     test_ds = data.Dataset(data=test_files, transform=val_transform)
    #     test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
    #     test_loader = data.DataLoader(
    #         test_ds,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=args.workers,
    #         sampler=test_sampler,
    #         pin_memory=True,
    #         persistent_workers=True,
    #     )
    #     loader = test_loader
    # else:
    #     datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir) # 读json文件中的"training"列表，和MSD提供的json文件格式一致
    #     if args.use_normal_dataset:
    #         train_ds = data.Dataset(data=datalist, transform=train_transform)
    #     else:
    #         train_ds = data.CacheDataset( # 默认走的这里
    #             data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
    #         )
    #     train_sampler = Sampler(train_ds) if args.distributed else None # 不分布训练就是none
    #     train_loader = data.DataLoader(
    #         train_ds,
    #         batch_size=args.batch_size,
    #         shuffle=(train_sampler is None), # sampler为none就shuffle
    #         num_workers=args.workers,
    #         sampler=train_sampler,
    #         pin_memory=True,
    #         persistent_workers=True,
    #     )
    #     val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir) # 读json文件中的"validation"列表
    #     val_ds = data.Dataset(data=val_files, transform=val_transform)
    #     val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    #     val_loader = data.DataLoader(
    #         val_ds,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=args.workers,
    #         sampler=val_sampler,
    #         pin_memory=True,
    #         persistent_workers=True,
    #     )
    #     loader = [train_loader, val_loader]
    #
    # return loader
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir) # 读json文件中的"training"列表，和MSD提供的json文件格式一致
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset( # 默认走的这里
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None # 不分布训练就是none
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None), # sampler为none就shuffle
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir) # 读json文件中的"validation"列表
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader