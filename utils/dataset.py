import copy
import json
import random

import torch.utils.data as data
import os
from PIL import Image
import torch
from utils.preprocesses import ToTensor, Pad, Resize, RoiCrop, Normalize
from utils.augmentations import RandomRotation, RandomFlip, ColorJitter, DepthNoise, RandomNoise
import numpy as np
import torchvision.transforms as transforms


def get_dataset(args):
    ds = {'train': {}, 'valid': {}, 'test': {}, 'meta': {}}
    data_views = ['cam_{}'.format(v.zfill(2)) for v in args.data_views.split('-')]
    input_views = ['cam_{}'.format(v.zfill(2)) for v in args.views.split('-')]
    rotations = args.rotations.split('-')

    for cls in os.listdir(args.path):
        with open(os.path.join(args.path, cls, 'meta.json')) as f:
            ds['meta'][cls] = json.load(f)

        for mode in ['valid', 'test']:
            ds[mode][cls] = []
            tdir = os.path.join(args.path, cls, '{}_data'.format(mode))

            for ts in os.listdir(tdir):
                sample = {}
                sdir = os.path.join(tdir, ts)
                for v in os.listdir(sdir):
                    if v in input_views:
                        _, v_id = v.split('_')
                        sample[v_id] = {
                            'rgb': os.path.join(sdir, v, '{}_rgb.png'.format(v_id)),
                            'mask': os.path.join(sdir, v, '{}_rgb_mask_gen.png'.format(v_id)),
                            'depth': os.path.join(sdir, v, '{}_depth.png'.format(v_id)),
                            'rgb_mean': os.path.join(sdir, v, '{}_rgb_mean.png'.format(v_id)),
                            'mask_mean': os.path.join(sdir, v, '{}_rgb_mean_mask_gen.png'.format(v_id)),
                            'depth_mean': os.path.join(sdir, v, '{}_depth_mean.png'.format(v_id)),
                            'meta': os.path.join(sdir, v, '{}_meta.json'.format(v_id)),
                            'view': v,
                            'hha': os.path.join(sdir, v, '{}_hha.png'.format(v_id))
                        }
                if args.multiview:
                    ds[mode][cls].append(sample)
                else:
                    ds[mode][cls] += list(sample.values())

        tdir = os.path.join(args.path, cls, 'train_data')

        ds['train'][cls] = []
        for pos in os.listdir(tdir):
            pdir = os.path.join(tdir, pos)
            for rot in os.listdir(pdir):
                if rot not in rotations:
                    continue

                sdir = os.path.join(pdir, rot)
                sample = {}
                for v in os.listdir(sdir):
                    if v in data_views:
                        _, v_id = v.split('_')
                        sample[v_id] = {
                            'rgb': os.path.join(sdir, v, '{}_rgb.png'.format(v_id)),
                            'mask': os.path.join(sdir, v, '{}_rgb_mask_gen.png'.format(v_id)),
                            'depth': os.path.join(sdir, v, '{}_depth.png'.format(v_id)),
                            'rgb_mean': os.path.join(sdir, v, '{}_rgb_mean.png'.format(v_id)),
                            'mask_mean': os.path.join(sdir, v, '{}_rgb_mean_mask_gen.png'.format(v_id)),
                            'depth_mean': os.path.join(sdir, v, '{}_depth_mean.png'.format(v_id)),
                            'meta': os.path.join(sdir, v, '{}_meta.json'.format(v_id)),
                            'rot': rot,
                            'view': v,
                            'hha': os.path.join(sdir, v, '{}_hha.png'.format(v_id))
                        }

                if args.multiview:
                    ds['train'][cls].append(sample)
                else:
                    ds['train'][cls] += list(sample.values())

    return ds


class Dataset(data.Dataset):
    def __init__(self, args, data, meta, mode=False, classes=None):
        self.args = args
        self.data = data
        self.meta = meta
        self.mode = mode

        if classes is None:
            self.classes = list(data.keys())
        else:
            self.classes = classes

        self.num_classes = len(classes)
        if mode != 'train':
            self.eval = True
        else:
            self.eval = False

        self.input_keys = args.input_keys.split('-')
        self.load_keys = args.load_keys.split('-')
        #self.views = args.views.split('-')
        self.input_views = [v.zfill(2) for v in args.views.split('-')]
        self.rotations = args.rotations.split('-')

        self.samples = []
        self.samples_ = {}

        for cls, samples in self.data.items():
            if args.shuf_views and not self.eval:
                self.samples_[cls] = {}
            cls_id = self.classes.index(cls)
            for s in samples:
                if args.multiview:
                    for view in s.keys():
                        s[view]['cls_id'] = cls_id
                        s[view]['cls'] = cls
                else:
                    s['cls_id'] = cls_id
                    s['cls'] = cls

                self.samples.append(s)

                if args.shuf_views and not self.eval:
                    for view in s.keys():
                        if view not in self.samples_[cls]:
                            self.samples_[cls][view] = []
                        self.samples_[cls][view].append(s[view])

        if args.shuf_views and not self.eval:
            self.ori_samples = copy.deepcopy(self.samples)
        else:
            self.ori_samples = None

        self.shuffle_data()

        '''
        #if 'depth' in self.args.input_keys and (args.depth_mean is None or args.depth_std is None):
            print('Getting Depth mean and std')
            #depth_mean, depth_std = self.get_depth_mean_std()
            #print('Depth mean: {}'.format(depth_mean))
            #print('Depth std: {}'.format(depth_std))
            #self.args.depth_mean = depth_mean
            #self.args.depth_std = depth_std
        '''

        if self.eval:
            self.augs = None
        else:
            augs = [ColorJitter()]
            if args.flip_aug:
                augs.append(RandomFlip())
            if args.rotation_aug:
                augs.append(RandomRotation())
            if args.enable_weight_input >= 0:
                augs.append(RandomNoise(disable=0.1 if 'size' in self.input_keys else 0.0))
            if 'depth' in args.input_keys and not args.depth2hha:
                augs.append(DepthNoise())
            self.augs = transforms.Compose(augs)

        print('{} |Augmentation transforms: {}'.format(mode, self.augs))
        self.multi_scale = args.multi_scale_training if not self.eval else False
        self.color_pre = transforms.Compose([
            RoiCrop(updsampling_threshold=args.updsampling_threshold),
            Resize(width=args.width, height=args.height,
                   multi_scale_training=self.multi_scale,
                   training_scale_low=args.training_scale_low, training_scale_high=args.training_scale_high)
        ])

        depth_mean = args.depth_mean if not args.depth2hha else [float(d) for d in args.depth_mean_hha.split('-')]
        depth_std = args.depth_std if not args.depth2hha else [float(d) for d in args.depth_std_hha.split('-')]

        print('{} |Color transforms: {}'.format(mode, self.color_pre))
        self.tensor_pre = transforms.Compose([
            ToTensor(args.depth2hha),
            Normalize(mean=None, std=None, depth_mean=depth_mean, depth_std=depth_std,
                      norm_depth=self.args.norm_depth if not self.args.depth2hha else False)
        ])
        print('{} |Tensor transforms: {}'.format(mode, self.tensor_pre))

        self.checking_batch_size = False
        self.batch_size = args.batch_size

    def __getitem__(self, index):
        x, y = self.load_sample(index)
        x['mode'] = self.mode
        if self.multi_scale:
            x['batch_size'] = self.batch_size
            if self.checking_batch_size:
                x['checking_batch_size'] = True
                self.checking_batch_size = False

        if 'meta' in self.load_keys:
            cls = self.samples[index]
            if self.args.multiview:
                cls = list(cls.values())[0]['cls']
            else:
                cls = cls['cls']
            meta = self.meta[cls]
            x['weight'] = torch.Tensor([meta['weight']])
            x['size'] = torch.Tensor([meta['size']]).sort().values / 1000# [150.0, 222.0, 157.0]

        if self.color_pre is not None:
            x = self.color_pre(x)

        if self.augs is not None:
            x = self.augs(x)

        if not self.args.visualize_samples:
            x = self.tensor_pre(x)
            if self.args.multiview:
                x = {key: x[key] for key in self.input_keys}
            else:
                #if 'weight' not in x:
                #    print(self.load_keys, self.input_keys, index, x.keys())
                if 'weight' in x:
                    x = {key: x[key] for key in self.input_keys}
                else:
                    x = {key: x[key][0] for key in self.input_keys}

                    #x['x'] = x['x'].unsqueeze(0)
                    #x['weight'] = x['weight'].unsqueeze(0)
                    #print(x['x'].shape)

        else:
            if isinstance(y, torch.Tensor):
                y = torch.where(y != 0), y[torch.where(y != 0)]
            x = {key: x[key] for key in self.load_keys}

        return x, y

    def __len__(self):
        return len(self.samples)

    def load_sample(self, index):
        s = self.samples[index]
        if self.args.multiview:
            sample = {}
            ys = []

            views = sorted(list(s.keys()))
            if len(views) > len(self.input_views):
                if self.eval:
                    views = [v for v in views if v in self.input_views]
                else:
                    views = [views[int(i)] for i in np.random.choice(list(range(len(views))),
                                                                     size=len(self.input_views), replace=False)]

            if self.args.random_view_order and not self.eval:
                random.shuffle(views)

            for view in views:
                x, y = self.load_view(s[view])
                for key in x:
                    if key not in sample:
                        sample[key] = []
                    sample[key].append(x[key])
                ys.append(y)
            if not self.args.shuf_views_cw and self.args.shuf_views:
                y = torch.zeros(self.num_classes)
                uni, c = torch.unique(torch.tensor(ys), return_counts=True)
                y[uni.long()] = c/torch.sum(c)
                if self.args.multi_head_classification:
                    y = {'out': y}
                    for j, y_ in enumerate(ys):
                        y[str(j)] = torch.zeros(self.num_classes)
                        y[str(j)][y_] = 1
            else:
                y = ys[0]
        else:
            sample, y = self.load_view(s)
            for key in sample.keys():
                sample[key] = [sample[key]]

        return sample, y

        pass

    def load_view(self, view):

        sample = {}
        if 'x' in self.load_keys:
            sample['x'] = Image.open(view['rgb'])
        if 'mask' in self.load_keys:
            sample['mask'] = Image.open(view['mask'])
        if 'depth' in self.load_keys:
            if self.args.depth2hha:
                sample['depth'] = Image.open(view['hha'])
            else:
                sample['depth'] = Image.open(view['depth'])

        y = view['cls_id']
        return sample, y

    def shuffle_data(self):
        if not self.eval and self.args.shuf_views:
            #print('Shuffeling Data: shuf_views_vw: {}, shuf_views_cw: {}, p_shuf_cw: {}, p_shuf_vw: {}'.format(
            #    self.args.shuf_views_vw, self.args.shuf_views_cw, self.args.p_shuf_cw, self.args.p_shuf_vw
            #))
            self.samples = []
            for s in self.ori_samples:
                views = list(s.keys())
                for view in views:
                    if not self.args.shuf_views_cw and np.random.rand() < self.args.p_shuf_cw:
                        cls = self.classes[int(np.random.randint(0, self.num_classes))]
                    else:
                        cls = s[view]['cls']

                    if not self.args.shuf_views_vw and np.random.rand() < self.args.p_shuf_vw:
                        v = views[int(np.random.randint(0, len(s)))]
                    else:
                        v = view
                    s_ = self.samples_[cls][v]
                    s[view] = s_[int(np.random.randint(0, len(s_)))]
                self.samples.append(s)

    def get_depth_mean_std(self):
        means = [] if not self.args.depth2hha else [[], [], []]
        stds = [] if not self.args.depth2hha else [[], [], []]
        for index in range(len(self)):
            #sample, _ = self.load_sample(index)
            sample, _ = self.__getitem__(index)
            for depth in sample['depth']:
                depth = np.array(depth)
                print(depth.shape, np.mean(depth), np.std(depth), self.args.depth2hha)
                input()
                if not self.args.depth2hha:
                    means.append(np.mean(depth))
                    stds.append(np.std(depth))
                else:
                    for i in range(3):
                        means[i].append(np.mean(depth[i, :, :]))
                        stds[i].append(np.std(depth[i, :, :]))
        if not self.args.depth2hha:
            mean = float(np.mean(means))
            std = float(np.mean(stds))
        else:
            mean = [float(np.mean(means[i])) for i in range(3)]
            std = [float(np.mean(stds[i])) for i in range(3)]
        return mean, std



if __name__ == '__main__':
    from main import get_args_parser
    import argparse


    parser = argparse.ArgumentParser('MultiView training script', parents=[get_args_parser()])
    args = parser.parse_args()

    setattr(args, 'depth2hha', False)
    setattr(args, 'norm_depth', True)
    #setattr(args, 'depth_mean', None)
    #setattr(args, 'depth_std', None)
    setattr(args, 'depth_mean_hha', None)
    setattr(args, 'depth_std_hha', None)
    setattr(args, 'input_keys', 'x-depth')
    setattr(args, 'load_keys', 'x-mask-depth')
    setattr(args, 'data_views', '1-2-3-4-5-6-7-8-9-10')
    setattr(args, 'views', '1-2-3-4-5-6-7-8-9-10')
    ds = get_dataset(args)
    classes = sorted(list(ds['train'].keys()))
    dataset = Dataset(args, ds['train'], ds['meta'], mode='train', classes=classes)
    dataset.color_pre = None
    print('color_pre', dataset.color_pre)
    print('augs', dataset.augs)
    print(dataset.get_depth_mean_std())
