import copy

import torch
import numpy as np
import json
import os
import argparse
from utils.dataset import get_dataset, Dataset
import torch.nn as nn
from models.multiview import get_model
import torch.optim.lr_scheduler as schedulers
from utils.metric import TopKAccuracy
from utils.stuff import bar_progress, lookup, load_fitting_state_dict
from utils.config import get_dataset_path
import time
import math
import sys
import random



def get_args_parser():
    parser = argparse.ArgumentParser('Set MultiView', add_help=False)

    # training
    parser.add_argument('--name', default='ResNet_50_nr3_1-6-9_dfuse_TFED_RGBD_v1', type=str) #ResNet_50_nr3_1-6-9_fuse_max-pool
    parser.add_argument('--outdir', default='./results/run011', type=str)
    parser.add_argument('--epochs', default=100, type=int) #100
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int) #32
    parser.add_argument('--num_workers', default=2, type=int) #34
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--multi_gpu', default=False, type=bool)

    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--weight_decay', default=0.0, type=float)

    # loss
    parser.add_argument('--lr_group_wise', default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1e-4, type=float)
    parser.add_argument('--lr_fusion', default=1e-4, type=float)

    # metrics
    parser.add_argument('--topk', default='1-3-5', type=str)

    # dataset
    parser.add_argument('--path', default=get_dataset_path(), type=str)
    parser.add_argument('--roicrop', default=True, type=bool)
    parser.add_argument('--shuf_views', default=False, type=bool)
    parser.add_argument('--shuf_views_cw', default=True, type=bool)
    parser.add_argument('--shuf_views_cw_disable', default=0.7, type=float)
    parser.add_argument('--enable_weight_input', default=-1, type=float)
    parser.add_argument('--shuf_views_vw', default=True, type=bool)
    parser.add_argument('--p_shuf_cw', default=1.0, type=float)
    parser.add_argument('--p_shuf_vw', default=1.0, type=float)
    parser.add_argument('--data_views', default='1-2-3-4-5-6-7-8-9-10', type=str) #1-6-9
    parser.add_argument('--views', default='1-6-9', type=str) #1-6-9
    parser.add_argument('--random_view_order', default=False, type=bool)
    parser.add_argument('--rotations', default='0-1-2-3-4-5-6-7-8-9-10-11', type=str)
    parser.add_argument('--input_keys', default='x', type=str)
    parser.add_argument('--load_keys', default='x-mask', type=str)
    parser.add_argument('--num_classes', default=-1, type=int)
    parser.add_argument('--visualize_samples', default=False, type=bool)
    parser.add_argument('--show_axis', default=True, type=bool)
    parser.add_argument('--depth_mean', default=897.8720890284345, type=float)
    parser.add_argument('--depth_std', default=859.9051176853665, type=float)
    parser.add_argument('--depth_mean_hha', default=897.8720890284345, type=float)
    parser.add_argument('--depth_std_hha', default=859.9051176853665, type=float)
    parser.add_argument('--norm_depth', default=False, type=bool)
    parser.add_argument('--depth2hha', default=False, type=bool)
    parser.add_argument('--rotation_aug', default=True, type=bool)
    parser.add_argument('--flip_aug', default=True, type=bool)
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--height', default=224, type=int)
    parser.add_argument('--multi_scale_training', default=False, type=bool)
    parser.add_argument('--training_scale_low', default=0.1, type=float)
    parser.add_argument('--training_scale_high', default=0.1, type=float)
    parser.add_argument('--updsampling_threshold', default=-1, type=int)

    # model
    parser.add_argument('--multiview', default=True, type=bool)
    parser.add_argument('--toogle_newview_model', default=False, type=bool)

    parser.add_argument('--hidden_channels', default=512, type=int)
    parser.add_argument('--model_name', default='ResNet', type=str)
    parser.add_argument('--model_version', default='50', type=str)
    parser.add_argument('--rgbd_version', default='v1', type=str)
    parser.add_argument('--fusion', default='Conv', type=str) #['Squeeze&Excite', 'SharedSqueeze&Excite', 'FC', 'Conv']
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--with_rednet_pretrained', default=False, type=bool)
    parser.add_argument('--rednet_pretrained_path', default='/home/kochpaul/Downloads/rednet_ckpt.pth', type=str)
    parser.add_argument('--overwrite_imagenet', default=True, type=bool)
    parser.add_argument('--encoder_path', default='', type=str)
    parser.add_argument('--depth_fusion', default='Squeeze&Excite', type=str) #['Squeeze&Excite',  'Conv']
    parser.add_argument('--fuse_layers', default=True, type=bool)
    parser.add_argument('--tf_layers', default=1, type=int) #['Squeeze&Excite',  'Conv']
    parser.add_argument('--multi_head_classification', default=True, type=bool)
    parser.add_argument('--rgbd_wise_multi_head', default=True, type=bool)
    parser.add_argument('--rgbd_wise_mv_fusion', default=True, type=bool)
    parser.add_argument('--with_positional_encoding', default=False, type=bool)
    parser.add_argument('--learnable_pe', default=False, type=bool)
    parser.add_argument('--pc_embed_channels', default=64, type=int)
    parser.add_argument('--pc_scale', default=200*math.pi, type=float)
    parser.add_argument('--pc_temp', default=2000, type=float)
    parser.add_argument('--use_weightNet', default=False, type=bool)
    parser.add_argument('--freeze_weightnet', default=False, type=bool)
    parser.add_argument('--use_propertyNet', default=False, type=bool)

    # scheduler
    parser.add_argument('--one_cycle', default=True, type=bool)
    parser.add_argument('--pct_start', default=0.5, type=float)
    parser.add_argument('--div_factor', default=10.0, type=float)
    parser.add_argument('--final_div_factor', default=100.0, type=float)

    # misc

    parser.add_argument('--time_max', default=1000, type=int)
    parser.add_argument('--return_view_activations', default=True, type=bool)

    return parser


def main(args, disable_sets=None):
    if disable_sets is None:
        disable_sets = []
    args.outdir = os.path.join(args.outdir, args.name)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    best_dir = os.path.join(args.outdir, '{}_best.ckpt'.format(args.name))
    logs_dir = os.path.join(args.outdir, '{}_test_logs.log'.format(args.name))
    ignore_keys = ['batch_size', 'num_workers']
    print('best_dir', best_dir)
    if os.path.exists(best_dir):
        cp = torch.load(best_dir, map_location='cpu')
        print('loaded best checkpoint from {}'.format(best_dir))
        for key, arg in cp['logs']['args'].items():
            if key in disable_sets:
                print('not setting', key)
                continue

            if getattr(args, key) != arg and key not in ignore_keys:
                print('set {}: {} -> {}'.format(key, getattr(args, key), arg))
                setattr(args, key, arg)
    else:
        print(best_dir, 'does not exist')
        cp = None

    ds = get_dataset(args)

    if args.shuf_views and not args.shuf_views_cw and args.p_shuf_cw == -1:
        args.p_shuf_cw = float(1.0/len(args.views.split('-')))
        print('Shuffle class wise with p={}'.format(args.p_shuf_cw))

    if args.shuf_views and not args.shuf_views_vw and args.p_shuf_vw == -1:
        args.p_shuf_vw = float(1 / len(args.views.split('-')))
        print('Shuffle view wise with p={}'.format(args.p_shuf_vw))


    classes = sorted(list(ds['test'].keys()))
    dataset = Dataset(args, ds['test'], ds['meta'], mode='test', classes=classes)
    args.num_classes = dataset.num_classes
    print('number of classes: {}'.format(args.num_classes))


    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             drop_last=False)

    print('##############')
    mode = 'test'
    print('------ {} ------'.format(mode))
    print('     classes: {}'.format(len(ds[mode])))
    print('     samples: {}'.format(sum([len(ds[mode][cls]) for cls in ds[mode]])))
    print('     dataset len: {}'.format(len(dataset)))
    print('     dataloader len: {}'.format(len(dataloader)))

    if not args.shuf_views_cw and args.shuf_views:
        print('Using BCEWithLogitsLoss')
        criterion = nn.BCEWithLogitsLoss()
        binary_cross = True
    else:
        print('Using CrossEntropyLoss')
        criterion = nn.CrossEntropyLoss()
        binary_cross = False

    # set device, build model, push model to device and load a checkpoint if given
    device = torch.device(args.device)
    if args.toogle_newview_model:
        args.multiview = False
    model = get_model(args)
    if args.toogle_newview_model:
        args.multiview = True
    if cp is not None:
        print('loading best model state dict from {}'.format(best_dir))
        #model.load_state_dict(cp['state_dict'])
        model = load_fitting_state_dict(model, cp['state_dict'])

    if args.multi_gpu and torch.cuda.device_count() > 1 and args.device != 'cpu':
        model = nn.DataParallel(model)
    model = model.to(device)

    metrics = {k: TopKAccuracy(int(k)) for k in args.topk.split('-')}
    metrics_copy = copy.deepcopy(metrics)
    metrics_cls = {cls: copy.deepcopy(metrics) for cls in classes}

    bestk = sorted(list(args.topk.split('-')))[0]
    del cp


    logs = {
        'acc_test': None,
        'topk_test': {k: None for k in args.topk.split('-')},
        'args': vars(args),
        'topk_cls_test': {cls: {k: None for k in args.topk.split('-')} for cls in classes}
    }

    print('##### Run Test #####')
    model.eval()
    losses = []
    t_step = time.time()
    all_activations = None

    for step, (x, y) in enumerate(dataloader):
        sys.stdout.write("\r" + 'step {}/{}'.format(step +1, len(dataloader)))
        sys.stdout.flush()
        #if step == 20:
        # break
        if args.toogle_newview_model:
            bs, views, c, h, w = x['x'].shape
            #y = y.repeat(views).flatten()
            x['x'] = x['x'].view(bs*views, c, h, w)
            #print(y)
            #input()

        cls = classes[int(y[0])]
        for key in x:
            x[key] = x[key].to(device)

        if isinstance(y, dict):
            for key in y:
                y[key] = y[key].to(device)
        else:
            y = y.to(device)

        with torch.no_grad():
            pred = model(**x)

        if args.toogle_newview_model:
            #print(pred.shape)
            pred = pred.view(bs, views, -1).max(dim=1).values
            #print(pred.shape)
            #input()

        if isinstance(pred, tuple):
            pred, activations = pred
            if all_activations is None:
                all_activations = activations
            else:
                for act_name, acts in activations.items():
                    all_activations[act_name] = torch.cat([all_activations[act_name], acts])

        #print(pred.keys())
        #input()
        if isinstance(pred, dict):
            if binary_cross:
                bi_c_loss = []
                for bi_key, bi_pred in pred.items():
                    for bi_y in y.keys():
                        if bi_y in bi_key:
                            bi_c_loss.append(criterion(bi_pred, y[bi_y]))
                            break
                loss = sum(bi_c_loss) / len(bi_c_loss)
                y = y['out']
            else:
                loss = sum([criterion(pred_, y) for pred_ in pred.values()]) / len(pred)
            #pred = pred['out']
        else:
            loss = criterion(pred, y)

        losses.append(float(loss.item()))

        if isinstance(pred, dict):
            if 'out' not in metrics:
                metrics = {'out': copy.deepcopy(metrics_copy)}

            for key, pred_ in pred.items():
                if key not in metrics:
                    metrics[key] = copy.deepcopy(metrics_copy)
                for m in metrics[key].values():
                    m.add(pred_, y)

            pred = pred['out']
        else:
            for m in metrics.values():
                m.add(pred, y)
        for k, m in metrics_cls[cls].items():
            m.add(pred, y)
            #print(cls, k, m.result())

    print('##### Test Results ######')
    if 'out' in metrics:
        logs_copy = copy.deepcopy(logs['topk_test'])
        for k_, m_ in metrics.items():
            for k, m in m_.items():
                if k == 'out':
                    logs_name = 'topk_test'
                else:
                    logs_name = 'topk_test' + '_' + str(k_)
                if logs_name not in logs:
                    logs[logs_name] = copy.deepcopy(logs_copy)

                logs[logs_name][k] = m.result()
                print('Test top {} {}: {}%'.format(k_, k, float(np.round(logs[logs_name][k], 3))))
        for cls, met in metrics_cls.items():
            for k, m in met.items():
                logs['topk_cls_test'][cls][k] = m.result()

        logs['acc_test'] = metrics['out'][bestk].result()
        logs['loss_test'] = float(np.mean(losses))
        print('Test loss: {}'.format(logs['loss_test']))

    else:
        for k, m in metrics.items():
            logs['topk_test'][k] = m.result()
            print('Test top {}: {}%'.format(k, float(np.round(logs['topk_test'][k], 3))))
        for cls, met in metrics_cls.items():
            for k, m in met.items():
                logs['topk_cls_test'][cls][k] = m.result()

        logs['acc_test'] = metrics[bestk].result()
        logs['loss_test'] = float(np.mean(losses))
        print('Test loss: {}'.format(logs['loss_test']))

    if all_activations is not None:
        logs['activations'] = {}


        for key, value in all_activations.items():
            #print(key)
            if key == 'topk':
                #print(value.shape)
                mean_act = torch.mean(value, dim=0).cpu().numpy()
                std_act = torch.mean(value, dim=0).cpu().numpy()
                #import matplotlib.pyplot as plt
                #print(mean_act.shape)
                #for l in mean_act:
                #    #plt.plot(l)
                #plt.show()
                logs['activations'][key] = {'mean': [[float(num) for num in l] for l in mean_act],
                                            'std': [[float(num) for num in l] for l in std_act]}

            else:
                #max_act = torch.max(value, dim=0)
                #print(max_act)
                topk = torch.topk(value, k=value.shape[1], dim=1)
                #print(topk.values.shape)

                max_v = topk.values
                max_id = topk.indices[:, 0]
                #print(max_id)
                uni = torch.unique(max_id, return_counts=True)
                #print(list(uni[1].cpu().numpy()))
                #print(list(torch.mean(max_v, dim=0).cpu().numpy()))
                #print(max_act)
                logs['activations'][key] = {'counts': [int(num) for num in uni[1].cpu().numpy()],
                                            'uni': [int(num) for num in uni[0].cpu().numpy()],
                                            'max': [float(num) for num in torch.mean(max_v, dim=0).cpu().numpy()]
                                            }

    with open(logs_dir, 'w') as f:
        json.dump(logs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MultiView training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

