import torch
import numpy as np
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')
    model.eval()
    loss_avg = []
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)
            loss_avg.append(loss1.item())
            val_loss_meter.add({'loss1': loss1.item()})


    model.train()
    print('loss: %.4f' % (val_loss_meter.pop('loss1')))
    return np.array(loss_avg).mean()

def load_dataloader(args, train_list_file_name, validation=False):
    if validation:
        val_dataset = voc12.dataloader.VOC12ClassificationDataset(train_list_file_name,
                                                                  voc12_root=args.voc12_root,
                                                                  crop_size=512)
        val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     pin_memory=True, drop_last=True)
        return val_data_loader
    else:
        train_dataset = voc12.dataloader.VOC12ClassificationDataset(train_list_file_name, voc12_root=args.voc12_root,
                                                                    resize_long=(320, 640), hor_flip=True,
                                                                    crop_size=512, crop_method="random")
        train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                       shuffle=True, num_workers=args.num_workers,
                                       pin_memory=True, drop_last=True)
        return train_data_loader

def load_optimizer(args, model):
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=args.iter)
    return optimizer

def load_model(args):
    return getattr(importlib.import_module(args.cam_network), 'Net')()

def train_loop(args, train_dataset, model, multi=False):
    try:
        pack = next(train_dataset)
    except StopIteration:
        train_list = args.train_multi_list if multi else args.train_single_list
        train_dataset = iter(load_dataloader(args, train_list))
        pack = next(train_dataset)

    img = pack['img'].cuda()
    label = pack['label'].cuda(non_blocking=True)

    x = model(img)
    loss = F.multilabel_soft_margin_loss(x, label)

    return loss, x, train_dataset

def consistency_loss(temper, single_logits, multi_logits):
    target = F.sigmoid(temper * multi_logits)
    input = temper * single_logits
    loss =  -(target * F.logsigmoid(input) + (1 - target) * F.logsigmoid(-input))
    loss = loss.sum(dim=1) / input.size(1)
    return loss.mean()

def update_ema_variables(single_model, multi_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    single_params = single_model.trainable_parameters()
    multi_params = multi_model.trainable_parameters()
    for single_param, multi_param in zip(single_params[0], multi_params[0]):
        single_param.data.mul_(alpha).add_(1 - alpha, multi_param.data)
    for single_param, multi_param in zip(single_params[1], multi_params[1]):
        multi_param.data.mul_(alpha).add_(1 - alpha, single_param.data)

def run(args):

    single_model = load_model(args)
    multi_model = load_model(args)

    train_single_dataset = iter(load_dataloader(args, args.train_single_list))
    train_multi_dataset = iter(load_dataloader(args, args.train_multi_list))

    single_optimizer = load_optimizer(args, single_model)
    multi_optimizer = load_optimizer(args, multi_model)

    # For fc weights independence in each classes.
    for param in single_model.trainable_parameters()[0]:
        param.requires_grad = False
    for param in multi_model.trainable_parameters()[1]:
        param.requires_grad = False

    single_model = single_model.cuda()
    multi_model = multi_model.cuda()
    single_model.train()
    multi_model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    best_miou_score = 0.
    best_iter = 0
    final_miou_list = []

    for it in range(0, args.iter):
        single_loss, single_logits, train_single_dataset = \
            train_loop(args, train_single_dataset, single_model)
        multi_loss, multi_logits, train_multi_dataset = \
            train_loop(args, train_multi_dataset, multi_model, multi=True)

        loss = single_loss + multi_loss
        if args.consistency:
            con_loss = consistency_loss(args.sigmoid_temperture,
                                        single_logits, multi_logits)
            loss = loss + args.con_lam * con_loss

        avg_meter.add({'single_loss': single_loss.item(),
                       'multi_loss': multi_loss.item()})

        # assert not torch.isnan(loss)
        single_optimizer.zero_grad()
        multi_optimizer.zero_grad()

        loss.backward()

        single_optimizer.step()
        multi_optimizer.step()

        torch.cuda.empty_cache()

        update_ema_variables(single_model, multi_model,
                             args.ema_decay, single_optimizer.global_step)

        if (single_optimizer.global_step-1)%100 == 0:
            timer.update_progress(single_optimizer.global_step / args.iter)

            print('step:%5d/%5d' % (single_optimizer.global_step - 1, args.iter),
                  'single loss:%.4f' % (avg_meter.pop('single_loss')),
                  'multi loss:%.4f' % (avg_meter.pop('multi_loss')),
                  'lr: %.4f' % (single_optimizer.param_groups[0]['lr']),
                  'etc:%s' % (timer.str_estimated_complete()), flush=True)

        if (it+1) % 1000 == 0:
            model = multi_model if args.choose_multi_to_val else single_model
            timer.reset_stage()

            if args.find_best_model:
                import step.make_cam
                import step.eval_cam

                args.train_list = "voc12/train.txt"
                step.make_cam.run(args, state_dict=model.state_dict(), epoch=it)
                final_miou = []
                class_score_list = []
                final_score_list = []
                for i in range(8, 40):
                    t = i / 100.0
                    args.cam_eval_thres = t
                    final_score, class_score = step.eval_cam.run(args)
                    final_miou.append(final_score["iou"])
                    final_score_list.append(final_score)
                    class_score_list.append(class_score)
                final_miou_score = np.max(np.array(final_miou)) * 100
                final_miou_list.append(final_miou_score)
                if final_miou_score > best_miou_score:
                    torch.save(single_model.state_dict(), args.cam_weights_name + '_single' + '.pth')
                    torch.save(multi_model.state_dict(), args.cam_weights_name + '_multi' + '.pth')
                    best_miou_score = final_miou_score
                    best_iter = it


    if args.find_best_model:
        print("Best Epoch:", best_iter)
        print("Best IoU:", best_miou_score)
        print("List IoU:", final_miou_list)