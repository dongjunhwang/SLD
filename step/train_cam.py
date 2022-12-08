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


def run(args, is_two_stage=False):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    if is_two_stage:
        cam_weights = torch.load(args.checkpoint_name + '.pth')
        model.load_state_dict(cam_weights, strict=True)

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    if args.freeze_fc and is_two_stage:
        for param in model.trainable_parameters()[1]:
            param.requires_grad = False

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    best_miou_score = 0.
    best_epoch = 0
    final_miou_list = []

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)


            x = model(img)
            if args.cross_entropy:
                loss = F.cross_entropy(x, label.nonzero(as_tuple=True)[1])
            else:
                loss = F.multilabel_soft_margin_loss(x, label)

            if not is_two_stage and args.error_min:
                if args.schedule_error_lambda:
                    error_lambda = (args.error_lambda * ep) / args.cam_num_epoches
                else:
                    error_lambda = args.error_lambda
                loss = loss + (error_lambda * torch.neg(torch.sum(F.softmax(x, dim=1) * F.log_softmax(x, dim=1))))

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            val_loss = validate(model, val_data_loader)
            timer.reset_stage()

        if args.find_best_model:
            import step.make_cam
            import step.eval_cam

            args.train_list = "voc12/train.txt"
            step.make_cam.run(args, state_dict=model.module.state_dict(), epoch=ep)
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
                torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
                best_miou_score = final_miou_score
                best_epoch = ep

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()

    if args.find_best_model:
        print("Best Epoch:", best_epoch)
        print("Best IoU:", best_miou_score)
        print("List IoU:", final_miou_list)