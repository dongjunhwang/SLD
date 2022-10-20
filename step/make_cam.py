# Original Code: https://github.com/jiwoon-ahn/irn


import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import cv2
from os.path import join as ospj
from os.path import dirname as ospd

import voc12.dataloader
from misc import torchutils, imutils
import sklearn

cudnn.enabled = True

def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)

def make_cam(input_img, scoremap):
    heatmap_resize = cv2.resize(scoremap, (input_img.shape[1], input_img.shape[0]), cv2.INTER_NEAREST)
    heatmap = np.uint8(255 * heatmap_resize.squeeze())
    heatmap_native = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = heatmap_native * 0.5 + input_img
    heatmap = heatmap

    return heatmap.squeeze(), heatmap_native.squeeze()

def qualitative_grid_cam(data_root, log_folder, cam, image_id, count_qualitative_cam,
                         grid_size, list_qualitative_cam, epoch):
    # For Make Qualitative CAM
    input_img = cv2.imread(ospj(data_root, image_id+".jpg"))
    input_img = cv2.resize(input_img, (cam.shape[0], cam.shape[1]))
    heatmap, _ = make_cam(input_img, cam)
    heatmap = heatmap.astype(int)
    if count_qualitative_cam < grid_size[0] * grid_size[1]:
        ri = int(count_qualitative_cam/grid_size[0])
        if list_qualitative_cgit am[ri] is None:
            list_qualitative_cam[ri] = [heatmap]
        else:
            list_qualitative_cam[ri].append(heatmap)
        count_qualitative_cam += 1
    elif count_qualitative_cam == grid_size[0] * grid_size[1] and epoch % 2 == 0:
        for i in range(grid_size[0]):
            list_qualitative_cam[i] = np.hstack(list_qualitative_cam[i])
        qualitative_cam = np.vstack(list_qualitative_cam)
        qualitative_cam_path = ospj(log_folder, f'qualitative_cam_{epoch}.jpg')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        cv2.imwrite(ospj(qualitative_cam_path), qualitative_cam)
        count_qualitative_cam += 1

    return count_qualitative_cam, list_qualitative_cam

def all_classification_metrics(y_true, y_pred):
    print('Exact Match Ratio: {0}'.format(
        sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
    print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred)))
    # "samples" applies only to multilabel problems. It does not calculate a per-class measure, instead calculating the metric over the true and predicted classes
    # for each sample in the evaluation data, and returning their (sample_weight-weighted) average.
    print('Recall: {0}'.format(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('Precision: {0}'.format(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('F1 Measure: {0}'.format(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples')))

def _work(process_id, model, dataset, args, epoch):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    data_root = ospj(args.voc12_root, "JPEGImages")
    count_qualitative_cam = 0
    grid_size = (2, 2)
    list_qualitative_cam = [None] * grid_size[0]
    resize_img = (224, 224)

    cls_list = []
    label_list = []
    with torch.no_grad(), cuda.device(process_id%n_gpus):

        model.cuda()
        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True), return_norelu=True)
                       for img in pack['img']]
            # print(len(outputs), len(outputs[0]), outputs[0][0].shape, outputs[0][1].shape)
            logits = []
            for o in outputs:
                logits.append(torchutils.gap2d(o[1].unsqueeze(0)).squeeze(0).cpu().detach())
            # logits = [torchutils.gap2d(o[1].unsqueeze(0)).squeeze(0) for o in outputs]
            # print(logits[0])

            # For classification score
            logits = torch.sigmoid(logits[0]).numpy()
            logits = list(map(int, logits > 0.5))
            cls_list.append(logits)
            label_list.append(label.tolist())

            outputs = [o[0] for o in outputs]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)
            valid_cat = torch.nonzero(label)[:, 0]

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]


            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            # for s_idx, s_cam in enumerate(strided_cam):
            #     c_now = valid_cat[s_idx]
            #     strided_cam[s_idx] = strided_cam[s_idx] * torch.sigmoid(logits[1][c_now]/2)
            # print(torch.sigmoid(logits[1][c_now]/2))
            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5


            # for h_idx, h_cam in enumerate(highres_cam):
            #     c_now = valid_cat[h_idx]
            #     highres_cam[h_idx] = highres_cam[h_idx] * torch.sigmoid(logits[1][c_now]/2)
            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            # Save Qualitative CAM
            highres_cam = t2n(highres_cam)
            for i, valid_label in enumerate(valid_cat):
                valid_label = valid_label.item()
                resize_cam = cv2.resize(highres_cam[i], resize_img,
                                        interpolation=cv2.INTER_CUBIC)
                count_qualitative_cam, list_qualitative_cam = \
                    qualitative_grid_cam(data_root, args.cam_out_dir,
                                         resize_cam, img_name, count_qualitative_cam,
                                         grid_size, list_qualitative_cam, epoch)

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

    label_list, cls_list = label_list, cls_list
    all_classification_metrics(label_list, cls_list)

def run(args, state_dict=None, epoch=0):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    if state_dict is not None:
        cam_weights = state_dict
    else:
        cam_weights = torch.load(args.cam_weights_name + '.pth')
    model.load_state_dict(cam_weights, strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count() * 2

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args, epoch), join=True)
    print(']')

    torch.cuda.empty_cache()