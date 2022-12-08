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
import cmapy
from os.path import join as ospj

import voc12.dataloader
from misc import torchutils, imutils
import sklearn.metrics
from chainercv.datasets import VOCSemanticSegmentationDataset

cudnn.enabled = True

CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)

def make_cam(input_img, scoremap, cam_option=cv2.COLORMAP_JET, cam_fuse_option=cv2.COLORMAP_JET):
    heatmap_resize = cv2.resize(scoremap, (input_img.shape[1], input_img.shape[0]), cv2.INTER_NEAREST)
    heatmap = np.uint8(255 * heatmap_resize.squeeze())
    heatmap_fuse_native = cv2.applyColorMap(heatmap, cam_fuse_option)
    heatmap_native = cv2.applyColorMap(heatmap, cam_option)

    heatmap = heatmap_fuse_native * 0.7 + input_img
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
        if list_qualitative_cam[ri] is None:
            list_qualitative_cam[ri] = [heatmap]
        else:
            list_qualitative_cam[ri].append(heatmap)
        count_qualitative_cam += 1
    elif count_qualitative_cam == grid_size[0] * grid_size[1]:
        for i in range(grid_size[0]):
            list_qualitative_cam[i] = np.hstack(list_qualitative_cam[i])
        qualitative_cam = np.vstack(list_qualitative_cam)
        qualitative_cam_path = ospj(log_folder, f'qualitative_cam_{epoch}.jpg')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        cv2.imwrite(ospj(qualitative_cam_path), qualitative_cam)
        count_qualitative_cam += 1

    return count_qualitative_cam, list_qualitative_cam

def qualitative_cam(data_root, log_folder, cam, image_id, label, gt_mask):
    input_img = cv2.imread(ospj(data_root, image_id + ".jpg"))
    input_img = cv2.resize(input_img, (cam.shape[0], cam.shape[1]))
    heatmap, heatmap_native = make_cam(input_img, cam, cv2.COLORMAP_HOT, cmapy.cmap('seismic'))
    heatmap = heatmap.astype(int)
    heatmap_native = heatmap_native.astype(int)


    save_path = ospj(log_folder, "scoremap_fuse", '{}_{}.png'.format(image_id, label))
    save_path_native = ospj(log_folder, "scoremap_native", '{}_{}.png'.format(image_id, label))
    save_path_mask = ospj(log_folder, "gt_mask", '{}_{}.png'.format(image_id, label))
    cv2.imwrite(save_path, heatmap)
    cv2.imwrite(save_path_native, heatmap_native)
    cv2.imwrite(save_path_mask, gt_mask.astype(int) * 255)


def all_classification_metrics(y_true, y_pred, only_f1=True):
    print('Exact Match Ratio: {0}'.format(
        sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
    print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred)))
    # "samples" applies only to multilabel problems. It does not calculate a per-class measure, instead calculating the metric over the true and predicted classes
    # for each sample in the evaluation data, and returning their (sample_weight-weighted) average.
    print('Recall: {0}'.format(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('Precision: {0}'.format(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('F1 Measure: {0}'.format(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples')))
    for cls in range(20):
        y_t = np.array(y_true).transpose((1, 0))[cls].reshape(-1, 1)
        y_p = np.array(y_pred).transpose((1, 0))[cls].reshape(-1, 1)
        if not only_f1:
            print('{} Recall: {}'.format(CLASS_NAMES[cls], sklearn.metrics.precision_score(y_true=y_t, y_pred=y_p, average='weighted')))
            print('{} Precision: {}'.format(CLASS_NAMES[cls], sklearn.metrics.recall_score(y_true=y_t, y_pred=y_p, average='weighted')))
        print('{} F1 Measure: {}'.format(CLASS_NAMES[cls], sklearn.metrics.f1_score(y_true=y_t, y_pred=y_p, average='weighted')))

def _work(process_id, model, dataset, args, epoch, dataset_label):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    data_root = ospj(args.voc12_root, "JPEGImages")
    count_qualitative_cam = 0
    grid_size = (20, 20)
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
            if not args.cam_to_ir_label_pass:
                gt_seg = dataset_label._get_label(dataset_label.ids.index(img_name))

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True), return_norelu=True)
                       for img in pack['img']]

            logits = []
            for o in outputs:
                logits.append(torchutils.gap2d(o[1].unsqueeze(0)).squeeze(0).cpu().detach())

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

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            np.save(os.path.join("train_log", args.cam_out_dir, "scoremap", img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            # Save Qualitative CAM
            if not args.cam_to_ir_label_pass:
                highres_cam = t2n(highres_cam)
                for i, valid_label in enumerate(valid_cat):
                    valid_label = valid_label.item()
                    resize_cam = cv2.resize(highres_cam[i], resize_img,
                                            interpolation=cv2.INTER_CUBIC)
                    count_qualitative_cam, list_qualitative_cam = \
                        qualitative_grid_cam(data_root, ospj("train_log", args.cam_out_dir),
                                             resize_cam, img_name, count_qualitative_cam,
                                             grid_size, list_qualitative_cam, epoch)
                    gt_mask = np.zeros(gt_seg.shape)
                    gt_mask[np.where(gt_seg == valid_label+1)] = 1
                    qualitative_cam(data_root, ospj("train_log", args.cam_out_dir),
                                    resize_cam, img_name, CLASS_NAMES[valid_label+1], gt_mask)

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

    label_list, cls_list = label_list, cls_list
    all_classification_metrics(label_list, cls_list)

def run(args, state_dict=None, epoch=10):
    if not args.cam_to_ir_label_pass:
        args.train_list = "voc12/train.txt"
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    if state_dict is not None:
        cam_weights = state_dict
    else:
        if args.train_ema:
            args.cam_weights_name += '_multi' if args.choose_multi_to_val else '_single'
        cam_weights = torch.load(args.cam_weights_name + '.pth')
    model.load_state_dict(cam_weights, strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count() * 2

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset_label = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args, epoch, dataset_label), join=True)
    print(']')

    torch.cuda.empty_cache()