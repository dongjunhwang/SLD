# Original Code: https://github.com/jiwoon-ahn/irn

import argparse
from voc12 import dataloader
import numpy as np

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.uint64)

    return img_name_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default='voc12/train_aug_sc.txt', type=str)
    parser.add_argument("--val_list", default='voc12/val.txt', type=str)
    parser.add_argument("--out", default="voc12/cls_labels.npy", type=str)
    parser.add_argument("--voc12_root", default="Dataset/VOC2012_SEG_AUG", type=str)
    args = parser.parse_args()

    train_name_list = load_img_name_list(args.train_list)
    val_name_list = load_img_name_list(args.val_list)

    train_val_name_list = np.concatenate([train_name_list, val_name_list], axis=0)
    label_list = dataloader.load_image_label_list_from_xml(train_val_name_list, args.voc12_root)

    total_label = np.zeros(20)

    d = dict()
    for img_name, label in zip(train_val_name_list, label_list):
        d[img_name] = label
        total_label += label

    print(total_label)
    np.save(args.out, d)