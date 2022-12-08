# Original Code: https://github.com/jiwoon-ahn/irn

import argparse
import os
import numpy as np

from misc import pyutils

n_class = 21

CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_performance(f_best, c_best, only_iou=True):
    for i in range(n_class):
        cls_name = CLASS_NAMES[i]
        iou_score = c_best["iou"][i]
        print("{}: {}".format(cls_name, iou_score))

    if not only_iou:
        print("Final Precision: {}, Recall: {}".format(f_best["precision"], f_best["recall"]))

        print("\nPrecision Score Per Class")

        for i in range(n_class):
            cls_name = CLASS_NAMES[i]
            precision_score = c_best["precision"][i]
            print("{}: {}".format(cls_name, precision_score))

        print("\nRecall Score Per Class")
        for i in range(n_class):
            cls_name = CLASS_NAMES[i]
            recall_score = c_best["recall"][i]
            print("{}: {}".format(cls_name, recall_score))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Ood Config
    parser.add_argument("--ood_root", default='WOoD_dataset/openimages/OoD_images', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument("--ood_list", default='WOoD_dataset/openimages/ood_list.txt', type=str)
    parser.add_argument("--ood_coeff", default=0.25, type=float)
    parser.add_argument("--ood_batch_size", default=16, type=int)
    parser.add_argument("--cluster_K", default=50, type=int)
    parser.add_argument("--distance_lambda", default=0.007, type=float)
    parser.add_argument("--ood_dist_topk", default=0.2, type=float)


    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", default='Dataset/VOC2012_SEG_AUG/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.15, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)
    parser.add_argument("--np_power", default=1.0, type=float)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_irn.pth", type=str)

    parser.add_argument("--cam_out_dir", default="result/cam_ood", type=str)
    #
    parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="sem_seg_ood", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg_ood", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=False, type=str2bool)
    parser.add_argument("--make_cam_pass", default=True, type=str2bool)
    parser.add_argument("--eval_cam_pass", default=True, type=str2bool)
    parser.add_argument("--cam_to_ir_label_pass", default=False, type=str2bool)
    parser.add_argument("--train_irn_pass", default=False, type=str2bool)
    parser.add_argument("--make_ins_seg_pass", default=False, type=str2bool)
    parser.add_argument("--eval_ins_seg_pass", default=False, type=str2bool)
    parser.add_argument("--make_sem_seg_pass", default=False, type=str2bool)
    parser.add_argument("--eval_sem_seg_pass", default=False, type=str2bool)

    # Ours
    parser.add_argument("--freeze_fc", default=False, type=str2bool)
    parser.add_argument("--cross_entropy", default=False, type=str2bool)
    parser.add_argument("--checkpoint_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument('--find_best_model', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument("--decrease_lr", default=0.1, type=float)
    parser.add_argument("--two_stage", default=False, type=str2bool)
    parser.add_argument("--error_min", default=False, type=str2bool)
    parser.add_argument("--error_lambda", default=0.001, type=float)
    parser.add_argument("--schedule_error_lambda", default=False, type=str2bool)

    # EMA (Mean-Teacher)
    parser.add_argument("--train_ema", default=False, type=str2bool)
    parser.add_argument("--train_single_list", default="voc12/diag_hfd.txt", type=str)
    parser.add_argument("--train_multi_list", default="voc12/diag_hfd.txt", type=str)
    parser.add_argument("--ema_decay", default=0.99, type=float,
                        help='ema variable decay rate (default: 0.99)')
    parser.add_argument("--iter", default=10000, type=int)
    parser.add_argument("--choose_multi_to_val", default=True, type=str2bool)
    parser.add_argument("--consistency", default=True, type=str2bool)
    parser.add_argument("--con_lam", default=0.1, type=float)
    parser.add_argument("--sigmoid_temperture", default=0.5, type=float)

    # Qualtative CAM
    parser.add_argument("--qual_cam", default=True, type=str2bool)

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)

    args.ir_label_out_dir = os.path.join("train_log", args.cam_out_dir, args.ir_label_out_dir)
    args.sem_seg_out_dir = os.path.join("train_log", args.cam_out_dir, args.sem_seg_out_dir)

    os.makedirs(os.path.join("train_log", args.cam_out_dir), exist_ok=True)

    os.makedirs(os.path.join("train_log", args.cam_out_dir, "scoremap"), exist_ok=True)
    os.makedirs(os.path.join("train_log", args.cam_out_dir, "scoremap_fuse"), exist_ok=True)
    os.makedirs(os.path.join("train_log", args.cam_out_dir, "scoremap_native"), exist_ok=True)
    os.makedirs(os.path.join("train_log", args.cam_out_dir, "gt_mask"), exist_ok=True)

    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)

    pyutils.Logger(os.path.join("train_log", args.cam_out_dir, args.log_name + '.log'))
    print(vars(args))

    if args.train_cam_pass is True and args.train_ema:
        args.cam_weights_name = os.path.join("train_log", args.cam_out_dir, "best")
        import step.train_cam_ema
        timer = pyutils.Timer('step.train_cam:')
        step.train_cam_ema.run(args)

    elif args.train_cam_pass is True:
        args.cam_weights_name = os.path.join("train_log", args.cam_out_dir, "best")
        import step.train_cam
        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

        if args.two_stage:
            args.train_list = "voc12/train_aug_dsc.txt"
            args.checkpoint_name = args.cam_weights_name
            args.cam_weights_name = os.path.join("train_log", args.cam_out_dir, "best_stage_two")
            args.cam_learning_rate *= args.decrease_lr
            step.train_cam.run(args, is_two_stage=True)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args, epoch=args.cam_num_epoches)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
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

        print("mIOU: ", np.max(np.array(final_miou)))
        best_index = np.argmax(np.array(final_miou))
        f_best = final_score_list[best_index]
        c_best = class_score_list[best_index]
        print_performance(f_best, c_best)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_ins_seg_pass is True:
        import step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.eval_ins_seg_pass is True:
        import step.eval_ins_seg

        timer = pyutils.Timer('step.eval_ins_seg:')
        step.eval_ins_seg.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg


        step.eval_sem_seg.run(args)

