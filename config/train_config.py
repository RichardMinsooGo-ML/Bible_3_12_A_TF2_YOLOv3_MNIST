import os
import argparse

from easydict import EasyDict as edict

def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation of YOLOv3 using Tensorflow')
    ##############     Model configs            ########################
    
    # Yolo3d - Yolov3
    parser.add_argument('--model_def', type=str, default='config/cfg/yolo3d_yolov3.cfg', metavar='PATH', help='The path for cfgfile (only for darknet)')
    # parser.add_argument('--pretrained_path', type=str, default="checkpoints/yolov3.weights", metavar='PATH', help='the path of the pretrained checkpoint')
    parser.add_argument('--pretrained_path', type=str, default="checkpoints/Model_yolo3d_yolov3.pth", metavar='PATH', help='the path of the pretrained checkpoint')
    parser.add_argument('--save_path', type=str, default="checkpoints/Model_yolo3d_yolov3.pth", metavar='PATH', help='the path of the save checkpoint')
    

    parser.add_argument('--use_giou_loss', action='store_true', help='If true, use GIoU loss during training. If false, use MSE loss for training')
    parser.add_argument('--gradient_accumulations', type=int, default=2, help="number of gradient accums before step")

    
    ##############     Dataloader and Running configs            #######



    ##############     Training strategy            ####################
    
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=2, metavar='N', help='number of total epochs to run')


    ##############     Loss weight            ##########################

    ##############     Distributed Data Parallel            ############
    
    ##############     Evaluation configurations     ###################

    # parser.add_argument('--evaluate', action='store_true', help='only evaluate the model, not training')

    configs = edict(vars(parser.parse_args()))

    ############## Hardware configurations #############################


    ############## Dataset, logs, Checkpoints dir ######################
    
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')
    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints')
    configs.logs_dir = os.path.join(configs.working_dir, 'logs')

    if not os.path.isdir(configs.checkpoints_dir):
        os.makedirs(configs.checkpoints_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    return configs
