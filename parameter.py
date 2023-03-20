import argparse
import torch
import numpy as np
import os
import pdb

np.set_printoptions(suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--n_cpu", type=int, default=4, help="for dataloader")
parser.add_argument("--optm", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--lambda_smooth", type=float, default=2000000.0, help="smooth regularization strength")
parser.add_argument("--lambda_mn", type=float, default=2000000.0, help="monotonicity regularization strength")

# epoch for train:  =1 starts from scratch, >1 load saved checkpoint of <epoch-1>
# epoch for eval:   load the model of <epoch> and evaluate
parser.add_argument("--epoch", type=int, default=310)

parser.add_argument("--n_epochs", type=int, default=380, help="last epoch of training (include)")
parser.add_argument("--dim", type=int, default=33, help="dimension of 3DLUT")
parser.add_argument("--losses", type=str, default="1*l1 1*cosine", help="one or more loss functions (splited by space)")
parser.add_argument("--model", type=str, default="2048+32+32", help="model configuration, n+s+w")
parser.add_argument("--name", type=str, help="name for this training (if None, use <model> instead)")

parser.add_argument("--save_root", type=str, default=".", help="root path to save images/models/logs")
parser.add_argument("--checkpoint_interval", type=int, default=10)
parser.add_argument("--data_root", type=str, default="/data", help="root path of data")

# Dataset Class should be implemented first for different dataset format")
parser.add_argument("--dataset", type=str, default="FiveK", help="which dateset to use")


parser.add_argument("--n_threads", type=int, default=8)
# parser.add_argument('--content_dir', type=str,  default='', 
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', type=str, default='',
#                     help='Directory path to a batch of style images')


parser.add_argument('--content_dir', type=str,  default='/home/chenys/datasets/coco/train2014/', 
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/home/chenys/datasets/coco/train2014/',
                    help='Directory path to a batch of style images')


parser.add_argument('--start_iter', type=int, default=0, help='starting iteration')
parser.add_argument('--max_iter', type=int, default=12900*2*2*2*2*4)
parser.add_argument("--pretrained", type=str, default='experiments/model.pth')
parser.add_argument('--resume', default='finetuning_train/checkpoint/resume.pth', type=str, metavar='PATH',)


parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

parser.add_argument('--mn_cons_weight', type=float, default=100)
parser.add_argument('--style_weight', type=float, default=1)
parser.add_argument('--content_weight', type=float, default=1)
parser.add_argument('--print_interval', type=int, default=100)

parser.add_argument('--save_dir', default='experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--save_model_interval', type=int, default=500)

parser.add_argument('--meth',type=int,default=0,help='0:ct , 1:ct_css , 2:mktc')
# inference iter parameter
parser.add_argument('--inferiter',type=int, default=4)


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

