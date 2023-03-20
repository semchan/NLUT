import argparse
import torch
import numpy as np
import os
import pdb

np.set_printoptions(suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--n_cpu", type=int, default=4, help="for dataloader")
parser.add_argument("--optm", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--lambda_smooth", type=float, default=2000000.0, help="smooth regularization strength")
parser.add_argument("--lambda_mn", type=float, default=2000000.0, help="monotonicity regularization strength")

parser.add_argument("--dim", type=int, default=33, help="dimension of 3DLUT")
parser.add_argument("--losses", type=str, default="1*l1 1*cosine", help="one or more loss functions (splited by space)")
parser.add_argument("--model", type=str, default="2048+32+32", help="model configuration, n+s+w")
parser.add_argument("--name", type=str, help="name for this training (if None, use <model> instead)")

parser.add_argument("--save_root", type=str, default=".", help="root path to save images/models/logs")
parser.add_argument("--data_root", type=str, default="/data", help="root path of data")

parser.add_argument("--n_threads", type=int, default=8)
parser.add_argument('--content_dir', type=str,  default='',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='',
                    help='Directory path to a batch of style images')

parser.add_argument('--start_iter', type=int, default=0, help='starting iteration')
parser.add_argument('--max_iter', type=int, default=40) #20l
parser.add_argument('--resume', default='experiments/resume_style_lut.pth', type=str, metavar='PATH',)
parser.add_argument("--pretrained", type=str, default='experiments/model.pth')
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

parser.add_argument('--mn_cons_weight', type=float, default=100)#1,100,1000
parser.add_argument('--style_weight', type=float, default=1)#1,2
parser.add_argument('--content_weight', type=float, default=1)#1,2


parser.add_argument('--save_dir', default='finetuning_train/checkpoint',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--print_interval', type=int, default=10) #1
parser.add_argument('--save_model_interval', type=int, default=10)
#path
parser.add_argument('--content_path', type=str, default='data/cityframeat0m0s.png')
parser.add_argument('--style_path', type=str, default='data/city.jpg')
parser.add_argument('--output_path', type=str, default='data/city2.jpg', help='stylized image output path')
parser.add_argument('--src_video', type=str, default='data/city.mp4')
parser.add_argument('--dst_video', type=str, default='data/city2.mp4',help='stylized video output path')

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


