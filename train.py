import os
import numpy as np
import time
from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
from torch.utils import data
from parameter import *
from utils.losses import *
from PIL import Image
import torch.utils.data as data
import net
from nlut_models import *

import os
import numpy as np

from parameter import cuda, Tensor, device
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print(f'now device is {device}')


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        # transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def adjust_learning_rate(optimizer, iteration_count, opt):
    """Imitating the original implementation"""
    # lr = opt.lr / (1.0 + opt.lr_decay * iteration_count)
    lr = opt.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def train(setting):

def train(opt):
    # opt = setting.opt

    # -------------------------------------------------------------
    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(opt.content_dir, content_tf)
    style_dataset = FlatFolderDataset(opt.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=opt.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=opt.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=opt.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=opt.n_threads))

    model = NLUTNet(opt.model, dim=opt.dim).to(device)
    print('Total params: %.2fM' % (sum(p.numel()
          for p in model.parameters()) / 1000000.0))

    # VGG
    vgg = net.vgg
    vgg.load_state_dict(torch.load(opt.vgg))
    encoder = net.Net(vgg)
    encoder.to(device)
    encoder.eval()


    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("--------loading checkpoint----------")
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("--------no checkpoint found---------")


    # if opt.resume:
    #     if os.path.isfile(opt.resume):
    #         print("--------loading checkpoint----------")
    #         print("=> loading checkpoint '{}'".format(opt.resume))
    #         checkpoint = torch.load(opt.resume)
    #         opt.start_iter = checkpoint['iter']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         # optimizer.load_state_dict(checkpoint['optimizer'])
    #     else:
    #         print("--------no checkpoint found---------")
    
    
    
    mseloss = nn.MSELoss()
    model.train()
    TVMN_temp = TVMN(opt.dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.resume:
        if os.path.isfile(opt.resume):
            optimizer.load_state_dict(checkpoint['optimizer'])

    log_c = []
    log_s = []
    log_mse = []
    Time = time.time()

    losses = AverageMeter()
    c_losses = AverageMeter()
    s_losses = AverageMeter()
    mse_losses = AverageMeter()
    tv_losses = AverageMeter()
    mn_losses = AverageMeter()

    # -----------------------training------------------------
    for i in range(opt.start_iter, opt.max_iter):
        adjust_learning_rate(optimizer, iteration_count=i, opt=opt)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)

        stylized, st_out, others = model(
            content_images, content_images, style_images, TVMN=TVMN_temp)

        tvmn = others.get("tvmn")
        mn_cons = opt.lambda_smooth * \
            (tvmn[0]+10*tvmn[2]) + opt.lambda_mn*tvmn[1]

        loss_c, loss_s = encoder(content_images, style_images, stylized)

        loss_c = loss_c.mean()
        loss_s = loss_s.mean()
        loss_mse = mseloss(content_images, stylized)
        loss_style = opt.content_weight*loss_c + \
            opt.style_weight*loss_s + mn_cons  # +tv_cons

        # optimizer update
        optimizer.zero_grad()
        loss_style.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()

        # update loss log
        log_c.append(loss_c.item())
        log_s.append(loss_s.item())
        log_mse.append(loss_mse.item())

        losses.update(loss_style.item())
        c_losses.update(loss_c.item())
        s_losses.update(loss_s.item())
        mse_losses.update(loss_mse.item())
        mn_losses.update(mn_cons.item())

        # save image
        if i % opt.print_interval == 0:

            output_name = os.path.join(opt.save_dir, "%06d.jpg" % i)

            output_images = torch.cat((content_images.cpu(), style_images.cpu(), stylized.cpu(), st_out.cpu()),  # refined_out
                                      # output_images = torch.cat((content_images.cpu(), style_images.cpu(), stylized_rgb.cpu()), #refined_out
                                      # color_stylized.cpu(), another_content.cpu(), another_real_stylized.cpu()),
                                      0)
            save_image(output_images, output_name, nrow=opt.batch_size)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print("iter %d   time/iter: %.2f  lr: %.6f loss_mn: %.4f loss_c: %.4f   loss_s: %.4f   loss_mse: %.4f losses: %.4f " % (i,
                                                                                                                                    (time.time(
                                                                                                                                    )-Time)/opt.print_interval,
                                                                                                                                    current_lr,
                                                                                                                                    # tv_losses.avg,
                                                                                                                                    mn_losses.avg,
                                                                                                                                    c_losses.avg, s_losses.avg,
                                                                                                                                    mse_losses.avg, losses.avg
                                                                                                                                    ))

            log_c = []
            log_s = []
            Time = time.time()

        if (i + 1) % opt.save_model_interval == 0 or (i + 1) == opt.max_iter:
            # state_dict = model.module.state_dict()
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))

            state = {'iter': i, 'state_dict': state_dict,
                     'optimizer': optimizer.state_dict()}
            torch.save(state, opt.resume)
            torch.save(state, "./"+opt.save_dir+"/"+str(i)+"_style_lut.pth")


if __name__ == "__main__":
    opt = parser.parse_args()
    train(opt)
    
