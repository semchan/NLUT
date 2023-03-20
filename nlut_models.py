import torch.nn as nn
import trilinear
import torchvision.transforms as transforms
from utils.LUT import *
import net
from torch.nn import functional as F


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero. reshape
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    # feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, content, style):
        assert (content.size()[:2] == style.size()[:2])
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / (content_std.expand(size))
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2 # same dimension after padding
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) # remember this dimension
        nn.init.normal_(self.conv2d.weight,mean=0,std=0.5)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.bn.weight,mean=0,std=0.5)
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.bn(out)
        return out

class SplattingBlock2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SplattingBlock2, self).__init__()
        # self.conv1 = ConvLayer(in_channels,out_channels,3, 1)
        self.conv1 = ConvLayer(in_channels,in_channels,3, 1)
        
        # self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1)
        # self.conv_short = nn.Conv2d(shortcut_channel, out_channels, 1, 1)
        self.adain = AdaIN()
        return

    def forward(self,c,s):
        c1 = F.tanh(self.conv1(c))
        c = F.tanh(self.conv2(c1+c))
        s1 = F.tanh(self.conv1(s))
        s = F.tanh(self.conv2(s1+s))
        sed = self.adain(c,s)
        return sed

class NLUTNet(nn.Module): 
    def __init__(self, nsw, dim, *args, **kwargs):
        super(NLUTNet, self).__init__()
        vgg = net.vgg
        vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
        self.encoder = net.Net(vgg)
        self.encoder.eval()
        self.adain = AdaIN()

        self.SB2 = SplattingBlock2(64,256) # 32 is not real
        self.SB3 = SplattingBlock2(128, 256)
        self.SB4 = SplattingBlock2(256, 256)
        self.SB5 = SplattingBlock2(512, 256)

        self.pg5 = nn.AdaptiveAvgPool2d(3)
        self.pg4 = nn.AdaptiveAvgPool2d(3)
        self.pg3 = nn.AdaptiveAvgPool2d(3)
        self.pg2 = nn.AdaptiveAvgPool2d(3)      
                
        self.pre = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        nsw = nsw.split("+")
        num, s, w = int(nsw[0]), int(nsw[1]), int(nsw[2])
        self.CLUTs = CLUT(num,dim,s,w)
        self.TrilinearInterpolation = TrilinearInterpolation()
        last_channel = 256*4
        self.classifier = nn.Sequential(
                nn.Conv2d(last_channel, 512,3,2),
                nn.BatchNorm2d(512),
                
                nn.Tanh(),
                nn.Conv2d(512, 512*2,1,1),
                nn.BatchNorm2d(512*2),
                
                nn.Tanh(),

                nn.Conv2d(512*2, 512,1,1),
                nn.BatchNorm2d(512),
               
                nn.Tanh(),

                nn.Conv2d(512, num,1,1),
                nn.BatchNorm2d(num),
            )


    def forward(self, img, img_org,style, TVMN=None):
        content = img
        B,C,H,W = content.size()
        content = self.pre(content)
        style = self.pre(style)     

        resize_style = torch.nn.functional.interpolate(style,(256, 256), mode='bilinear', align_corners=False)#[1, 3, 256, 256]
        resize_content = torch.nn.functional.interpolate(content,(256, 256), mode='bilinear', align_corners=False)#[1, 3, 256, 256]
        style_feats = self.encoder.encode_with_intermediate(resize_style)#style_images[2, 3, 256, 256];4
        content_feat = self.encoder.encode_with_intermediate(resize_content)#content_feat[2, 512, 32, 32]

        stylized5 = self.SB5(content_feat[-1],style_feats[-1])#[1, 256, 16, 16]
        stylized4 = self.SB4(content_feat[-2],style_feats[-2])#([1, 256, 32, 32])
        stylized3 = self.SB3(content_feat[-3],style_feats[-3])#([1, 256, 64, 64])
        stylized2 = self.SB2(content_feat[-4],style_feats[-4])#([1, 256, 128, 128])

        stylized5 = self.pg5(stylized5)#[1, 256, 16, 16]->[1, 256, 1, 1]
        stylized4 = self.pg4(stylized4)#[1, 256, 32, 32]->[1, 256, 1, 1]
        stylized3 = self.pg3(stylized3)#[1, 256, 64, 64]->[1, 256, 1, 1]
        stylized2 = self.pg2(stylized2)#[1, 256, 128, 128]->[1, 256, 1, 1]


        stylized1 = torch.cat((stylized2,stylized3,stylized4,stylized5),dim=1)
        pred = self.classifier(stylized1)[:,:,0,0]
    
        D3LUT, tvmn = self.CLUTs(pred, TVMN)


        img_out = self.TrilinearInterpolation(D3LUT, img_org)
        
        img_out = img_out + img_org
        
        output =img_out
        return  img_out, output,{
        # return img_res , {
            "LUT": D3LUT,
            "tvmn": tvmn,
        }


class CLUT(nn.Module):
    def __init__(self, num, dim=33, s="-1", w="-1", *args, **kwargs):
        super(CLUT, self).__init__()
        self.num = num
        self.dim = dim
        self.s,self.w = s,w = eval(str(s)), eval(str(w))
        # +: compressed;  -: uncompressed
        if s == -1 and w == -1: # standard 3DLUT
            self.mode = '--'
            self.LUTs = nn.Parameter(torch.zeros(num,3,dim,dim,dim))
        elif s != -1 and w == -1:  
            self.mode = '+-'
            self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(s, num*3*dim*dim))
        elif s == -1 and w != -1: 
            self.mode = '-+'
            self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(num*3*dim, w))

        else: # full-version CLUT
            self.mode = '++'
            self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
            self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(s*num*3,w))
        print("n=%d s=%d w=%d"%(num, s, w), self.mode)

    def reconstruct_luts(self):
        dim = self.dim
        num = self.num
        if self.mode == "--":
            D3LUTs = self.LUTs
        else:
            if self.mode == "+-":
                # d,s  x  s,num*3dd  -> d,num*3dd -> d,num*3,dd -> num,3,d,dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs).reshape(dim,num*3,dim*dim).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
            if self.mode == "-+":
                # num*3d,w x w,dd -> num*3d,dd -> num,3ddd
                CUBEs = self.LUTs.mm(self.w_Layers).reshape(num,3,self.dim,self.dim,self.dim)
            if self.mode == "++":
                # s*num*3, w  x   w, dd -> s*num*3,dd -> s,num*3*dd -> d,num*3*dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs.mm(self.w_Layers).reshape(-1,num*3*dim*dim)).reshape(dim,num*3,dim**2).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
            D3LUTs = cube_to_lut(CUBEs)

        return D3LUTs

    def combine(self, weight, TVMN): # n,num
        dim = self.dim
        num = self.num

        D3LUTs = self.reconstruct_luts()
        if TVMN is None:
            tvmn = 0
        else:
            tvmn = TVMN(D3LUTs)
        D3LUT = weight.mm(D3LUTs.reshape(num,-1)).reshape(-1,3,dim,dim,dim)
        return D3LUT, tvmn

    def forward(self, weight, TVMN=None):
        lut, tvmn = self.combine(weight, TVMN)
        return lut, tvmn

class BackBone(nn.Module): 
    def __init__(self, last_channel=128, ): # org both
        super(BackBone, self).__init__()
        ls = [
            *discriminator_block(3, 16, normalization=True), # 128**16
            *discriminator_block(16, 32, normalization=True), # 64**32
            *discriminator_block(32, 64, normalization=True), # 32**64
            *discriminator_block(64, 128, normalization=True), # 16**128
            *discriminator_block(128, last_channel, normalization=False), # 8**128
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(1),
        ]
        self.model = nn.Sequential(*ls)
        
    def forward(self, x):
        return self.model(x)


class TVMN(nn.Module): # (n,)3,d,d,d   or   (n,)3,d
    def __init__(self, dim=33):
        super(TVMN,self).__init__()
        self.dim = dim
        self.relu = torch.nn.ReLU()
       
        weight_r = torch.ones(1, 1, dim, dim, dim - 1, dtype=torch.float)
        weight_r[..., (0, dim - 2)] *= 2.0
        weight_g = torch.ones(1, 1, dim, dim - 1, dim, dtype=torch.float)
        weight_g[..., (0, dim - 2), :] *= 2.0
        weight_b = torch.ones(1, 1, dim - 1, dim, dim, dtype=torch.float)
        weight_b[..., (0, dim - 2), :, :] *= 2.0        
        self.register_buffer('weight_r', weight_r, persistent=False)
        self.register_buffer('weight_g', weight_g, persistent=False)
        self.register_buffer('weight_b', weight_b, persistent=False)

        self.register_buffer('tvmn_shape', torch.empty(3), persistent=False)


    def forward(self, LUT): 
        dim = self.dim
        tvmn = 0 + self.tvmn_shape
        if len(LUT.shape) > 3: # n,3,d,d,d  or  3,d,d,d
            dif_r = LUT[...,:-1] - LUT[...,1:]
            dif_g = LUT[...,:-1,:] - LUT[...,1:,:]
            dif_b = LUT[...,:-1,:,:] - LUT[...,1:,:,:]
            tvmn[0] =   torch.mean(dif_r**2 * self.weight_r[:,0]) + \
                        torch.mean(dif_g**2 * self.weight_g[:,0]) + \
                        torch.mean(dif_b**2 * self.weight_b[:,0])
            tvmn[1] =   torch.mean(self.relu(dif_r * self.weight_r[:,0])**2) + \
                        torch.mean(self.relu(dif_g * self.weight_g[:,0])**2) + \
                        torch.mean(self.relu(dif_b * self.weight_b[:,0])**2)
            tvmn[2] = 0
        else: # n,3,d  or  3,d
            dif = LUT[...,:-1] - LUT[...,1:]
            tvmn[1] = torch.mean(self.relu(dif))
            dif = dif**2
            dif[...,(0,dim-2)] *= 2.0
            tvmn[0] = torch.mean(dif)
            tvmn[2] = 0
        return tvmn


def discriminator_block(in_filters, out_filters, kernel_size=3, sp="2_1", normalization=False):
    stride = int(sp.split("_")[0])
    padding = int(sp.split("_")[1])

    layers = [
        nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        if batch == 1:
            assert 1 == trilinear.forward(lut,
                                          x,
                                          output,
                                          dim,
                                          shift,
                                          binsize,
                                          W,
                                          H,
                                          batch)
        elif batch > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == trilinear.forward(lut,
                                          x.permute(1,0,2,3).contiguous(),
                                          output,
                                          dim,
                                          shift,
                                          binsize,
                                          W,
                                          H,
                                          batch)
            output = output.permute(1, 0, 2, 3).contiguous()

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        if batch == 1:
            assert 1 == trilinear.backward(x,
                                           x_grad,
                                           lut_grad,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           batch)
        elif batch > 1:
            assert 1 == trilinear.backward(x.permute(1,0,2,3).contiguous(),
                                           x_grad.permute(1,0,2,3).contiguous(),
                                           lut_grad,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           batch)
        return lut_grad, x_grad

# trilinear_need: imgs=nchw, lut=3ddd or 13ddd
class TrilinearInterpolation(torch.nn.Module):
    def __init__(self, mo=False, clip=False):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        
        if lut.shape[0] > 1:
            if lut.shape[0] == x.shape[0]: # n,c,H,W
                res = torch.empty_like(x)
                for i in range(lut.shape[0]):
                    res[i:i+1] = TrilinearInterpolationFunction.apply(lut[i:i+1], x[i:i+1])[1]
            else:
                n,c,h,w = x.shape
                res = torch.empty(n, lut.shape[0], c, h, w).cuda()
                for i in range(lut.shape[0]):
                    res[:,i] = TrilinearInterpolationFunction.apply(lut[i:i+1], x)[1]
        else: # n,c,H,W
            res = TrilinearInterpolationFunction.apply(lut, x)[1]
        return res
