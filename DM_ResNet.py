from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import os
import math
import logging

import torch
import torch.nn as nn
from torch.nn import Upsample
from torch.nn import functional as F

import numpy as np
from config import cfg

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if(isinstance(x,list)):
            print(x)
        #print(x.data.shape)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        #pdb.set_trace()
        out += residual
        out = self.relu(out)

        return out



blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class DM_Module(nn.Module):
	def __init__(self, in_channel, in_spatial, cha_ratio=1, spa_ratio=1, down_ratio=1):
	    super(DM_Module, self).__init__()

	    self.in_channel = in_channel
	    self.in_spatial = in_spatial
		
	    self.inter_channel = in_channel // cha_ratio
	    self.inter_spatial = in_spatial // spa_ratio

	    self.sigmoid = nn.Sigmoid()
            self.is_skip = True
	    self.gx_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
	    )

	    self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
	    )

	    num_channel_s = 1 + self.inter_spatial
	    self.W_spatial = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_s//down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
	    )

	    self.theta_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                            kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
	    )
	    self.phi_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                        kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
	    )

				
	def forward(self, x):
	    b, c, h, w = x.size()
	    theta_xs = self.theta_spatial(x)	
	    phi_xs = self.phi_spatial(x)
	    theta_xs = theta_xs.view(b, self.inter_channel, -1)
	    theta_xs = theta_xs.permute(0, 2, 1)
	    phi_xs = phi_xs.view(b, self.inter_channel, -1)
		
	    Gs = torch.matmul(theta_xs, phi_xs)
	    Gs_in = Gs.permute(0, 2, 1).view(b, h*w, h, w)
	    Gs_out = Gs.view(b, h*w, h, w)
	    Gs_joint = torch.cat((Gs_in, Gs_out), 1)
        
	    Gs_joint = self.gg_spatial(Gs_joint)
    
	    g_xs = self.gx_spatial(x)
	    g_xs = torch.mean(g_xs, dim=1, keepdim=True)
	    ys = torch.cat((g_xs, Gs_joint), 1)
        
	    W_ys = self.W_spatial(ys)
        
            out = self.sigmoid(W_ys.expand_as(x)) * x
	    return out
        
        
class DM_ResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False

        super(DM_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.cfg = cfg
        self.dm_att4 = DM_Module(2048, 20*13,cha_ratio = self.cfg.MODEL.chr, spa_ratio=self.cfg.MODEL.spr, down_ratio=self.cfg.MODEL.dr)      
        self.is_skip = True
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256,256,256],
            [4,4,4],
        )

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=6,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if(self.is_skip):
            x = self.dm_att4(x) + x
        else:
            x = self.dm_att4(x)

        x = self.deconv_layers(x)
        out = self.final_layer(x)
        
        return out

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name == 'conv1.weight':
                    #pdb.set_trace()
                    need_init_state_dict[name] = m[:,1:2,:,:]
                elif name == 'final_layer.weight':
                    need_init_state_dict[name] = m[:6,:,:,:]
                elif name == 'final_layer.bias':
                    need_init_state_dict[name] = m[:6]
                else:
                    need_init_state_dict[name] = m
            #pdb.set_trace()
            self.load_state_dict(need_init_state_dict, strict=False)
            #self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}

def generate_mask(input,landmark,area=4):
    mask = np.zeros(input.shape)
    width = input.shape[-1]
    height= input.shape[-2]
    
    landx = landmark[:,:,0].astype(int)
    landy = landmark[:,:,1].astype(int)
    
    landx_left = landx - area
    landx_left[landx_left<0] = 0
    landx_right= landx + area+1
    landx_right[landx_right>width]=width

    landy_top = landy - area
    landy_top[landy_top<0] = 0
    landy_bottom= landy + area+1
    landy_bottom[landy_bottom>height]=height
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            mask[i,j,landy_top[i,j]:landy_bottom[i,j],landx_left[i,j]:landx_right[i,j]]=1
    return mask
    
def voting_landamrk(batch_heatmaps,isPred=False):

    #pdb.set_trace()
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    height = batch_heatmaps.shape[2]
    length = width*height
    
    x_w = (np.arange(0,length)%(int(width)))[:,np.newaxis]
    y_h = (np.arange(0,length)%(int(height)))[:,np.newaxis]
    
    x_heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    x_heatmaps_reshaped[x_heatmaps_reshaped<0]=0
    x_heatmaps_refined = x_heatmaps_reshaped/(x_heatmaps_reshaped.sum(axis=2,keepdims=1)+1e-4)
    y_heatmaps_reshaped = batch_heatmaps.transpose(0,1,3,2).reshape((batch_size, num_joints, -1))
    y_heatmaps_reshaped[y_heatmaps_reshaped<0]=0
    y_heatmaps_refined = y_heatmaps_reshaped/(y_heatmaps_reshaped.sum(axis=2,keepdims=1)+1e-4)
    x_out = np.dot((x_heatmaps_refined),x_w)
    y_out = np.dot((y_heatmaps_refined),y_h)
    pred = np.append(x_out,y_out,axis=2)
    return pred
    
def LVA(output,pred,mr):   
    mask = generate_mask(output,pred,mr)
    refine_output = output*mask
    refine_pred = voting_landamrk(refine_output)
    pred = refine_pred
    return pred
