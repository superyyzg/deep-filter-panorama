import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F

from torch.autograd import Variable

# coding=utf-8
import math
#import torch
from torch.nn.parameter import Parameter
#from .. import functional as F
#from torch.module import Module
#from torch.nn.module.utils import _single, _pair, _triple
from numpy import prod
import collections
from itertools import repeat
import pdb

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class _ConvNd_filtermap_1x1_compression(torch.nn.Module):

    def __init__(self, in_channels, out_channels, channel_compression_1x1, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd_filtermap_1x1_compression, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        if kernel_size[0] == 1 and kernel_size[1] == 1: #for 1x1 conv layer
           self.sample_y = 1
           self.sample_x = 1
           self.sample_c = out_channels
 
           #channel_compression_1x1 == 8 or 16
           reduced_param =  (in_channels // groups)*out_channels/channel_compression_1x1
           
           #we need the reduced_param to be larger than the input channels
           assert reduced_param >= in_channels
           
           self.stride_y = 1 #kernel_size[0] 
           self.stride_x = 1 #kernel_size[1] 
           self.stride_c = reduced_param // self.sample_c
           
        else:
           if out_channels == 64: #64 = 4*4*4
              self.sample_y = 4
              self.sample_x = 4
              self.sample_c = 4
           elif out_channels == 128: #128 = 8*4*4
              self.sample_y = 8
              self.sample_x = 4
              self.sample_c = 4
           elif out_channels == 256: #256 = 8*8*4
              self.sample_y = 8
              self.sample_x = 8
              self.sample_c = 4
           elif out_channels == 512: #512 = 8*8*8
              self.sample_y = 8
              self.sample_x = 8
              self.sample_c = 8
           else:
              size_sqrt = int(math.ceil(math.sqrt(out_channels)))
              self.sample_y = size_sqrt
              self.sample_x = size_sqrt
              self.sample_c = 1
              print('undefined out_channels = %d' % (out_channels))
           #fm_channel = (in_channels // groups) // 2 #for compression rate of 18

           if in_channels // groups == 3:
              if out_channels == 64: #64 = 8*8
                 self.sample_y = 8
                 self.sample_x = 8
                 self.sample_c = 1
              else:
                 print('undefined out_channels = %d when in_channels is 3' %d (out_channels))
           self.stride_y = 2 #kernel_size[0] 
           self.stride_x = 2 #kernel_size[1] 
           self.stride_c = (in_channels // groups) // self.sample_c #in_channels // groups
        
        fm_height = self.sample_y*self.stride_y
        fm_width = self.sample_x*self.stride_x
        fm_channel = self.sample_c*self.stride_c   # for compression rate of 9     
        
        self.filtermap = Parameter(torch.Tensor(fm_channel, fm_height, fm_width))       
        #self.input_weight = Parameter(torch.Tensor(in_channels // groups * kernel_size[0] * kernel_size[1], out_channels))
        #self.transform_mat = Parameter(torch.Tensor(out_channels, self.compressed_channels))
        #self.transform_back_mat = Parameter(torch.Tensor(self.compressed_channels, out_channels))
        #if transposed:
            #self.weight = Parameter(torch.Tensor(
            #    in_channels, out_channels // groups, *kernel_size))
        #else:
            #self.weight = Parameter(torch.Tensor(
            #    out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        #self.weight.data.uniform_(-stdv, stdv)
        
        #pdb.set_trace()
        self.filtermap.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
       
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class _ConvNd_filtermap(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd_filtermap, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        if out_channels == 64: #64 = 4*4*4
           self.sample_y = 4
           self.sample_x = 4
           self.sample_c = 4
        elif out_channels == 128: #128 = 8*4*4
           self.sample_y = 8
           self.sample_x = 4
           self.sample_c = 4
        elif out_channels == 256: #256 = 8*8*4
           self.sample_y = 8
           self.sample_x = 8
           self.sample_c = 4
        elif out_channels == 512: #512 = 8*8*8
           self.sample_y = 8
           self.sample_x = 8
           self.sample_c = 8
        else:
           size_sqrt = int(math.ceil(math.sqrt(out_channels)))
           self.sample_y = size_sqrt
           self.sample_x = size_sqrt
           self.sample_c = 1
           print('undefined out_channels = %d' % (out_channels))
           #fm_channel = (in_channels // groups) // 2 #for compression rate of 18

        if in_channels // groups == 3:
           if out_channels == 64: #64 = 8*8
              self.sample_y = 8
              self.sample_x = 8
              self.sample_c = 1
           else:
              print('undefined out_channels = %d when in_channels is 3' %d (out_channels))
        self.stride_y = 2 #kernel_size[0] 
        self.stride_x = 2 #kernel_size[1] 
        self.stride_c = (in_channels // groups) // self.sample_c #in_channels // groups
        
        fm_height = self.sample_y*self.stride_y
        fm_width = self.sample_x*self.stride_x
        fm_channel = self.sample_c*self.stride_c   # for compression rate of 9     
        
        self.filtermap = Parameter(torch.Tensor(fm_channel, fm_height, fm_width))       
        #self.input_weight = Parameter(torch.Tensor(in_channels // groups * kernel_size[0] * kernel_size[1], out_channels))
        #self.transform_mat = Parameter(torch.Tensor(out_channels, self.compressed_channels))
        #self.transform_back_mat = Parameter(torch.Tensor(self.compressed_channels, out_channels))
        #if transposed:
            #self.weight = Parameter(torch.Tensor(
            #    in_channels, out_channels // groups, *kernel_size))
        #else:
            #self.weight = Parameter(torch.Tensor(
            #    out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        #self.weight.data.uniform_(-stdv, stdv)
        
        #pdb.set_trace()
        self.filtermap.data.uniform_(-stdv, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
       
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    
    
class Conv2d_filtermap(_ConvNd_filtermap):
    def __init__(self, in_channels, out_channels, kernel_size, binary_filtermap = False, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        super(Conv2d_filtermap, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        
        self.binary_filtermap = binary_filtermap
        self.reset_parameters1()
        #pdb.set_trace()
        #self.conv_weight = Parameter(torch.Tensor(out_channels, self.in_channels // self.groups, \
        #              self.kernel_size[0], self.kernel_size[1]))
        #self.conv_weight.requires_grad = False
        #self.conv_weight.cuda() 
    def reset_parameters1(self):
        fm_size = self.filtermap.size()
        fm_width = fm_size[2]
        fm_height = fm_size[1]
        fm_depth = fm_size[0]
        # not for 1x1 conv, do the padding on the spatial 
        
        self.fm_pad_width = fm_width + 1
        self.fm_pad_height = fm_height + 1
        

        self.fm_pad_depth = fm_depth*2
        #set the ids for extracting filters from filtermap
        out_channels = self.out_channels
        in_channels = self.in_channels // self.groups
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]

        sample_y = self.sample_y
        sample_x = self.sample_x
        sample_c = self.sample_c

        stride_y = self.stride_y
        stride_x = self.stride_x
        stride_c = self.stride_c
        
 
        fm_depth = self.fm_pad_depth
        fm_height = self.fm_pad_height
        fm_width = self.fm_pad_width
        
        
        ids = (torch.Tensor(range(0,k_h*k_w)))
        tmp_count = 0
        for y in range(0,k_h):
            for x in range(0,k_w):
                ids[tmp_count] = y*fm_width+x
                tmp_count = tmp_count+1
 
        ids0 = ids
               
        #pdb.set_trace() 
        for c in range(1,in_channels):
            ids_c = ids0 + c*fm_height*fm_width
            ids = torch.cat((ids,ids_c),0)
        
        #ids0 = ids
        #for x in range(1, out_channels):
        #    ids = torch.cat((ids,ids0),0)
        #pdb.set_trace()
        ids0 = ids
        for y in range(0,sample_y):
            for x in range(0,sample_x):
                if y == 0 and x == 0:
                   continue
                ss = y*stride_y*fm_width + x*stride_x
                ids_ss = ids0+ss
                ids = torch.cat((ids,ids_ss),0)
        
        #pdb.set_trace() 
        ids0 = ids
        for c in range(1,sample_c):
            ids_c = ids0+c*stride_c*fm_height*fm_width
            ids = torch.cat((ids,ids_c),0)
        
        #pdb.set_trace()
        #ids = ids.long()
        #ids = ids.detach()

        #pdb.set_trace()
        ids = ids.long()
        self.ids = Parameter(ids)
        self.ids.requires_grad = False
        #self.register_parameter()
        #if torch.max(ids) >= fm_depth*fm_height*fm_width or torch.min(ids) < 0:
        #print(torch.max(ids))
        #ids = Variable(ids)
    def extract_filters(self):
        #pdb.set_trace()
         
        out_channels = self.out_channels
        in_channels = self.in_channels // self.groups 
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]

        #for compressing the channel by 2 times
        #if in_channels != 3:
        #   filtermap_pad_tmp = torch.cat((self.filtermap,self.filtermap),0)
        #else:
        #   filtermap_pad_tmp = self.filtermap
        #filtermap_pad = torch.cat((filtermap_pad_tmp,filtermap_pad_tmp),0)
        
        #for not compressing the channel
        filtermap_pad = torch.cat((self.filtermap,self.filtermap),0)
        # not for 1x1 conv, do the padding on the spatial 
        filtermap_pad_s1 = filtermap_pad[:,1,:]
        filtermap_pad_s1 = filtermap_pad_s1[:,None,:]
        filtermap_pad = torch.cat((filtermap_pad,filtermap_pad_s1),1)
        filtermap_pad_s2 = filtermap_pad[:,:,1]
        filtermap_pad_s2 = filtermap_pad_s2[:,:,None]
        filtermap_pad = torch.cat((filtermap_pad,filtermap_pad_s2),2)

        #pdb.set_trace()
        ids = self.ids.detach()
        conv_weight = filtermap_pad.view(-1,1).index_select(0,ids)
        conv_weight = conv_weight.view(out_channels,in_channels,k_h,k_w)
        if self.binary_filtermap:
           binary_conv_weight = conv_weight.clone()
           for nf in range(0,out_channels):
               float_filter = conv_weight[nf,:,:,:];
               L1_norm = torch.norm(float_filter.view(-1,1),1);
               sign_filter = torch.sign(float_filter);
               binary_filter = sign_filter*L1_norm;
               binary_conv_weight[nf,:,:,:] = binary_filter
           return binary_conv_weight
        else:
           return conv_weight
        #pdb.set_trace()
        #for c in range(0,sample_c):
        #   for y in range(0,sample_y):
        #      for x in range(0,sample_x):
        #          filter_count = c*sample_y*sample_x + y*sample_x + x
        #          conv_weight_clone[filter_count,:,:,:] = filtermap_pad[c*stride_c:c*stride_c+in_channels, \
        #                                      y*stride_y:y*stride_y+k_h, x*stride_x:x*stride_x+k_w]
        #return conv_weight
    def forward(self, input):
        #return F.conv2d(input, self.weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        
        #conv_weight = torch.mm(self.compressed_weight, torch.tanh(self.transform_mat))
        #conv_weight = torch.mm(self.compressed_weight, (self.transform_mat))
        
        #compressed_weight = torch.mm(self.input_weight, torch.tanh(self.transform_mat))
        #conv_weight = torch.mm(compressed_weight, torch.tanh(self.transform_back_mat))
       
        #conv_weight = conv_weight.view(self.in_channels // self.groups, self.kernel_size[0], \
        #        self.kernel_size[1], self.out_channels);
        #conv_weight = conv_weight.permute(3, 0, 1, 2)
        #conv_weight = conv_weight.contiguous()
        
        #fit_loss = torch.norm(conv_weight-self.ref_conv_weight,2)

        #pdb.set_trace()
        #conv_weight = Variable(torch.Tensor(self.out_channels, self.in_channels // self.groups, \
        #              self.kernel_size[0], self.kernel_size[1]))
        #conv_weight.cuda()
        conv_weight = self.extract_filters()
        #conv_weight[0,:,:,:] = self.filtermap[0:self.in_channels, \
        #                                      0:0+self.kernel_size[0], 0:0+self.kernel_size[1]]
        out = F.conv2d(input, conv_weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
   
        return out        
        #return F.conv2d(input, conv_weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)
    def fit_filtermap1(self, conv):
        conv_weight = conv.weight.data
        conv_weight_size = conv_weight.size()
        out_channels = conv_weight_size[0]
        in_channels = conv_weight_size[1]
        k_h = conv_weight_size[2]
        k_w = conv_weight_size[3]

        sample_y = self.sample_y
        sample_x = self.sample_x
        sample_c = self.sample_c

        stride_y = self.stride_y
        stride_x = self.stride_x
        stride_c = self.stride_c

        fm_ext = torch.Tensor(sample_c*in_channels,sample_y*k_h,sample_x*k_w)
        for c in range(0,sample_c):
            for y in range(0,sample_y):
                for x in range(0,sample_x):
                    filter_count = c*sample_y*sample_x + y*sample_x + x
                    fm_ext[c*in_channels:(c+1)*in_channels,y*k_h:(y+1)*k_h,x*k_w:(x+1)*k_w] = conv_weight[filter_count,:,:,:]


        #pdb.set_trace()
        for oc in range(0,sample_c):
            for c in range(1,sample_c):
                idx = sample_c-c+oc
                if idx > sample_c-1:
                   idx = 0
                fm_ext[oc*stride_c:(oc+1)*stride_c,:,:] += \
                   fm_ext[idx*stride_c:(idx+1)*stride_c,:,:]

        fm_ext = fm_ext[0:in_channels,:,:]/sample_c

        fm_ext_h = fm_ext.size()[1]
        fm_ext_w = fm_ext.size()[2]

        for y in range(0,sample_y):
            fm_ext[:,y*k_h,:] += fm_ext[:,(y*k_h-1+fm_ext_h)%fm_ext_h,:]
            fm_ext[:,y*k_h,:] = fm_ext[:,y*k_h,:]/2

        for x in range(0,sample_x):
            fm_ext[:,:,x*k_w] += fm_ext[:,:,(x*k_w-1+fm_ext_w)%fm_ext_w] 
            fm_ext[:,:,x*k_w] = fm_ext[:,:,x*k_w]/2

        y_ids = torch.Tensor([0,1])
        y_ids0 = y_ids
        for y in range(1,sample_y):
            y_ids = torch.cat((y_ids,y_ids0+y*k_h),0)
    
        x_ids = torch.Tensor([0,1])
        x_ids0 = x_ids
        for x in range(1,sample_x):
            x_ids = torch.cat((x_ids,x_ids0+x*k_w),0)

        fm = torch.index_select(fm_ext,1,y_ids.long())
        fm = torch.index_select(fm,2,x_ids.long())
    
    
        self.filtermap.data = fm

class Conv2d_filtermap_1x1_compression(_ConvNd_filtermap_1x1_compression):
    def __init__(self, in_channels, out_channels, channel_compression_1x1, kernel_size, binary_filtermap = False, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        super(Conv2d_filtermap_1x1_compression, self).__init__(
            in_channels, out_channels, channel_compression_1x1, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        
        self.binary_filtermap = binary_filtermap
        self.reset_parameters1()
        #pdb.set_trace()
        #self.conv_weight = Parameter(torch.Tensor(out_channels, self.in_channels // self.groups, \
        #              self.kernel_size[0], self.kernel_size[1]))
        #self.conv_weight.requires_grad = False
        #self.conv_weight.cuda() 
    def reset_parameters1(self):
        fm_size = self.filtermap.size()
        fm_width = fm_size[2]
        fm_height = fm_size[1]
        fm_depth = fm_size[0]
        # not for 1x1 conv, do the padding on the spatial 
        if self.filtermap.size()[1] > 1 and self.filtermap.size()[2] > 1:
           self.fm_pad_width = fm_width + 1
           self.fm_pad_height = fm_height + 1
        #for 1x1 conv no padding on the spatial 
        else:
           self.fm_pad_width = fm_width
           self.fm_pad_height = fm_height

        self.fm_pad_depth = fm_depth*2
        #set the ids for extracting filters from filtermap
        out_channels = self.out_channels
        in_channels = self.in_channels // self.groups
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]

        sample_y = self.sample_y
        sample_x = self.sample_x
        sample_c = self.sample_c

        stride_y = self.stride_y
        stride_x = self.stride_x
        stride_c = self.stride_c
        
 
        fm_depth = self.fm_pad_depth
        fm_height = self.fm_pad_height
        fm_width = self.fm_pad_width
        
        
        ids = (torch.Tensor(range(0,k_h*k_w)))
        tmp_count = 0
        for y in range(0,k_h):
            for x in range(0,k_w):
                ids[tmp_count] = y*fm_width+x
                tmp_count = tmp_count+1
 
        ids0 = ids
               
        #pdb.set_trace() 
        for c in range(1,in_channels):
            ids_c = ids0 + c*fm_height*fm_width
            ids = torch.cat((ids,ids_c),0)
        
        #ids0 = ids
        #for x in range(1, out_channels):
        #    ids = torch.cat((ids,ids0),0)
        #pdb.set_trace()
        ids0 = ids
        for y in range(0,sample_y):
            for x in range(0,sample_x):
                if y == 0 and x == 0:
                   continue
                ss = y*stride_y*fm_width + x*stride_x
                ids_ss = ids0+ss
                ids = torch.cat((ids,ids_ss),0)
        
        #pdb.set_trace() 
        ids0 = ids
        for c in range(1,sample_c):
            ids_c = ids0+c*stride_c*fm_height*fm_width
            ids = torch.cat((ids,ids_c),0)
        
        #pdb.set_trace()
        #ids = ids.long()
        #ids = ids.detach()

        #pdb.set_trace()
        ids = ids.long()
        self.ids = Parameter(ids)
        self.ids.requires_grad = False
        #self.register_parameter()
        #if torch.max(ids) >= fm_depth*fm_height*fm_width or torch.min(ids) < 0:
        #print(torch.max(ids))
        #ids = Variable(ids)
    def extract_filters(self):
        #pdb.set_trace()
         
        out_channels = self.out_channels
        in_channels = self.in_channels // self.groups 
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]

        #for compressing the channel by 2 times
        #if in_channels != 3:
        #   filtermap_pad_tmp = torch.cat((self.filtermap,self.filtermap),0)
        #else:
        #   filtermap_pad_tmp = self.filtermap
        #filtermap_pad = torch.cat((filtermap_pad_tmp,filtermap_pad_tmp),0)
        
        #for not compressing the channel
        filtermap_pad = torch.cat((self.filtermap,self.filtermap),0)
        # not for 1x1 conv, do the padding on the spatial 
        if self.filtermap.size()[1] > 1 and self.filtermap.size()[2] > 1: 
           filtermap_pad_s1 = filtermap_pad[:,1,:]
           filtermap_pad_s1 = filtermap_pad_s1[:,None,:]
           filtermap_pad = torch.cat((filtermap_pad,filtermap_pad_s1),1)
           filtermap_pad_s2 = filtermap_pad[:,:,1]
           filtermap_pad_s2 = filtermap_pad_s2[:,:,None]
           filtermap_pad = torch.cat((filtermap_pad,filtermap_pad_s2),2)

        #pdb.set_trace()
        ids = self.ids.detach()
        conv_weight = filtermap_pad.view(-1,1).index_select(0,ids)
        conv_weight = conv_weight.view(out_channels,in_channels,k_h,k_w)
        if self.binary_filtermap:
           binary_conv_weight = conv_weight.clone()
           for nf in range(0,out_channels):
               float_filter = conv_weight[nf,:,:,:];
               L1_norm = torch.norm(float_filter.view(-1,1),1);
               sign_filter = torch.sign(float_filter);
               binary_filter = sign_filter*L1_norm;
               binary_conv_weight[nf,:,:,:] = binary_filter
           return binary_conv_weight
        else:
           return conv_weight
        #pdb.set_trace()
        #for c in range(0,sample_c):
        #   for y in range(0,sample_y):
        #      for x in range(0,sample_x):
        #          filter_count = c*sample_y*sample_x + y*sample_x + x
        #          conv_weight_clone[filter_count,:,:,:] = filtermap_pad[c*stride_c:c*stride_c+in_channels, \
        #                                      y*stride_y:y*stride_y+k_h, x*stride_x:x*stride_x+k_w]
        #return conv_weight
    def forward(self, input):
        #return F.conv2d(input, self.weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        
        #conv_weight = torch.mm(self.compressed_weight, torch.tanh(self.transform_mat))
        #conv_weight = torch.mm(self.compressed_weight, (self.transform_mat))
        
        #compressed_weight = torch.mm(self.input_weight, torch.tanh(self.transform_mat))
        #conv_weight = torch.mm(compressed_weight, torch.tanh(self.transform_back_mat))
       
        #conv_weight = conv_weight.view(self.in_channels // self.groups, self.kernel_size[0], \
        #        self.kernel_size[1], self.out_channels);
        #conv_weight = conv_weight.permute(3, 0, 1, 2)
        #conv_weight = conv_weight.contiguous()
        
        #fit_loss = torch.norm(conv_weight-self.ref_conv_weight,2)

        #pdb.set_trace()
        #conv_weight = Variable(torch.Tensor(self.out_channels, self.in_channels // self.groups, \
        #              self.kernel_size[0], self.kernel_size[1]))
        #conv_weight.cuda()
        conv_weight = self.extract_filters()
        #conv_weight[0,:,:,:] = self.filtermap[0:self.in_channels, \
        #                                      0:0+self.kernel_size[0], 0:0+self.kernel_size[1]]
        out = F.conv2d(input, conv_weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
   
        return out        
        #return F.conv2d(input, conv_weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)
    def fit_filtermap1(self, conv):
        conv_weight = conv.weight.data
        conv_weight_size = conv_weight.size()
        out_channels = conv_weight_size[0]
        in_channels = conv_weight_size[1]
        k_h = conv_weight_size[2]
        k_w = conv_weight_size[3]

        sample_y = self.sample_y
        sample_x = self.sample_x
        sample_c = self.sample_c

        stride_y = self.stride_y
        stride_x = self.stride_x
        stride_c = self.stride_c

        fm_ext = torch.Tensor(sample_c*in_channels,sample_y*k_h,sample_x*k_w)
        for c in range(0,sample_c):
            for y in range(0,sample_y):
                for x in range(0,sample_x):
                    filter_count = c*sample_y*sample_x + y*sample_x + x
                    fm_ext[c*in_channels:(c+1)*in_channels,y*k_h:(y+1)*k_h,x*k_w:(x+1)*k_w] = conv_weight[filter_count,:,:,:]


        #pdb.set_trace()
        for oc in range(0,sample_c):
            for c in range(1,sample_c):
                idx = sample_c-c+oc
                if idx > sample_c-1:
                   idx = 0
                fm_ext[oc*stride_c:(oc+1)*stride_c,:,:] += \
                   fm_ext[idx*stride_c:(idx+1)*stride_c,:,:]

        fm_ext = fm_ext[0:in_channels,:,:]/sample_c

        fm_ext_h = fm_ext.size()[1]
        fm_ext_w = fm_ext.size()[2]

        for y in range(0,sample_y):
            fm_ext[:,y*k_h,:] += fm_ext[:,(y*k_h-1+fm_ext_h)%fm_ext_h,:]
            fm_ext[:,y*k_h,:] = fm_ext[:,y*k_h,:]/2

        for x in range(0,sample_x):
            fm_ext[:,:,x*k_w] += fm_ext[:,:,(x*k_w-1+fm_ext_w)%fm_ext_w] 
            fm_ext[:,:,x*k_w] = fm_ext[:,:,x*k_w]/2

        y_ids = torch.Tensor([0,1])
        y_ids0 = y_ids
        for y in range(1,sample_y):
            y_ids = torch.cat((y_ids,y_ids0+y*k_h),0)
    
        x_ids = torch.Tensor([0,1])
        x_ids0 = x_ids
        for x in range(1,sample_x):
            x_ids = torch.cat((x_ids,x_ids0+x*k_w),0)

        fm = torch.index_select(fm_ext,1,y_ids.long())
        fm = torch.index_select(fm,2,x_ids.long())
    
    
        self.filtermap.data = fm


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_filtermap(in_planes, out_planes, binary_filtermap = False, stride=1):
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return Conv2d_filtermap(in_planes, out_planes, kernel_size=3, \
                            binary_filtermap=binary_filtermap, stride=stride, padding=1, bias=False)

def conv1x1_filtermap(in_planes, out_planes, channel_compression_1x1, binary_filtermap = False, stride=1):
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return Conv2d_filtermap(in_planes, out_planes, channel_compression_1x1, kernel_size=1, \
                            binary_filtermap=binary_filtermap, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

class BasicBlock_filtermap(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, binary_filtermap, stride=1, downsample=None):
        super(BasicBlock_filtermap, self).__init__()
        self.conv1 = conv3x3_filtermap(inplanes, planes, binary_filtermap, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_filtermap(planes, planes, binary_filtermap)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_filtermap(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, binary_filtermap, stride=1, downsample=None):
        super(Bottleneck_filtermap, self).__init__()
        self.conv1 = Conv2d_filtermap(inplanes, planes, kernel_size=1, \
                                      binary_filtermap=binary_filtermap, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_filtermap(planes, planes, kernel_size=3, \
                                      binary_filtermap=binary_filtermap, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_filtermap(planes, planes * 4, kernel_size=1, \
                                      binary_filtermap=binary_filtermap, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

        out += residual
        out = self.relu(out)

        return out

class ResNet_imagenet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_imagenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_imagenet_filtermap(nn.Module):

    def __init__(self, block, layers, binary_filtermap, num_classes=1000):
        self.inplanes = 64
        super(ResNet_imagenet_filtermap, self).__init__()
        self.conv1 = Conv2d_filtermap(3, 64, kernel_size=7, \
                                      binary_filtermap=binary_filtermap, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],  binary_filtermap)
        self.layer2 = self._make_layer(block, 128, layers[1], binary_filtermap, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], binary_filtermap, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], binary_filtermap, stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Conv2d_filtermap):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.filtermap.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, binary_filtermap, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d_filtermap(self.inplanes, planes * block.expansion, kernel_size=1, \
                          binary_filtermap=binary_filtermap, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, binary_filtermap, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, binary_filtermap))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18_imagenet(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_imagenet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def ResNet34_imagenet(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_imagenet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def ResNet50_imagenet(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_imagenet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def ResNet101_imagenet(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_imagenet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def ResNet152_imagenet(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_imagenet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



def ResNet_filtermap18_imagenet(channel_compression_1x1, binary_filtermap):
    return ResNet_imagenet_filtermap(BasicBlock_filtermap, [2, 2, 2, 2], binary_filtermap)


def ResNet_filtermap34_imagenet(channel_compression_1x1, binary_filtermap):
    return ResNet_imagenet_filtermap(BasicBlock_filtermap, [3, 4, 6, 3], binary_filtermap)


def ResNet_filtermap50_imagenet(channel_compression_1x1, binary_filtermap):
    return ResNet_imagenet_filtermap(Bottleneck_filtermap, [3, 4, 6, 3], binary_filtermap)

def ResNet_filtermap101_imagenet(channel_compression_1x1, binary_filtermap):
    return ResNet_imagenet_filtermap(Bottleneck_filtermap, [3, 4, 23, 3], binary_filtermap)


def ResNet_filtermap152_imagenet(channel_compression_1x1, binary_filtermap):
    return ResNet_imagenet_filtermap(Bottleneck_filtermap, [3, 8, 36, 3], binary_filtermap)
