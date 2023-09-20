# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
model architecture of densenet named as "Status Discriminator" in paper
"""

import math
from collections import OrderedDict

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common import initializer as init
from utils import default_recurisive_init, KaimingNormal
import mindspore.numpy as np
from mindspore import Parameter, Tensor
import mindspore.ops as ops

__all__ = ["DenseNet21plusTransformer_reg", ]






class GlobalAvgPooling(nn.Cell):
    """
    GlobalAvgPooling function.
    """
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(True)
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, x):
        # x = self.mean(x, (2, 3, 4))
        b, c, _, _, _ = self.shape(x)
        x = self.reshape(x, (c, b))
        return x

class CommonHead(nn.Cell):
    def __init__(self, num_classes, out_channels):
        super(CommonHead, self).__init__()
        self.avgpool = GlobalAvgPooling()
        self.fc = nn.Dense(out_channels, num_classes, has_bias=True)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        return x

def conv7x7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False):
    return nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv3x3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv1x1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


class _DenseLayer(nn.Cell):
    """
    the dense layer, include 2 conv layer
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = conv1x1x1(num_input_features, bn_size*growth_rate)

        self.norm2 = nn.BatchNorm3d(bn_size*growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3x3(bn_size*growth_rate, growth_rate)

        # nn.Dropout in MindSpore use keep_prob, diff from Pytorch
        self.p = 1.0 - drop_rate
        self.dropout = nn.Dropout(self.p)

    def construct(self, features):
        bottleneck = self.conv1(self.relu1(self.norm1(features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.p < 1:
            new_features = self.dropout(new_features)
        return new_features

class _DenseBlock(nn.Cell):
    """
    the dense block
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self.cell_list = nn.CellList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.cell_list.append(layer)

        self.concate = P.Concat(axis=1)

    def construct(self, init_features):
        features = init_features
        for layer in self.cell_list:
            new_features = layer(features)
            features = self.concate((features, new_features))
        return features

class _Transition(nn.Cell):
    """
    the transition layer
    """
    def __init__(self, num_input_features, num_output_features, avgpool=False):
        super(_Transition, self).__init__()
        # if avgpool:
        #     poollayer = P.AvgPool3D(kernel_size=2, strides=2)
        # else:
        #     poollayer = P.MaxPool3D(kernel_size=2, strides=2)
        self.avgpool = P.AvgPool3D(kernel_size=2, strides=2)

        self.features = nn.SequentialCell(OrderedDict([
            ('norm', nn.BatchNorm3d(num_input_features)),
            ('relu', nn.ReLU()),
            ('conv', conv1x1x1(num_input_features, num_output_features)),
            # ('poollayer', P.AvgPool3D(kernel_size=2, strides=2))
        ]))


    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class MHSA(nn.Cell):
    def __init__(self, C=62, width=4, height=5, axisz=4):
        super(MHSA, self).__init__()

        self.query = nn.Dense(C*height*width*axisz, C*height*width*axisz, has_bias=True)
        self.key = nn.Dense(C*height*width*axisz, C*height*width*axisz, has_bias=True)
        self.value = nn.Dense(C*height*width*axisz, C*height*width*axisz, has_bias=True)

        self.rel_h = ops.randn([C, height, 1, 1])
        self.rel_w = ops.randn([C, 1, width, 1])
        self.rel_z = ops.randn([C, 1, 1, axisz]) 
        self.softmax = nn.Softmax()

    def construct(self, x):
        n_batch, C, width, height, axisz = x.shape

        q = self.query(x.view(n_batch, -1)).view(n_batch, C, -1)
        k = self.key(x.view(n_batch, -1)).view(n_batch, C, -1)
        v = self.value(x.view(n_batch, -1)).view(n_batch, C, -1)

        content_content = ops.bmm(q.permute(0, 2, 1), k)

        if n_batch == 1:
            content_position = (self.rel_h + self.rel_w + self.rel_z).view(C, -1).reshape(n_batch, C, -1)  
        else:
            temp = (self.rel_h + self.rel_w + self.rel_z).view(C, -1).reshape(1, C, -1)
            content_position = (self.rel_h + self.rel_w + self.rel_z).view(C, -1).reshape(1, C, -1)
            for cat_time in range(n_batch-1):
                content_position = ops.cat((content_position, temp), axis=0)

        content_position = ops.matmul(q.permute(0, 2, 1), content_position)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = ops.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height, axisz)

        return out
    
class Densenet(nn.Cell):
    """
    the densenet architecture
    """
    __constants__ = ['features']

    def __init__(self, growth_rate, block_config, num_init_features=None, bn_size=4, drop_rate=0):
        super(Densenet, self).__init__()

        self.maxpool = P.MaxPool3D(kernel_size=3, strides=2,)
        self.maxpool_end = P.MaxPool3D(kernel_size=(4, 5, 4), strides=2,)

        layers_init = OrderedDict()
        if num_init_features:
            layers_init['conv0'] = conv3x3x3(1, num_init_features, stride=1, padding=1)
            layers_init['norm0'] = nn.BatchNorm3d(num_init_features)
            layers_init['relu0'] = nn.ReLU()
            num_features = num_init_features
        else:
            layers_init['conv0'] = conv3x3x3(3, growth_rate*2, stride=1, padding=1)
            layers_init['norm0'] = nn.BatchNorm3d(growth_rate*2)
            layers_init['relu0'] = nn.ReLU()
            num_features = growth_rate * 2

        layers = OrderedDict()
        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            layers['denseblock%d'%(i+1)] = block
            num_features = num_features + num_layers*growth_rate

            if i != len(block_config)-1:
                if num_init_features:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                        avgpool=True)# Mark This
                else:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                        avgpool=True)
                layers['transition%d'%(i+1)] = trans
                num_features = num_features // 2


        # Transformer
        self.mapping = nn.SequentialCell(MHSA(num_features, 4, 5, 4), )


        # Final batch norm
        layers['norm5'] = nn.BatchNorm3d(num_features)
        layers['relu5'] = nn.ReLU()

        self.features_init = nn.SequentialCell(layers_init)
        self.features = nn.SequentialCell(layers)

        self.classifier = nn.Dense(num_features, 2, has_bias=True)

        self.out_channels = num_features


    def construct(self, x):
        x = self.features_init(x)
        x = self.maxpool(x)
        x = self.features(x)

        mappings = self.mapping(x) #Transformer
        
        x = self.maxpool_end(mappings)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return mappings

    def get_out_channels(self):
        return self.out_channels


def DENSENET21plusTRANSFORMER(**kwargs):
    return Densenet(growth_rate=16, block_config=(2, 2, 2, 2), num_init_features=16, **kwargs)


class DenseNet21plusTransformer_reg(nn.Cell):
    """
    the densenet21 architecture with Transformer
    """
    def __init__(self, num_classes, include_top=False):
        super(DenseNet21plusTransformer_reg, self).__init__()
        self.backbone = DENSENET21plusTRANSFORMER()
        out_channels = self.backbone.get_out_channels()
        self.include_top = include_top
        if self.include_top:
            self.head = CommonHead(num_classes, out_channels)

        '''
        default_recurisive_init(self)
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv3d):
                cell.weight.set_data(init.initializer(KaimingNormal(a=math.sqrt(5), mode='fan_out',
                                                                    nonDenseity='leaky_relu'),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
            # elif isinstance(cell, nn.BatchNorm3d):
            #     cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            #     cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
            elif isinstance(cell, nn.Dense):
                cell.bias.set_data(init.initializer('zeros', cell.bias.shape))
        '''

    def construct(self, x):
        x = self.backbone(x)
        if not self.include_top:
            return x
        x = self.head(x)
        return x
    