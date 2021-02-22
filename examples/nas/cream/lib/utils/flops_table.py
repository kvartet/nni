# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import torch

from ptflops import get_model_complexity_info


class FlopsEst(object):
    def __init__(self, model, input_shape=(2, 3, 224, 224), device='cpu'):
        self.block_num = len(model.blocks)
        self.choice_num = len(model.blocks[0])
        self.flops_dict = {}
        self.params_dict = {}

        if device == 'cpu':
            model = model.cpu()
        else:
            model = model.cuda()

        self.params_fixed = 0
        self.flops_fixed = 0

        input = torch.randn(input_shape)


        flops, params = get_model_complexity_info(
            model.conv_stem, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
        print('stem flops {}'.format(flops))
        self.params_fixed += params / 1.0
        self.flops_fixed += flops / 1.0

        input = model.conv_stem(input)


        for block_id, block in enumerate(model.blocks):
            self.flops_dict[block_id] = {}
            self.params_dict[block_id] = {}
            for module_id, module in enumerate(block):
                flops, params = get_model_complexity_info(module, tuple(
                    input.shape[1:]), as_strings=False, print_per_layer_stat=False)
                # print('input shape {}, block_id {}, module_id {}, flops {}'.format(tuple(
                    # input.shape[1:]),block_id,module_id,flops))
                # with open('raw.csv','a+') as f:
                    # f.write(str(int(flops))+'\n')
                # Flops(M)
                self.flops_dict[block_id][module_id] = flops / 1.0
                # Params(M)
                self.params_dict[block_id][module_id] = params / 1.0

            input = module(input)

        # globalpool
        flops, params = get_model_complexity_info(model.global_pool, tuple(
            input.shape[1:]), as_strings=False, print_per_layer_stat=False)
        print('conv_last flops {}'.format(flops))
        self.params_fixed += params / 1.0
        self.flops_fixed += flops / 1.0

        input = model.global_pool(input)

        # conv_last
        flops, params = get_model_complexity_info(model.conv_head, tuple(
            input.shape[1:]), as_strings=False, print_per_layer_stat=False)
        self.params_fixed += params / 1.0
        self.flops_fixed += flops / 1.0
        print('globalpool flops {}'.format(flops))
        # print('flops dict {} '.format(self.flops_dict))
        f = self.flops_fixed
        for i in range(23):
            if len(self.flops_dict[i]) < 6:
                f += self.flops_dict[i][0]
            else:
                f += self.flops_dict[i][5]
        print(f)


    # return params (M)
    def get_params(self, arch):
        params = 0
        for block_id, block in enumerate(arch):
            if block == -1:
                continue
            params += self.params_dict[block_id][block]
        return params + self.params_fixed

    # return flops (M)
    def get_flops(self, arch):
        flops = 0
        for block_id, block in enumerate(arch):
            if block == 'LayerChoice1' or block_id == 'LayerChoice23':
                continue
            for idx, choice in enumerate(arch[block]):
                flops += self.flops_dict[block_id][idx] * (1 if choice else 0)
        return flops + self.flops_fixed
