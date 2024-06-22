import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
eps = torch.finfo(float).eps
torch.manual_seed(2)
np.random.seed(2)
cfg = {}

class SpecAug(nn.Module):
    def __init__(self, time_mask_max_len=10, time_mask_num=1, freq_mask_max_len=15, freq_mask_num=1, p=1):
        super().__init__()
        self._time_mask_max_len = time_mask_max_len
        self._time_mask_num = time_mask_num

        self._freq_mask_max_len = freq_mask_max_len
        self._freq_mask_num = freq_mask_num

        self._p = p

        self.requires_grad_ = False
    
    def forward(self, x):
        '''
        input: 
            x (Tensor): feature channels, size (batch x channel x time x freq)
        output:
            y (Tensor): T-F masked x, same size as x
        '''
        if np.random.rand() > self._p:
            return x
        transform = transforms.SpecAugment(n_freq_masks=self._freq_mask_num, freq_mask_param=self._freq_mask_max_len,\
                                           n_time_masks=self._time_mask_num, time_mask_param=self._time_mask_max_len, p=0.15)
        # x[:,4:,:,:] = transform(x.permute(0,1,3,2)).permute(0,1,3,2)
        return transform(x.permute(0,1,3,2)).permute(0,1,3,2)

class RandomCutoff(nn.Module):
    def __init__(self, time_mask_max_len=10, time_mask_step=40, freq_mask_max_len=35, p=0.3):
        super().__init__()
        self._time_mask_step = time_mask_step
        self._time_mask_max_len = time_mask_max_len
        self._freq_mask_max_len = freq_mask_max_len
        self._p = p

        self.requires_grad_ = False

    def forward(self, x):
        '''
        input: 
            x (Tensor): feature channels, size (channel x time x freq)
        output:
            y (Tensor): T-F masked x, same size as x, last 3 channel is not masked
        '''
        if np.random.rand() > self._p:
            return x

        nb_mels = x.shape[3]
        for channel in range(x.shape[1]-3):
            for time in range (int(x.shape[2]//self._time_mask_step)):
                time_mask_len = torch.randint(low=0, high=self._time_mask_max_len,size=(1,))[0]
                time_mask_start = torch.randint(low=0, high=self._time_mask_step - time_mask_len,size=(1,))[0]
                freq_mask_len = torch.randint(low=0, high=self._freq_mask_max_len, size=(1,))[0]
                freq_mask_start = torch.randint(low=0, high=nb_mels-torch.max(freq_mask_len), size=(1,))[0]
                x[:, channel, self._time_mask_step*time + time_mask_start: self._time_mask_step*time + time_mask_start + time_mask_len, freq_mask_start:freq_mask_start+freq_mask_len] = np.log(eps)

        return x

class AudioChannelSwapping(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()

        self._p = p
        self.requires_grad_ = False

    def forward(self, x, gt_list, format = 'foa'):
        '''
        input:
            x (Tensor): features channels, size (batch x [w,y,z,x] x time x freq)
            gt_list: ground truth label of x, tensor size (batch,frame,track,[x,y,z])
            format: audio format
        output:
            y (Tensor): swapped channels, same size as x
            y_gt_list: new ground truth label, same size as gt_list
        '''
        
        y = x
        y_gt_list = gt_list
        if format == 'foa':
            rot_azi = torch.randint(0, 8, (1,))[0]
            if rot_azi == 0:
                y[:,[1, 3, 4, 6],:,:] = x[:,[3, 1, 6, 4],:,:] # swap channels
                y[:,[1, 4],:,:] *= -1

                y_gt_list[:,:,:,[0,1]] = gt_list[:,:,:,[1,0]]
                y_gt_list[:,:,:,1] *= -1
            elif rot_azi == 1:
                pass
            elif rot_azi == 2:
                y[:,[1, 3, 4, 6],:,:] = x[:,[3, 1, 6, 4],:,:]
                y[:,[3,6],:,:] *= -1

                y_gt_list[:,:,:,[0,1]] = gt_list[:,:,:,[1,0]]
                y_gt_list[:,:,:,0] *= -1
            elif rot_azi == 3:
                y[:,[1, 3, 4, 6],:,:] = -x[:,[1, 3, 4, 6],:,:]

                y_gt_list[:,:,:,[0,1]] = -gt_list[:,:,:,[0,1]] 
            elif rot_azi == 4:
                y[:,[1, 3, 4, 6],:,:] = -x[:,[3, 1, 6, 4],:,:]

                y_gt_list[:,:,:,[0,1]] = -gt_list[:,:,:,[1,0]]
            elif rot_azi == 5:
                y[:,[1, 4],:,:] = -x[:,[1, 4],:,:]

                y_gt_list[:,:,:,1] = -gt_list[:,:,:,1]  
            elif rot_azi == 6:
                y[:,[1, 3, 4, 6],:,:] = x[:,[3, 1, 6, 4],:,:]

                y_gt_list[:,:,:,[0,1]] = gt_list[:,:,:,[1,0]]  
            elif rot_azi == 7:
                y[:,[3, 6],:,:] = -x[:,[3, 6],:,:]

                y_gt_list[:,:,0] = -gt_list[:,:,0]  

            rot_ele = torch.randint(0, 2, (1,))[0]
            if rot_ele == 0:
                pass
            elif rot_ele == 1:
                y[:,[2, 5],:,:] = -x[:,[2, 5],:,:]

                y_gt_list[:,:,:,2] = -gt_list[:,:,:,2]
        # elif format == 'mic':
            # rot = torch.randint(0, 8, (1,))[0]
            # if rot == 0:
            #     y = x[[2, 4, 1, 3],:]
            #     y_gt_list[:,:,3] -= 90
            #     y_gt_list[:,4] *= -1 
            # elif rot == 1:
            #     y = x[[4, 2, 3, 1],:]
            #     y_gt_list[:,3] = -gt_list[:,3]-90
            # elif rot == 2:
            #     pass
            # elif rot == 3:
            #     y = x[[2, 1, 4, 3],:]
            #     y_gt_list[:,[3,4]] *= -1
            # elif rot == 4:
            #     y = x[[3, 1, 4, 2],:]
            #     y_gt_list[:,3] += 90
            #     y_gt_list[:,4] *= -1
            # elif rot == 5:
            #     y = x[[1, 3, 2, 4],:]
            #     y_gt_list[:,3] = -gt_list[:,3]+90
            # elif rot == 6:
            #     y = x[[4, 3, 2, 1],:]
            #     y_gt_list[:,3] += 90
            # elif rot == 7:
            #     y = x[[3, 4, 1, 2],:]
            #     y_gt_list[:,[3,4]] *= -1
            #     y_gt_list[:,3] += 180
        else:
            raise NotImplementedError('This format is not supported')
        return y, y_gt_list

class FrequencyShifting(nn.Module):
    def __init__(self, freq_band_shift_range=10, p=1):
        super().__init__()
        self._shift_range = freq_band_shift_range
        self._p = p

        self.requires_grad_ = False

    def forward(self, x):
        '''
        input:
            x (Tensor): features channels, size (channel x time x freq)
        output:
            y (Tensor): shifted frequency channels
        ''' 
        if np.random.rand() > self._p:
            return x

        y = x
        shift_len = np.random.choice(self._shift_range)
        dir = np.random.choice(['up', 'down'])
        if dir == 'up':
            y = torch.roll(y, shift_len, dims=2)
            y[:,:,0:shift_len] = 0
        else:
            y = torch.roll(y, -shift_len, dims=2)
            y[:,:,-shift_len:] = 0

        return y









                








