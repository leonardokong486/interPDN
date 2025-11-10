import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from layers.revin import RevIN

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin_layer = RevIN(configs.enc_in,affine=True,subtract_last=False)

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.fc1 = nn.Linear(self.pred_len * 2, self.pred_len * 25)
        self.fc2 = nn.Linear(self.pred_len * 2, self.pred_len * 25)
        self.fc3 = nn.Linear(self.pred_len * 2, int(self.pred_len * 6.25))
        self.fc4 = nn.Linear(self.pred_len * 2, int(self.pred_len * 6.25))

        df = pd.read_excel('idx2.xlsx', usecols=[1, 2], header=None, skiprows=1, nrows=25)
        self.idx_1 = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(25, 1).cuda()
        self.idx_2 = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(25, 1).cuda()

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

        else:
            self.Linear_Seasonal_1 = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend_1 = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal_2 = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend_2 = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal_3 = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend_3 = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal_4 = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend_4 = nn.Linear(self.seq_len,self.pred_len)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.revin_layer(x, 'norm')
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output_1 = self.Linear_Seasonal_1(seasonal_init)
        trend_output_1 = self.Linear_Trend_1(trend_init) # (32, 7, 96)
        seasonal_output_2 = self.Linear_Seasonal_2(seasonal_init)
        trend_output_2 = self.Linear_Trend_2(trend_init)
        seasonal_output_3 = self.Linear_Seasonal_3(seasonal_init)
        trend_output_3 = self.Linear_Trend_3(trend_init)
        seasonal_output_4 = self.Linear_Seasonal_4(seasonal_init)
        trend_output_4 = self.Linear_Trend_4(trend_init)

        out_1 = torch.cat((seasonal_output_1, trend_output_1), dim=2)
        x_1 = self.fc1(out_1)
        x_1 = x_1.permute(0,2,1)
        x_1 = x_1.reshape(x_1.shape[0], x_1.shape[1] // 25, 25, x_1.shape[2])
        x_cls_1 = torch.softmax(x_1, dim=2)
        output_1 = torch.einsum("blec,ei->blic", x_cls_1, self.idx_1).squeeze(2)

        out_2 = torch.cat((seasonal_output_2, trend_output_2), dim=2)
        x_2 = self.fc2(out_2)
        x_2 = x_2.permute(0,2,1)
        x_2 = x_2.reshape(x_2.shape[0], x_2.shape[1] // 25, 25, x_2.shape[2])
        x_cls_2 = torch.softmax(x_2, dim=2)
        output_2 = torch.einsum("blec,ei->blic", x_cls_2, self.idx_2).squeeze(2)  # (4, 96, 7)

        entro_1 = x_cls_1.permute(0, 3, 1, 2) 
        entro_1 = torch.max(entro_1, dim=-1)[0] 
        entro_2 = x_cls_2.permute(0, 3, 1, 2) 
        entro_2 = torch.max(entro_2, dim=-1)[0]
        mask = entro_1 / (entro_1 + entro_2)
        mask = mask.permute(0, 2, 1)
        output_fine = (output_1 * mask + output_2 * (1 - mask))

        output_down = output_fine.reshape(x_2.shape[0], x_2.shape[1] // 4, 4, x_2.shape[3])
        output_down_fine = output_down.mean(dim=2) 

        out_3 = torch.cat((seasonal_output_3, trend_output_3), dim=2)
        x_3 = self.fc3(out_3)
        x_3 = x_3.permute(0,2,1)     
        x_3 = x_3.reshape(x_3.shape[0], x_3.shape[1] // 25, 25, x_3.shape[2])
        x_cls_3 = torch.softmax(x_3, dim=2)
        output_3 = torch.einsum("blec,ei->blic", x_cls_3, self.idx_1).squeeze(2)
        out_4 = torch.cat((seasonal_output_4, trend_output_4), dim=2)
        x_4 = self.fc4(out_4)
        x_4 = x_4.permute(0,2,1)      
        x_4 = x_4.reshape(x_4.shape[0], x_4.shape[1] // 25, 25, x_4.shape[2])
        x_cls_4 = torch.softmax(x_4, dim=2)
        output_4 = torch.einsum("blec,ei->blic", x_cls_4, self.idx_2).squeeze(2)

        entro_3 = x_cls_3.permute(0, 3, 1, 2) 
        entro_3 = torch.max(entro_3, dim=-1)[0] 
        entro_4 = x_cls_4.permute(0, 3, 1, 2) 
        entro_4 = torch.max(entro_4, dim=-1)[0]
        mask = entro_3 / (entro_3 + entro_4)
        mask = mask.permute(0, 2, 1)
        output_grain = (output_3 * mask + output_4 * (1 - mask))

        con_loss_time = nn.functional.mse_loss(output_grain, output_down_fine)
        con_loss_cls_1 = nn.functional.mse_loss(output_2, output_1)
        cor_loss_cls_2 = nn.functional.mse_loss(output_4, output_3)

        output = self.revin_layer(output_fine, 'denorm')

        return output, con_loss_cls_1, cor_loss_cls_2, con_loss_time
