import torch
import torch.nn as nn


def channel_shuffle(x, groups = 2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    
    return x


def conv_1x1_bn(in_channels_, out_channels_, stride_ = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels= in_channels_, out_channels= out_channels_, kernel_size= 1, stride= stride_, padding= 0, bias= False),
        nn.BatchNorm2d(out_channels_),
        nn.ReLU6(inplace= True)
    )


class ShuffleBlock(nn.Module):
    def __init__(self, in_channels_, out_channels_, downsample = False):
        super(ShuffleBlock, self).__init__()
        
        self.downsample = downsample
        half_channels_ = out_channels_ //2
        
        if self.downsample:
            self.branch1 = nn.Sequential(
                # 3*3 DW-Conv, stride = 2
                nn.Conv2d(in_channels= in_channels_, out_channels= in_channels_, kernel_size= 3, stride= 2, padding= 1, groups= in_channels_, bias= False),
                nn.BatchNorm2d(in_channels_),
                
                # 1*1 PW-Conv
                nn.Conv2d(in_channels= in_channels_, out_channels= half_channels_, kernel_size= 1, stride= 1, padding= 0, bias= False),
                nn.BatchNorm2d(half_channels_),
                nn.ReLU6(inplace= True)                
            )
            
            self.branch2= nn.Sequential(
                # 1*1 PW-Conv
                nn.Conv2d(in_channels= in_channels_, out_channels= half_channels_, kernel_size= 1, stride= 1, padding= 0, bias= False),
                nn.BatchNorm2d(half_channels_),
                nn.ReLU6(inplace= True),
                
                # 3*3 DW-Conv, stride = 2
                nn.Conv2d(in_channels= half_channels_, out_channels= half_channels_, kernel_size= 3, stride= 2, padding= 1, groups= half_channels_, bias= False),
                nn.BatchNorm2d(half_channels_),
                
                # 1*1 PW-Conv
                nn.Conv2d(in_channels= half_channels_, out_channels= half_channels_, kernel_size= 1, stride= 1, padding= 0, bias= False),
                nn.BatchNorm2d(half_channels_),
                nn.ReLU6(inplace= True)                
            )
            
        else:
            assert in_channels_  == out_channels_
            
            self.branch2 = nn.Sequential(
                
                # 1*1 pw conv
                nn.Conv2d(in_channels= half_channels_, out_channels= half_channels_, kernel_size= 1, stride= 1, padding= 0, bias= False),
                nn.BatchNorm2d(half_channels_),
                nn.ReLU(inplace= True),
                
                # 3*3 dw conv, stride = 1
                nn.Conv2d(in_channels= half_channels_, out_channels= half_channels_, kernel_size= 3, stride= 1, padding= 1, groups= half_channels_, bias= False),
                nn.BatchNorm2d(half_channels_),
                
                # 1*1 pw conv
                nn.Conv2d(in_channels= half_channels_, out_channels= half_channels_, kernel_size= 1, stride= 1, padding= 0, bias= False),
                nn.BatchNorm2d(half_channels_),
                nn.ReLU6(inplace= True)              
            )
        
    def forward(self, x):
        out = None
        if self.downsample:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)

        else:
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = torch.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)
    
    

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes= 2, input_size= 224, net_type= 1):
        super(ShuffleNetV2, self).__init__()
        assert input_size % 32 == 0
        
        self.stage_repeat_num = [4, 8, 4]
        if net_type == 0.5:
            self.out_channels = [3, 24, 48,  96,  192, 1024]
        elif net_type == 1:
            self.out_channels = [3, 24, 116, 232, 464, 1024]
        elif net_type == 1.5:
            self.out_channels = [3, 24, 176, 352, 704, 1024]
        elif net_type == 2:
            self.out_channels = [3, 24, 244, 488, 976, 2948]
        else:
            print("the type is error")
            
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= self.out_channels[1], kernel_size= 3, stride= 2, padding= 1)
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
    
        in_c = self.out_channels[1]    
        self.stages = []
    
        for stage_idx in range(len(self.stage_repeat_num)):
            out_c = self.out_channels[2 + stage_idx]
            repeat_num = self.stage_repeat_num[stage_idx]

            for i in range(repeat_num):
                if i == 0:
                    self.stages.append(ShuffleBlock(in_c, out_c, downsample = True))
                else:
                    self.stages.append(ShuffleBlock(in_c, out_c, downsample = False))
                in_c = out_c

        self.stages = nn.Sequential(*self.stages)

        in_c = self.out_channels[-2]
        out_c = self.out_channels[-1]

        self.conv5 = conv_1x1_bn(in_c, out_c, 1)
        self.g_avg_pool = nn.AvgPool2d(kernel_size = (int)(input_size/32))
        #self.g_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(in_features= out_c, out_features= num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = self.g_avg_pool(x)
        x = x.view(-1, self.out_channels[-1])
        x = self.fc(x)
        return x       



def load_model(path):
    try:
        checkpoint = torch.load(path, map_location= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        modelnet = ShuffleNetV2()
        modelnet.parameters = checkpoint["parameters"]
        modelnet.load_state_dict(checkpoint["state_dict"])
        modelnet.eval()
        
        mode = "eval" if not modelnet.training else "train"
        print(f" The model mode is " + mode)
        
    except Exception as err:
        print(err)
        return None
       
    return modelnet





















