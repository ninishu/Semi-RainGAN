import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Softmax


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf"), device='cuda').repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Depth-guided Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.y_dowm = nn.Conv2d(1, in_dim, kernel_size=4, stride=2, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                y : depth map
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        x_batchsize, x_C, x_height, x_width = x.size()
        y_down = self.y_dowm(y)
        # y_batchsize, y_C, y_height, y_width = y.size()
        proj_query = y_down.view(x_batchsize, x_C, -1)
        proj_key = x.view(x_batchsize, x_C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(x_batchsize, x_C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(x_batchsize, x_C, x_height, x_width)

        out = self.gamma * out + x
        return out


class CCA(nn.Module):
    """ attention-guided criss-cross module"""
    def __init__(self, in_channels):
        super(CCA, self).__init__()
        self.x_down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, groups=in_channels, bias=False)
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, attention_map):
        ##################### feature
        x_down = self.x_down(x)
        m_batchsize, _, height, width = x_down.size()
        proj_query = self.query_conv(x_down)
        # (B*W, H, c)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        # (B*H, W, c)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x_down)
        # (B*W, c, H)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # (B*H, c, W)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x_down)
        # (B*W, C, H)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # (B*H, C, W)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        # (B*W, H, H) => (B, H, W, H)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        # (B*H, W, W) => (B, H, W, W)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        # (B, H, W, H+W)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        ###################### attention
        attention_map = F.interpolate(attention_map, size=[height, width], mode='bilinear', align_corners=True)
        n, c, h, w = attention_map.size()
        # (B*H, W, c)
        attention_H = attention_map.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h).permute(0, 2, 1)
        # (B*H, W, c)
        attention_W = attention_map.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w).permute(0, 2, 1)

        # (B*W, c, H)
        attention_H2 = attention_map.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)
        # (B*H, c, W)
        attention_W2 = attention_map.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)

        # (B*W, H, H) => (B, H, W, H)
        affinity_H = (torch.bmm(attention_H, attention_H2) + self.INF(n, h, w)).view(n, w, h, h).permute(0, 2, 1, 3)
        # (B*H, W, W) => (B, H, W, W)
        affinity_W = torch.bmm(attention_W, attention_W2).view(n, h, w, w)
        # (B, H, W, H+W)
        concate_a = self.softmax(torch.cat([affinity_H, affinity_W], 3))

        S = self.softmax(concate * concate_a)

        # (B*W, H, H)
        att_H = S[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # (B*H, W, W)
        att_W = S[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        out_H = F.upsample(out_H, size=x.size()[2:], mode='bilinear', align_corners=True)
        out_W = F.upsample(out_W, size=x.size()[2:], mode='bilinear', align_corners=True)
        return self.gamma * (out_H + out_W) + x


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU()
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        return x + conv1


class DRBs(nn.Module):
    def __init__(self, in_channels):
        super(DRBs, self).__init__()
        self.drbs = []
        for i in range(0, 3):
            self.drbs.append(DilatedResidualBlock(in_channels, 2 * i + 1))

        self.drbs_module = nn.ModuleList(self.drbs)
        self.conv = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        output_slices = []

        for module in self.drbs_module:
            out = module(x)
            output_slices.append(out)

        out = torch.cat(output_slices, dim=1)
        out = self.conv(out)

        return out
