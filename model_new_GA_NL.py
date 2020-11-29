# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from gtad_lib.align import Align1DLayer
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# dynamic graph from knn
def knn(x, y=None, k=10):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    if y is None:
        y = x
    # logging.info('Size in KNN: {} - {}'.format(x.size(), y.size()))
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return idx


# get graph feature
def get_graph_feature(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn


# basic block
class GCNeXt(nn.Module):
    def __init__(self, channel_in, channel_out, k=3, norm_layer=None, groups=32, width_group=4, idx=None):
        super(GCNeXt, self).__init__()
        self.k = k
        self.groups = groups

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups
        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        ) # temporal graph

        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        ) # semantic graph

        self.relu = nn.ReLU(True)
        self.idx_list = idx

    def forward(self, x):
        identity = x  # residual
        tout = self.tconvs(x)  # conv on temporal graph

        x_f, idx = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)

        out = tout + identity + sout  # fusion
        if not self.idx_list is None:
            self.idx_list.append(idx)
        return self.relu(out)


class GraphAlign(nn.Module):
    def __init__(self, k=3, t=100, d=100, bs=64, samp=0, style=0):
        super(GraphAlign, self).__init__()
        self.k = k
        self.t = t
        self.d = d
        self.bs = bs
        self.style = style
        self.expand_ratio = 0.5
        self.resolution = 32
        self.align_inner = Align1DLayer(self.resolution, samp)
        self.align_context = Align1DLayer(4)
        self._get_anchors()

    def forward(self, x, index):
        bs, ch, t = x.shape
        if not self.anchors.is_cuda:  # run once
            self.anchors = self.anchors.cuda()

        anchor = self.anchors[:self.anchor_num * bs, :]  # (bs*tscale*tscal, 3)
        # print('first value in anchor is', anchor[0])
        feat_inner = self.align_inner(x, anchor)  # (bs*tscale*tscal, ch, resolution)
        if self.style == 1: # use last layer neighbours
            feat, _ = get_graph_feature(x, k=self.k, style=2)  # (bs,ch,100) -> (bs, ch, 100, k)
            feat = feat.mean(dim=-1, keepdim=False)  # (bs. 2*ch, 100)
            feat_context = self.align_context(feat, anchor)  # (bs*tscale*tscal, ch, resolution//2)
            feat = torch.cat((feat_inner,feat_context), dim=2).view(bs, t, t, -1)
        elif self.style == 2: # use all layers neighbour
            feat, _ = get_graph_feature(x, k=self.k, style=2, idx_knn=index)  # (bs,ch,100) -> (bs, ch, 100, k)
            feat = feat.mean(dim=-1, keepdim=False)  # (bs. 2*ch, 100)
            feat_context = self.align_context(feat, anchor)  # (bs*tscale*tscal, ch, resolution//2)
            feat = torch.cat((feat_inner,feat_context), dim=2).view(bs, t, t, -1)
        else:
            feat = torch.cat((feat_inner,), dim=2).view(bs, t, t, -1)
        # print('shape after align is', feat_context.shape)

        return feat.permute(0, 3, 2, 1)  # (bs,2*ch*(-1),t,t)

    def _get_anchors(self):
        anchors = []
        for k in range(self.bs):
            for start_index in range(self.t):
                for duration_index in range(self.d):
                    if start_index + duration_index < self.t:
                        p_xmin = start_index
                        p_xmax = start_index + duration_index
                        center_len = float(p_xmax - p_xmin) + 1
                        sample_xmin = p_xmin - center_len * self.expand_ratio
                        sample_xmax = p_xmax + center_len * self.expand_ratio
                        anchors.append([k, sample_xmin, sample_xmax])
                    else:
                        anchors.append([k, 0, 0])
        self.anchor_num = len(anchors) // self.bs
        self.anchors = torch.tensor(np.stack(anchors)).float()  # save to cpu
        return  # anchors, anchor_num


class GTAD(nn.Module):
    def __init__(self, opt,G):
        super(GTAD, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.feat_dim = opt["feat_dim"]
        self.bs = opt["batch_size"]
        self.h_dim_1d = opt["h_dim_1d"]
        self.h_dim_2d = opt["h_dim_2d"]
        self.h_dim_3d = opt["h_dim_3d"]
        self.goi_style = opt['goi_style']
        self.h_dim_goi = self.h_dim_1d*(32,32+4,32+4)[opt['goi_style']]
        self.idx_list = []

        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list),
        )

        # Regularization
        self.regu_s = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32),
            nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        )
        self.regu_e = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32),
            nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        )

        # Backbone Part 2
        self.backbone2 = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32,idx=self.idx_list),
        )

        # SGAlign: sub-graph of interest alignment
        self.goi_align = GraphAlign(
            t=self.tscale, d=opt['max_duration'], bs=self.bs,
            samp=opt['goi_samp'], style=opt['goi_style']  # for ablation
        )

        # Localization Module
        self.localization = nn.Sequential(
            nn.Conv2d(self.h_dim_goi, self.h_dim_3d, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_3d, self.h_dim_2d, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, self.h_dim_2d, kernel_size=opt['kern_2d'], padding=opt['pad_2d']), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, self.h_dim_2d, kernel_size=opt['kern_2d'], padding=opt['pad_2d']), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, 2, kernel_size=1), nn.Sigmoid()
        )

        self.att_net = NONLocalBlock1D(in_channels=400,G=G)


    def forward(self, snip_feature):
        del self.idx_list[:]  # clean the idx list
        # snip_feature = self.pos(snip_feature)
        snip_feature = self.att_net(snip_feature)
        base_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        #base_feature = self.att_net(base_feature)
        gcnext_feature = self.backbone2(base_feature)  #

        regu_s = self.regu_s(base_feature).squeeze(1)  # start
        regu_e = self.regu_e(base_feature).squeeze(1)  # end

        if self.goi_style==2:
            idx_list = [idx for idx in self.idx_list if idx.device == snip_feature.device]
            idx_list = torch.cat(idx_list, dim=2)
        else:
            idx_list = None

        subgraph_map = self.goi_align(gcnext_feature, idx_list)
        iou_map = self.localization(subgraph_map)
        return iou_map, regu_s, regu_e

    def get_optim_policies(self):

        main_net = []
        att_net = []

        for m in self.att_net.modules():
            att_net.extend(m.parameters())
           # print(m, 'att_net')

        for i, m in enumerate(self.modules()):
            if i != 0:
                if m not in self.att_net.modules():
                    if not isinstance(m, NONLocalBlock1D):
                        main_net.extend(m.parameters())
              #          print(m, 'main_net')

        return [
            {'params': att_net, 'lr_mult': 1, 'decay_mult': 1,
             'name': "att_net_weight"},
            {'params': main_net, 'lr_mult': 1, 'decay_mult': 1,
             'name': "main_net_weight"}]


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels,G,  dimension=1, mode='embedded_gaussian',
                 sub_sample=True):
        super(_NonLocalBlockND,self).__init__()

        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.G = G  # (K,D)(256,100)
        self.G_f = nn.Sequential(nn.Linear(400,100),nn.PReLU())
        self.A = nn.Parameter(torch.FloatTensor(1,512,400)) #nn.Sequential(
            #nn.Conv1d(100,512,kernel_size=1),
            #nn.PReLU(),
          #  nn.Linear(100,400),

           # nn.PReLU()
       # )

        self.w4 = nn.Sequential(
            nn.Linear(200,100),nn.PReLU(),
            nn.Linear(100, 1), nn.PReLU()
        )
       # self.conv2 = nn.Sequential(
       #     nn.Conv1d(400, 400, kernel_size=1), nn.PReLU(),
       #     nn.Conv1d(400, 400, kernel_size=1), nn.PReLU())

        self.W5 = nn.Sequential(
            nn.Linear(100,100),nn.PReLU()
        )
        # nn.init.kaiming_normal(self.W[0].weight)
        # nn.init.constant(self.W[0].bias, 0)
        # nn.init.constant(self.W[1].weight, 0)
        # nn.init.constant(self.W[1].bias, 0)
        # nn.init.kaiming_normal(self.conv1.weight)
        # nn.init.constant(self.conv1.bias, 0)
        # nn.init.kaiming_normal(self.conv2.weight)
        # nn.init.constant(self.conv2.bias, 0)
        # nn.init.kaiming_normal(self.A.weight)
        # nn.init.constant(self.A.bias, 0)

        self.gcn1 = GraphConvolution(512, 100)
        self.gcn2 = GraphConvolution(512, 512)
        self.gcn3 = GraphConvolution(512, 400)
       # self.conv3 = nn.Sequential(nn.Conv1d(256, 400, kernel_size=1), nn.PReLU())

        self.cos = sim  # torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        # self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, x):
        '''
        :param x: (b, 400,100)
        :return:
        '''

        batch_size = x.size(0)
      #  x = x.permute(0,2,1) # x(b,100,400)
        device = x.device
        self.G = self.G.to(device)

        # AX
        #AX = self.A(x)
        self.G1 = self.G_f(self.G.repeat(batch_size, 1, 1))  # (b,512,400)

        self.A1 = self.A.repeat(batch_size,1,1)
        G_AX = torch.matmul(self.G1, torch.matmul(self.A1,x).permute(0,2,1)) #(b,512,512)


        P = F.softmax(G_AX, dim=-1)


        simlarity1 = self.cos(P.detach())


        simlarity = norm(simlarity1)

        # print(simlarity.shape, 'similarity',P.shape,'P')
        P = P.squeeze(0).squeeze(0)
        gx = F.relu(self.gcn1(simlarity, P))
      #  simlarity2 = norm(self.cos(gx1.detach()))
      #  gx2 = self.gcn2(simlarity2, gx1)
     #   simlarity3 = norm(self.cos(gx2.detach()))
     #   gx = self.gcn3(simlarity3, gx2)  ### (b,512,100)
        # print(gx.shape,'gx')
       # gx_tile =
     #   div_gx_x = F.softmax(self.conv1(torch.cat([gx, x], dim=1)), dim=-1)  # (b,912,400)#gx(b,512,100) x (b,400,100)
        gx_tile = gx.unsqueeze(-3).repeat(1,400,1,1)#(b,400,512,100)
        x_tile = x.unsqueeze(-2).repeat(1,1,512,1) #(b,400,512,100)
        div_gx_x = F.softmax(self.w4(torch.cat([x_tile,gx_tile],dim=-1)),dim=-3).squeeze(-1)
        #print(div_gx_x.shape,gx.shape)
        output = self.W5(torch.matmul(div_gx_x, gx))#(b,)
        output = output + x
       # output = output.permute(0,2,1) #(b,400,100)

        return output





def norm(A , symmetric=True):
	# A = A+I
    D = torch.zeros(A.shape).cuda()
    B = torch.zeros(A.shape).cuda()
    for i in range(A.shape[0]):
        B[i]= A[i] + torch.eye(A[i].size(0)).cuda()
        # 所有节点的度
        d = B[i].sum(1)
        if symmetric:
            #D = D^-1/2
            D[i] = torch.diag(torch.pow(d , -0.5)).cuda()
            D[i]= D[i].mm(B[i]).mm(D[i])
        else :
            # D=D^-1
            D[i] =torch.diag(torch.pow(d,-1))
        D[i]=D[i].mm(B[i])
    return D


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, adj, features):

        out1 = torch.matmul(adj, features)
        out = self.linear(out1)
        return out


def sim(x):
    A = x.detach().cpu().numpy()
    distB = torch.zeros((A.shape[0],A.shape[1],A.shape[1]))
    distB = distB.detach().cpu().numpy()

    # A是一个向量矩阵：euclidean代表欧式距离
    for i in range(A.shape[0]):
        distA = pdist(A[i], metric='euclidean')
    # 将distA数组变成一个矩阵
        distB[i] = squareform(distA)

    return torch.tensor(distB).cuda()


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, G):
        super(NONLocalBlock1D, self).__init__(in_channels,G)

if __name__ == '__main__':
    from gtad_lib import opts
    from torchsummary import summary
    opt = opts.parse_opt()
    opt = vars(opt)
    model = GTAD(opt).cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0])
    # input = torch.randn(4, 400, 100).cuda()
    # a, b, c = model(input)
    # print(a.shape, b.shape, c.shape)

    summary(model, (400,100))

    '''
    Total params: 9,495,428
    Trainable params: 9,495,428
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.15
    Forward/backward pass size (MB): 1398.48
    Params size (MB): 36.22
    Estimated Total Size (MB): 1434.85
    '''
