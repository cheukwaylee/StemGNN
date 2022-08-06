import torch
import torch.nn as nn
import torch.nn.functional as F

# 门控单元paper: Language Modeling with Gated Convolutional Networks
# 作用： 1. 序列深度建模； 2. 减轻梯度弥散，减速收敛
class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        # 对应位置的乘法？
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,  # 1-4-1-12*5-5*12
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)

        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)  # 5*12-12*5
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)  # 12*5-12

        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)  # 12*5-12
        # 数据原特征的表达
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)  # 12-12

        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:  # +2个GLU
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:  # +2个GLU
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:  # +2个GLU --> 共计6个
                # , 12*（4*5）
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input): # input (32-4-1)-140-12
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)

        # FFT 快速离散傅里叶变换； 
        # RFFT 去除那些共轭对称的值，减小储存
        ffted = torch.rfft(input, 1, onesided=False)  # 32-4-140-12-2
        # 4*12 放在最后一个维度
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)

        # GLU 实部虚部分别经过三个GLU
        for i in range(3):
            real = self.GLUs[i * 2](real)  # 0 2 4
            img = self.GLUs[2 * i + 1](img)  # 1 3 5
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()  # 32-4-140-60
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous() 
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)  # 32-4-140-60-2

        # IDFT
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)  # 32-4-140-60
        return iffted

    def forward(self, x, mul_L):
        # x: 32-1-140-12
        # mul_L: 4-140-140
        mul_L = mul_L.unsqueeze(1)  # 4-1-140-140
        x = x.unsqueeze(1)  # 32-1-1-140-12

        # learning latent representations of multiple time-series in the spectral domain
        # 支持广播机制？自动可以相乘？ 
        # mul_L: 4-1-140-140 --> (32-4-1)-140-140
        # x    : 32-1-1-140-12 --> (32-4-1)-140-12
        gfted = torch.matmul(mul_L, x)  # (32-4-1)-140-12

        # captures the repeated patterns in the periodic data
        # or the auto-correlation features among different timestamps
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)  # 32-4-1-140-60

        # GConv + IGFT
        # weight: 1-4-1-60-60
        igfted = torch.matmul(gconv_input, self.weight)  # 32-4-1-140-60 广播机制
        igfted = torch.sum(igfted, dim=1)  # 32-1-140-60

        # forecast： 全连接层 using "igfted" as input
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))  # 32-140-60
        forecast = self.forecast_result(forecast_source)  # 32-140-12 最终要预测的是12d输出

        # backcast： using "x" 32-1-1-140-12 as input 
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)  # 32-1-140-12
            # x_hat - x
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)  # 32-1-140-12
        else:
            backcast_source = None
        return forecast, backcast_source


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        self.unit = units  # 输出的维度==140
        self.stack_cnt = stack_cnt  # StemGNN的block的个数==2
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step  # Window size窗口的大小==12 length of input sequence
        self.horizon = horizon  # 预测的长度 1表示未来的一天 H predictions in the future, after time t

        # self attention
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)  # 初始化方法
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.GRU = nn.GRU(self.time_step, self.unit)  # dim of inputs==12, and dim of outputs==140
        self.multi_layer = multi_layer  # 层数

        # 1. block参数将自动放到主模型中
        # 2. 没有顺序性要求
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(  # stack_cnt标记现在是第几个StemGNN
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N] 140-140
        # 在0维度扩张
        laplacian = laplacian.unsqueeze(0)  # 1-140-140
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)  # 1-140-140
        second_laplacian = laplacian  # 1-140-140
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian  # 1-140-140
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian  # 1-140-140
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian  # 4-140-140

    def latent_correlation_layer(self, x):
        # 把输入x数据的排列转化为GRU构造函数能接受的形式
        # x: (batch, sequence, feature) 32-12-140 originally from data_loader
        # the required GRU default input: (sequence, batch, feature), when default batch_first==False

        # however, here after permute() it is (feature, batch, sequence) 140-32-12, NOT match!
        # expected 12-140? switch feature and sequence?
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        # 140-32-140? due to fefinition of GRU? idk

        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)  # 32-140-140
        attention = torch.mean(attention, dim=0)  # 140-140 

        degree = torch.sum(attention, dim=1)  # 140

        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)  # 对称的laplace
        degree_l = torch.diag(degree)  # 对角线为degree 140-140
        # D^-0.5
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))  # 140-140
        # D^-0.5 * A * D^-0.5
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))  # 140-140
        # 多阶切比雪夫的拉普拉斯
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        # input shape here: (batch, sequence, output_size)
        input = input.permute(0, 2, 1).contiguous()
        # after trans: (batch, output_size, sequence)
        # this is why input==output??
        bat, N, fea = input.size()

        # 32-140-140 x 140-1 --> 32-140-1 key.shape()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)

        # torch.repeat 当参数有三个的时候 （通道数的重复倍数，行的重复倍数，列的）
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        # data shape: 32-140*140-1 

        data = data.squeeze(2)  # 去掉最后一维
        data = data.view(bat, N, -1)  # 展开？ 32-140-140？
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention  # 32-140-140

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        # part 1
        mul_L, attention = self.latent_correlation_layer(x)  # 多阶laplace 4-140-140
        # x:  (batch, sequence, feature) 
        # --> (batch, 1, sequence, feature) 
        # --> (batch, 1, feature, sequence)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()  # 32-1-140-12

        # part 2
        result = []
        for stack_i in range(self.stack_cnt): # first(with backforst) and second(without)
            # pass to StemGNN
            forecast, X = self.stock_block[stack_i](X, mul_L)
            # output X: backcast = X_hat - X, 32-1-140-12
            result.append(forecast)
        # 两层的forecast相加Y1+Y2
        forecast = result[0] + result[1]  # 32-140-12
        forecast = self.fc(forecast)  # 经过全连接层

        # Horizon = 1
        if forecast.size()[-1] == 1:
            # 32-140-1 --> 32-1-140-1 --> 32-1-140
            return forecast.unsqueeze(1).squeeze(-1), attention
        # example Horizon = 1
        else:
            # 32-140-3 --> 32-3-140
            return forecast.permute(0, 2, 1).contiguous(), attention
