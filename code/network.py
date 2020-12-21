import torch
import torch.nn as nn
import torch.nn.functional as F

'''modified from https://github.com/ndrplz/ConvLSTM_pytorch/'''
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, forget_bias=1.0):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.forget_bias = forget_bias # TODO: add or not

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        torch.nn.init.xavier_normal_(self.conv.weight)

    def forward(self, input_tensor, cur_state):
        '''
        :param input_tensor: B x input_dim x height x width
        :param cur_state: B x 2hidden_dim x height x width
        :return: h_next, [h_next, c_next]
        '''
        h_cur, c_cur = torch.split(cur_state, self.hidden_dim, dim=1)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f + self.forget_bias)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, torch.cat([h_next, c_next], dim=1)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class RNNmodule(nn.Module):
    def __init__(self, img_channel, hidden_channel, adj_channel, kernel_size, pixelwise):
        super(RNNmodule, self).__init__()

        self.img_channel = img_channel
        self.adj_channel = adj_channel
        self.pixelwise = pixelwise

        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.in_cnv = nn.Conv2d(img_channel, hidden_channel, kernel_size, stride=1, padding=self.padding)
        self.bn = nn.BatchNorm2d(hidden_channel, momentum=0.01) # TODO
        self.RNNcell = ConvLSTMCell(hidden_channel, hidden_channel, kernel_size)
        self.rnn_cnv = nn.Conv2d(hidden_channel, hidden_channel, kernel_size, stride=1, padding=self.padding)
        if pixelwise:
            self.out_cnv = nn.Conv2d(hidden_channel, adj_channel, kernel_size, stride=1, padding=self.padding)
        else:
            self.out_linear = nn.Linear(hidden_channel, 1)
        # TODO: or kernel size=1
        # TODO: or multi-adjustment using rnn

        self.init_param()

    def init_param(self):
        torch.nn.init.xavier_normal_(self.in_cnv.weight)
        torch.nn.init.xavier_normal_(self.rnn_cnv.weight)
        if self.pixelwise:
            torch.nn.init.xavier_normal_(self.out_cnv.weight)
        self.bn.weight.data.normal_(1.0, 0.02)
        self.bn.bias.data.fill_(0)

    def forward(self, inputs, last_hidden):
        b, c, h, w = inputs.size()
        assert c == self.img_channel

        if last_hidden == None:
            last_hidden = torch.cat(self.RNNcell.init_hidden(b, (h, w)), dim=1)

        conv = self.in_cnv(inputs)
        conv = self.bn(conv)
        conv = F.relu(conv)

        rnn, new_hidden = self.RNNcell(conv, last_hidden)
        rnn = self.rnn_cnv(rnn)
        # rnn = self.bn(rnn)
        rnn = F.relu(rnn)

        if self.pixelwise:
            adjs = F.relu(self.out_cnv(rnn))
            adj_t = torch.split(adjs, 1, dim=1)
            x = inputs
            for i in range(self.adj_channel):
                x = torch.pow(x, 1/adj_t[i]) # adj_t[i] B x 1 x h x w
        else:
            adjs = torch.mean(rnn, dim=[-1, -2])
            adjs = F.relu(self.out_linear(adjs)) # B x 1
            adjs = adjs.view(-1, 1, 1, 1)
            x = torch.pow(inputs, 1/adjs)
        # x_gray = torch.mean(x, dim=1, keepdim=True)
        # xp = adjs[:, 0].view(-1, 1, 1, 1)
        # yp = adjs[:, 1].view(-1, 1, 1, 1)
        # x = (x_gray<=xp) * yp/xp * x + (x_gray>xp) * ((1-yp)/(1-xp) * x + (yp-xp)/(1-xp))

        return x, adjs, new_hidden