import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
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

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input, h_prev), dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = F.sigmoid(cc_i)
        f = F.sigmoid(cc_f)
        o = F.sigmoid(cc_o)
        g = F.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * F.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, cuda=True, device=None):
        state = (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                 Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))
        if cuda:
            state = (state[0].to(device), state[1].to(device))
        return state


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, effective_layers=[-1],
                 batch_norm=False, batch_first=False, bias=True, return_all_layers=False, device=None):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.effective_layers = effective_layers
        self.fc_conv = nn.Conv2d(sum([self.hidden_dim[i] for i in self.effective_layers]), self.input_dim, 1)
        self.device = device
        self.batch_norm = batch_norm

        cell_list = []
        bn_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
            bn_list.append(nn.BatchNorm2d(self.hidden_dim[i]))

        self.cell_list = nn.ModuleList(cell_list)
        if self.batch_norm:
            self.bn_list = nn.ModuleList(bn_list)

    def forward(self, input, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input = input.permute(1, 0, 2, 3, 4)


        if hidden_state is None:
            hidden_state = self.get_init_states(batch_size=input.size(0), device=self.device)

        layer_output_list = []
        last_state_list   = []

        seq_len = input.size(1)
        cur_layer_input = input

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input=cur_layer_input[:, t, :, :, :],
                                                 prev_state=[h, c])
                if self.batch_norm:
                    output_inner.append(self.bn_list[layer_idx](h))
                else:
                    output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))


        layer_output = self.fc_conv(torch.stack([layer_output_list[i] for i in self.effective_layers], dim=2).view(-1, sum([self.hidden_dim[i] for i in self.effective_layers]), self.height, self.width))
        layer_output = layer_output.view(*input.size())
        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2, 3, 4)

        return layer_output, last_state_list

    def get_init_states(self, batch_size, cuda=True, device=None):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, cuda, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple, list or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param