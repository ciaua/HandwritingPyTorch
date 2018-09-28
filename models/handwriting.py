import numpy as np
from scipy.signal import argrelextrema
from . import my_utils as utils

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

USE_CUDA = False

strokes = np.load('../data/strokes.npy', encoding='latin1')
stroke = strokes[0]

stroke_mean = np.load('../data/stroke_xy_mean.npy')
stroke_std = np.load('../data/stroke_xy_std.npy')

allchars = [term[0] for term in utils.read_csv('../data/allchars.csv')]
num_allchars = len(allchars)
char2num = {ch: ii for ii, ch in enumerate(allchars)}


class UncondStackedGRUCell(nn.Module):
    ''' Stacked GRUCell for unconditional generation
    This is a stack of GRU cells. It will process information in `one timestep`
    in a sequence.

    It takes a stroke as the input, and output the parameters of Gaussians and
    Bernoulli distributions.

    This module is used in UncondNet.

    Args:
        input_size: int
            3

        output_size: int
            1 + num_components*num_params_per_comp

        hidden_sizes: list
            the feature size in a GRU cell

        num_cells: int
            the number of GRU cells in a stack

    Inputs:
        next_stroke: Tensor
            shape=(batch_size, 3)

        hiddens: list
            a list of hidden layer outputs from the previous timestep
            len(hiddens) == num_layers
            The i-th item in `hiddens` has the
            shape=(batch_size, hidden_sizes[i])

    Outputs:
        output: Tensor
            shape=(batch_size, output_size)

        new_hiddens: list
            similar to `hiddens`


    '''
    def __init__(self, input_size, output_size, hidden_sizes, num_cells):
        super(UncondStackedGRUCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.num_cells = num_cells

        self.GRUCells = nn.ModuleList(
            [nn.GRUCell(input_size, hidden_size) if ii == 0 else
             nn.GRUCell(hidden_size, hidden_size)
             for ii, hidden_size in enumerate(hidden_sizes)]
        )

        self.relu = nn.LeakyReLU()

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, next_stroke, hiddens):
        input = next_stroke

        new_hiddens = []
        for ii, [hidden, cell] in enumerate(zip(hiddens, self.GRUCells)):
            new_hidden = cell(input, hidden)

            new_hiddens.append(new_hidden)

            input = new_hidden

        output = self.output_layer(input)

        return output, new_hiddens


class UncondNet(nn.Module):
    ''' A recurrent neural network that unconditionally generates strokes
    '''
    def __init__(self, input_size, output_size, hidden_sizes,
                 num_components, num_params_per_comp):
        super(UncondNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.num_cells = len(hidden_sizes)

        self.num_components = num_components
        self.num_params_per_comp = num_params_per_comp

        self.stackedGRUCell = UncondStackedGRUCell(
            input_size, output_size, hidden_sizes, self.num_cells)

    def make_init_hidden(self, batch_size, k):
        init_hidden = torch.zeros(batch_size, k)
        if USE_CUDA:
            init_hidden = init_hidden.cuda()
        return init_hidden

    def sample_next_stroke(self, params):
        bs, _, = params.size()

        e = torch.sigmoid(params[:, 0])

        params = params[:, 1:].view(
            bs, self.num_components, self.num_params_per_comp)

        pp = torch.softmax(params[:, :, 0], dim=1)
        mu = params[:, :, 1:3]
        sig = torch.exp(params[:, :, 3:5])
        tho = torch.tanh(params[:, :, 5])

        # Sample from Bernoulli
        stop_bit = torch.bernoulli(e)

        # Sample from Multinomial
        comp = torch.multinomial(pp, num_samples=1)[:, 0]

        # Sample from Gaussian
        mu = mu[range(bs), comp]
        sig = sig[range(bs), comp]
        tho = tho[range(bs), comp]

        sig2 = sig**2
        offdiag = tho*sig.prod(1)

        cov = torch.zeros(bs, 2, 2)
        cov[:, 0, 0] = sig2[:, 0]
        cov[:, 1, 1] = sig2[:, 1]
        cov[:, 0, 1] = offdiag
        cov[:, 1, 0] = offdiag

        coords = []
        for ii in range(bs):
            mn = MultivariateNormal(mu[ii], covariance_matrix=cov[ii])
            coord = mn.sample()

            coords.append(coord)

        coords = torch.stack(coords, dim=0)

        # Next stroke
        next_stroke = torch.cat([stop_bit[:, None], coords], dim=1)

        return next_stroke

    def forward(self, num_steps=None, real_stroke=None, seed=None):
        '''
        If `real_stroke' is given, it means that this is for training.
        Therefore, the `outputs' will be the predicted parameters
        for Gaussian and Bernoulli.

        If real_stroke is None, it means that this is for evaluation.
        Therefore, the `outputs' will the predicted strokes.

        noise.shape=(batch_size, noise_size)
        real_stroke.shape=(batch_size, 3, num_steps)
        '''
        if seed is not None:
            torch.manual_seed(seed)

        if real_stroke is not None:
            num_steps = real_stroke.size(2)
            batch_size = real_stroke.size(0)
        else:
            batch_size = 1

        hiddens = [self.make_init_hidden(batch_size, hidden_size)
                   for hidden_size in self.hidden_sizes]

        next_stroke = torch.zeros(batch_size, self.input_size)
        # next_stroke = torch.zeros(batch_size, self.input_size).cuda(gid)
        outputs = []
        for ii in range(num_steps):
            output, hiddens = self.stackedGRUCell(next_stroke, hiddens)
            if real_stroke is None:
                next_stroke = self.sample_next_stroke(output)
                outputs.append(next_stroke)
            else:
                next_stroke = real_stroke[:, :, ii]
                outputs.append(output)

        # shape=(batch_size, output_size, timesteps)
        outputs = torch.stack(outputs, dim=2)

        return outputs


class CondStackedGRUCell(nn.Module):
    ''' Stacked GRUCell for conditional generation
    This is a stack of GRU cells. It will process information in `one timestep`
    in a sequence.

    It takes a stroke, a sentence and a soft window as the input, and output
    the parameters of Gaussians and Bernoulli distributions. In the process, it
    also produces the parameters for making the soft window.

    This module is used in CondNet.

    Args:
        input_size: int
            3

        output_size: int
            1 + num_components*num_params_per_comp

        hidden_sizes: list
            the feature size in a GRU cell

        num_cells: int
            the number of GRU cells in a stack

        num_allchars: int
            the number of unique chars in the whole dataset.

        num_sw_components: int
            the number of Gaussian functions used in the soft window

    Inputs:
        next_stroke: Tensor
            shape=(batch_size, 3)

        hiddens: list
            a list of hidden layer outputs from the previous timestep
            len(hiddens) == num_layers
            The i-th item in `hiddens` has the
            shape=(batch_size, hidden_sizes[i])

    Outputs:
        output: Tensor
            shape=(batch_size, output_size)

        new_hiddens: list
            similar to `hiddens`

    '''
    def __init__(self, input_size, output_size, hidden_sizes,
                 num_cells, num_allchars, num_sw_components):
        super(CondStackedGRUCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.num_cells = num_cells

        # Soft window
        self.num_allchars = num_allchars
        self.num_sw_components = num_sw_components

        self.num_sw_params = num_sw_components*3

        self.GRUCells = nn.ModuleList(
            [nn.GRUCell(input_size+num_allchars, hidden_size)
             if ii == 0 else
             nn.GRUCell(hidden_size+num_allchars, hidden_size)
             for ii, hidden_size in enumerate(hidden_sizes)]
        )

        self.relu = nn.LeakyReLU()

        self.swo = nn.Linear(hidden_sizes[0], self.num_sw_params)

        nn.init.normal_(self.swo.bias, -3, 0.1)

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def compute_soft_window(self, sentence, sw_params, ka, mask=None):
        '''
        sentence.shape=(bs, num_allchars, max_sentence_len)
        mask.shape=(bs, max_sentence_len)
        '''
        max_sen_len = sentence.size(2)
        bs = sw_params.size(0)
        sw_params = torch.exp(sw_params).view(bs, self.num_sw_components, 3)

        # shape=(bs, num_sw_components)
        aa = sw_params[:, :, None, 0]
        bb = sw_params[:, :, None, 1]
        ka = ka[:, :, None] + sw_params[:, :, None, 2]

        # shape=(bs, num_sw_components, max_sentence_len)
        uu = torch.arange(1, sentence.size(2)+2).float()[None, None, :]
        if USE_CUDA:
            uu = uu.cuda()

        # shape=(bs, max_sentence_len+1)
        extended_phi = torch.sum(aa*torch.exp(-(bb*(ka - uu)**2)), dim=1)

        # shape=(bs, max_sentence_len)
        phi = extended_phi[:, :-1]
        if mask is not None:
            masked_phi = phi*mask
        else:
            masked_phi = phi

        soft_window = (masked_phi[:, None]*sentence).sum(2)

        # check stop condition
        if mask is None:
            # (Option1) Follow Alex Graves's paper to decide when to stop writing
            # stop_writing = (phi[:, -1] >= phi[:, :-1].max(dim=1)[0]).min().item()
            # (Option2) My way of stopping
            stop_writing = (ka[:, :, 0].min(1)[0] > (max_sen_len+1)).min().item()
        else:
            stop_writing = (ka[:, :, 0].mean(1) > (mask.sum(1)+1)).min().item()

        return soft_window, ka[:, :, 0], stop_writing

    def forward(self, next_stroke, hiddens, sentence, soft_window, ka,
                mask=None):
        input = torch.cat([next_stroke, soft_window], dim=1)

        new_hiddens = []
        for ii, [hidden, cell] in enumerate(zip(hiddens, self.GRUCells)):
            new_hidden = cell(input, hidden)
            if ii == 0:
                sw_params = self.swo(new_hidden)
                soft_window, ka, stop_writing = self.compute_soft_window(
                    sentence, sw_params, ka, mask=mask)

            new_hiddens.append(new_hidden)

            input = torch.cat([new_hidden, soft_window], dim=1)

        output = self.output_layer(new_hidden)

        return output, new_hiddens, soft_window, ka, stop_writing


class CondNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes,
                 num_components, num_params_per_comp,
                 num_allchars, num_sw_components):
        super(CondNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.num_cells = len(hidden_sizes)

        self.num_components = num_components
        self.num_params_per_comp = num_params_per_comp

        # Soft window
        self.num_allchars = num_allchars
        self.num_sw_components = num_sw_components

        self.num_sw_params = num_sw_components*3

        self.stackedGRUCell = CondStackedGRUCell(
            input_size, output_size, hidden_sizes, self.num_cells,
            self.num_allchars, self.num_sw_components)

    def make_init_hidden(self, batch_size, k):
        init_hidden = torch.zeros(batch_size, k)
        if USE_CUDA:
            init_hidden = init_hidden.cuda()
        return init_hidden

    def sample_next_stroke(self, params):
        bs, _, = params.size()

        e = torch.sigmoid(params[:, 0])

        params = params[:, 1:].view(
            bs, self.num_components, self.num_params_per_comp)

        pp = torch.softmax(params[:, :, 0], dim=1)
        mu = params[:, :, 1:3]
        sig = torch.exp(params[:, :, 3:5])
        tho = torch.tanh(params[:, :, 5])

        # Sample from Bernoulli
        stop_bit = torch.bernoulli(e)

        # Sample from Multinomial
        comp = torch.multinomial(pp, num_samples=1)[:, 0]

        # Sample from Gaussian
        mu = mu[range(bs), comp]
        sig = sig[range(bs), comp]
        tho = tho[range(bs), comp]

        sig2 = sig**2
        offdiag = tho*sig.prod(1)

        cov = torch.zeros(bs, 2, 2)
        cov[:, 0, 0] = sig2[:, 0]
        cov[:, 1, 1] = sig2[:, 1]
        cov[:, 0, 1] = offdiag
        cov[:, 1, 0] = offdiag

        coords = []
        for ii in range(bs):
            mn = MultivariateNormal(mu[ii], covariance_matrix=cov[ii])
            coord = mn.sample()

            coords.append(coord)

        coords = torch.stack(coords, dim=0)

        # Next stroke
        next_stroke = torch.cat([stop_bit[:, None], coords], dim=1)

        return next_stroke

    def forward(self, sentence, mask=None, real_stroke=None, seed=None,
                max_steps=None):
        '''
        sentence.shape=(batch_size, num_chars, num_steps)
        real_stroke.shape=(batch_size, 3, num_steps)
        '''
        if seed is not None:
            torch.manual_seed(seed)

        if real_stroke is not None:
            num_steps = real_stroke.size(2)
            batch_size = real_stroke.size(0)
        else:
            batch_size = 1

        hiddens = [self.make_init_hidden(batch_size, hidden_size)
                   for ii, hidden_size in enumerate(self.hidden_sizes)]

        next_stroke = torch.zeros(batch_size, self.input_size)
        soft_window = torch.zeros(batch_size, self.num_allchars)
        ka = torch.zeros(batch_size, self.num_sw_components)
        if USE_CUDA:
            next_stroke = next_stroke.cuda()
            soft_window = soft_window.cuda()
            ka = ka.cuda()
        next_stroke[:, 0] = 1

        outputs = []
        count_steps = 0
        while True:
            count_steps += 1
            output, hiddens, soft_window, ka, stop_writing = \
                self.stackedGRUCell(
                    next_stroke, hiddens, sentence, soft_window,
                    ka, mask=mask)
            if real_stroke is None:
                next_stroke = self.sample_next_stroke(output)
                outputs.append(next_stroke)
                if stop_writing == 1 or count_steps == max_steps:
                    break
            else:
                next_stroke = real_stroke[:, :, count_steps - 1]
                outputs.append(output)
                if count_steps == num_steps:
                    break

        # shape=(batch_size, output_size, timesteps)
        outputs = torch.stack(outputs, dim=2)

        return outputs


class RecogNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RecogNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        self.relu = nn.LeakyReLU()
        self.GRU = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True,
            bidirectional=True
        )

        self.conv1 = nn.Conv1d(2*hidden_size, 512, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.char_head = nn.Conv1d(256, output_size, 3, 1, 1)
        self.cut_head = nn.Conv1d(256, 1, 3, 1, 1)

    def make_init_hidden(self, batch_size):
        # return torch.zeros(batch_size, k)
        init_hidden = torch.zeros(
            self.num_layers*self.num_directions, batch_size, self.hidden_size)

        if USE_CUDA:
            init_hidden = init_hidden.cuda()

    def forward(self, x):
        batch_size = x.size(0)
        init_hidden = self.make_init_hidden(batch_size)

        x, _ = self.GRU(x, init_hidden)
        x = x.transpose(1, 2)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x_char = self.char_head(x)
        x_cut = self.cut_head(x)[:, 0]

        x_char = x_char.transpose(1, 2)

        return x_char, x_cut


def setup_unconditional_model():
    unconditional_model_params_fp = '../models/unconditional_model.params.torch'

    num_components = 20
    num_params_per_comp = 6  # pi:1, mu:2, sigma:2, tho: 1

    input_size = 3
    output_size = 1 + num_components*num_params_per_comp  # e:1
    hidden_sizes = [400, 400, 400]

    net = UncondNet(
        input_size, output_size, hidden_sizes,
        num_components, num_params_per_comp)
    utils.load_model(unconditional_model_params_fp, net, device_id='cpu')
    if USE_CUDA:
        net = net.cuda()

    net.eval()

    return net


def setup_conditional_model():
    conditional_model_params_fp = '../models/conditional_model.params.torch'

    num_components = 20
    num_params_per_comp = 6  # pi:1, mu:2, sigma:2, tho: 1

    input_size = 3
    output_size = 1 + num_components*num_params_per_comp  # e:1
    hidden_sizes = [400, 400, 400]

    num_sw_components = 10

    net = CondNet(
        input_size, output_size, hidden_sizes,
        num_components, num_params_per_comp,
        num_allchars, num_sw_components)
    utils.load_model(conditional_model_params_fp, net, device_id='cpu')
    if USE_CUDA:
        net = net.cuda()

    net.eval()

    return net


def setup_recognition_model():
    recognition_model_params_fp = '../models/recognition_model.params.torch'

    input_size = 3
    output_size = 77
    hidden_size = 400
    num_hidden_layers = 3

    net = RecogNet(input_size, output_size, hidden_size, num_hidden_layers)
    utils.load_model(recognition_model_params_fp, net, device_id='cpu')
    if USE_CUDA:
        net = net.cuda()

    net.eval()

    return net


net_uncond = setup_unconditional_model()
net_cond = setup_conditional_model()
net_recog = setup_recognition_model()


def sentence2onehot(sentence, num_allchars):
    '''
    sentence.shape=(batch_size, sentence_length)
    '''
    bs, sl = sentence.size()

    # print(bs, num_allchars, sl)
    out = torch.zeros(bs, num_allchars, sl)

    out.scatter_(1, sentence[:, None].long(), 1.)
    return out


def generate_unconditionally(random_seed=1, num_steps=600):
    '''
    Input:
      random_seed - integer

    Output:
      stroke - numpy 2D-array (T x 3)
    '''
    # Generate
    gen = net_uncond(num_steps=num_steps, seed=random_seed).detach().numpy()

    # Convert from standardized strokes to raw strokes
    gen[:, 1:] = gen[:, 1:]*stroke_std[None, :, None] + stroke_mean[None, :, None]
    stroke = gen[0].T

    return stroke


def generate_conditionally(text='welcome home', random_seed=1,
                           max_strokes_per_char=40):
    '''
    Input:
      text - str
      random_seed - integer

    Output:
      stroke - numpy 2D-array (T x 3)
    '''
    # Preprocess text
    sentence = np.array([char2num[ch] for ch in text],
                        dtype='float32')
    sentence = torch.from_numpy(sentence[None, :])
    sentence = sentence2onehot(sentence, num_allchars)

    # Generate
    max_steps = len(text)*max_strokes_per_char
    gen = net_cond(
        sentence, max_steps=max_steps, seed=random_seed).detach().numpy()

    # Convert from standardized strokes to raw strokes
    gen[:, 1:] = gen[:, 1:]*stroke_std[None, :, None] + stroke_mean[None, :, None]
    stroke = gen[0].T

    return stroke


def recognize_stroke(stroke, cut_threshold=0.3):
    '''
    Input:
      stroke - numpy 2D-array (T x 3)
      cut_threshold - float

    Output:
      text - str
    '''

    # ### Proprocess stroke ###
    stroke[:, 1:] = (stroke[:, 1:] - stroke_mean[None]) / stroke_std[None]

    stroke = torch.from_numpy(stroke[None])

    # ### Predict characters and cuts ###
    char_pred, cut_pred = net_recog(stroke)
    char_pred = char_pred.transpose(1, 2)[0]

    char_pred = char_pred.argmax(0).data.numpy()
    cut_pred = torch.sigmoid(cut_pred)[0].data.numpy()

    # ### Process cuts ###
    # Goal: Get predicted locations where we should cut two neighboring characters

    # Thresholding
    cut_ind_high_val = (cut_pred > cut_threshold).nonzero()[0].tolist()

    # Local maximum
    cut_ind_local_max = argrelextrema(cut_pred, np.greater)[0].tolist()

    # Intersection of both sets of indices
    mm = list(set.intersection(set(cut_ind_high_val), set(cut_ind_local_max)))
    mm.sort()

    # Prepend initial index and append the final index
    mm = [0] + mm + [char_pred.shape[0]]

    # ### Recognition ###
    pred = ''
    for ii in range(len(mm)-1):
        # ### Process a segment defined by two predicted cuts ###

        chars_ = char_pred[mm[ii]:mm[ii+1]]

        # Make a histogram for this segment
        unique_chars = list(set(chars_.tolist()))
        dd = {cc: (chars_ == cc).sum() for cc in unique_chars}

        # Get the most frequent character
        char_idx = max(unique_chars, key=lambda x: dd[x])
        char = allchars[char_idx]

        pred += char

    return pred
