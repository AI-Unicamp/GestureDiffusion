# Extracted from the paper ZeroEGGS: Zero-shot Example-based Gesture Generation from Speech
# from https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS
# @article{ghorbani2022zeroeggs,
#  author = {Ghorbani, Saeed and Ferstl, Ylva and Holden, Daniel and Troje, Nikolaus F. and Carbonneau, Marc-André},
#  title = {ZeroEGGS: Zero-shot Example-based Gesture Generation from Speech},
#  journal = {Computer Graphics Forum},
#  volume = {42},
#  number = {1},
#  pages = {206-216},
#  keywords = {animation, gestures, character control, motion capture},
#  doi = {https://doi.org/10.1111/cgf.14734},
#  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14734},
#  eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14734},
#  year = {2023}
#}

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.nn import functional as F

class StyleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size, out_classes, type="gru", use_vae=False):
        super(StyleClassifier, self).__init__()
        self.use_vae = use_vae
        self.styleenc = StyleEncoder(input_size, hidden_size, style_embedding_size, type, self.use_vae)
        #output_size = 2 * style_embedding_size if use_vae else style_embedding_size
        self.fc = nn.Linear(style_embedding_size, out_classes)
        
    def forward(self, x):
        if self.use_vae:
            x = F.relu(self.styleenc(x)[0])
        else:
            x = F.relu(self.styleenc(x))
        x = self.fc(x)
        return x
    
class StyleVAE(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size, length, type="gru", use_vae=True, use_initial_pose=True):
        super(StyleVAE, self).__init__()
        self.use_vae = use_vae
        self.use_initial_pose = use_initial_pose
        if type != "gru":
            raise ValueError("Unknown encoder type: {}".format(type))
        self.styleenc = StyleEncoder(input_size, hidden_size, style_embedding_size, type, self.use_vae)

        self.styledec = Decoder(input_size, input_size, style_embedding_size, hidden_size, num_rnn_layers=2, num_frames=length)
        
    def forward(self, input):
        initial_pose = input[:, 0, :] if self.use_initial_pose else torch.zeros_like(input[:, 0, :])
        x, mu, logvar = self.styleenc(input)
        x = self.styledec(initial_pose, x)
        return x, mu, logvar


# ===============================================
#                   Decoder
# ===============================================
class Decoder(nn.Module):
    def __init__(self, pose_input_size, pose_output_size, style_encoding_size, hidden_size, num_rnn_layers, num_frames):
        super(Decoder, self).__init__()

        self.num_frames = num_frames

        self.recurrent_decoder = RecurrentDecoderNormal(pose_input_size, style_encoding_size, pose_output_size, hidden_size, num_rnn_layers)

        self.cell_state_encoder = CellStateEncoder(pose_input_size + style_encoding_size, hidden_size, num_rnn_layers)

    def forward(self, initial_pose, style_encoding):
        #batchsize = speech_encoding.shape[0]
        #nframes = speech_encoding.shape[1]
        # Initialize the hidden state of decoder
        decoder_state = self.cell_state_encoder( initial_pose, style_encoding )
        motion = [initial_pose.unsqueeze(1)]
        for i in range(1, self.num_frames):
            # Prepare Input
            pose_encoding = motion[-1].squeeze(1)
            # Predict
            predicted, decoder_state = self.recurrent_decoder(pose_encoding, style_encoding, decoder_state)
            # Append
            motion.append(predicted.unsqueeze(1))

        return torch.cat(motion, dim=1)


class RecurrentDecoderNormal(nn.Module):
    def __init__(
            self, pose_input_size, style_size, output_size, hidden_size, num_rnn_layers
    ):
        super(RecurrentDecoderNormal, self).__init__()

        all_input_size = pose_input_size + style_size
        self.layer0 = nn.Linear(all_input_size, hidden_size)
        self.layer1 = nn.GRU(all_input_size + hidden_size, hidden_size, num_rnn_layers, batch_first=True)

        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, pose, style, cell_state):
        hidden = F.elu(self.layer0(torch.cat([pose, style], dim=-1)))
        cell_output, cell_state = self.layer1(torch.cat([hidden, pose, style], dim=-1).unsqueeze(1), cell_state)
        output = self.layer2(cell_output.squeeze(1))
        return output, cell_state

class CellStateEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers):
        super(CellStateEncoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.layer0 = nn.Linear(input_size, hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * num_rnn_layers)

    def forward(self, pose, style):
        #print('cell state encoder')
        # POSE: torch.Size([BS, 249])
        # STYLE: torch.Size([BS, 64])
        hidden = F.elu(self.layer0(torch.cat([pose, style], dim=-1)))
        #print(hidden.shape)
        hidden = F.elu(self.layer1(hidden))
        #print(hidden.shape)
        output = self.layer2(hidden)
        #print(output.shape)
        output = output.reshape(output.shape[0], self.num_rnn_layers, -1).transpose(0, 1).contiguous()
        #print(output.shape)
        return output
    


###########################################
# Style Encoder
###########################################


class StyleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size, type="gru", use_vae=True):
        super(StyleEncoder, self).__init__()
        self.use_vae = use_vae
        self.style_embedding_size = style_embedding_size
        output_size = 2 * style_embedding_size if use_vae else style_embedding_size
        if type == "gru":
            self.encoder = StyleEncoderGRU(input_size, hidden_size, output_size)
        else:
            raise ValueError("Unknown encoder type: {}".format(type))
        #elif type == "attn":
        #    self.encoder = StyleEncoderAttn(input_size, hidden_size, output_size)

    def forward(self, input, temprature: float = 1.0):
        encoder_output = self.encoder(input)
        if self.use_vae:
            mu, logvar = (
                encoder_output[:, : self.style_embedding_size],
                encoder_output[:, self.style_embedding_size:],
            )

            # re-parameterization trick
            std = torch.exp(0.5 * logvar) / temprature
            eps = torch.randn_like(std)

            style_embedding = mu + eps * std
            return style_embedding, mu, logvar
        else:
            return encoder_output
        

class StyleEncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size):
        super(StyleEncoderGRU, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm1D(
                input_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            # AvgPoolNorm1D(kernel_size=2),
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
        )
        self.rnn_layer = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.projection_layer = LinearNorm(
            hidden_size * 2, style_embedding_size, w_init_gain="linear"
        )

    def forward(self, input):
        #print('encoder')
        #print('input', input.shape)
        input = self.convs(input)
        #print('after convs' , input.shape)
        output, _ = self.rnn_layer(input)
        #print('after gru', output.shape)
        #print('used after gru',output[:, -1].shape)
        style_embedding = self.projection_layer(output[:, -1])
        #print('after fc', style_embedding.shape)
        return style_embedding
    


class LinearNorm(nn.Module):
    """ Linear Norm Module:
        - Linear Layer
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """ Forward function of Linear Norm
            x = (*, in_dim)
        """
        x = self.linear_layer(x)  # (*, out_dim)

        return x

class ConvNorm1D(nn.Module):
    """ Conv Norm 1D Module:
        - Conv 1D
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True,
            w_init_gain="linear",
    ):
        super(ConvNorm1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """ Forward function of Conv Norm 1D
            x = (B, L, in_channels)
        """
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.conv(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)

        return x


class AvgPoolNorm1D(nn.Module):
    def __init__(
            self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True
    ):
        super(AvgPoolNorm1D, self).__init__()
        self.avgpool1d = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.avgpool1d(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)

        return x

def generalized_logistic_function(x, center=0.0, B=1.0, A=0.0, K=1.0, C=1.0, Q=1.0, nu=1.0):
    """ Equation of the generalised logistic function
        https://en.wikipedia.org/wiki/Generalised_logistic_function

    :param x:           abscissa point where logistic function needs to be evaluated
    :param center:      abscissa point corresponding to starting time
    :param B:           growth rate
    :param A:           lower asymptote
    :param K:           upper asymptote when C=1.
    :param C:           change upper asymptote value
    :param Q:           related to value at starting time abscissa point
    :param nu:          affects near which asymptote maximum growth occurs

    :return: value of logistic function at abscissa point
    """
    value = A + (K - A) / (C + Q * np.exp(-B * (x - center))) ** (1 / nu)
    return value

def compute_KL_div(mu, logvar, iteration):
    """ Compute KL divergence loss
        mu = (B, embed_dim)
        logvar = (B, embed_dim)
    """
    # compute KL divergence
    # see Appendix B from VAE paper:
    # D.P. Kingma and M. Welling, "Auto-Encoding Variational Bayes", ICLR, 2014.

    kl_weight_center = 7500  # iteration at which weight of KL divergence loss is 0.5
    kl_weight_growth_rate = 0.005  # growth rate for weight of KL divergence loss
    kl_threshold = 2e-1  # KL weight threshold
    # kl_threshold = 1.0

    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B, )
    kl_div = torch.mean(kl_div)

    # compute weight for KL cost annealing:
    # S.R. Bowman, L. Vilnis, O. Vinyals, A.M. Dai, R. Jozefowicz, S. Bengio,
    # "Generating Sentences from a Continuous Space", arXiv:1511.06349, 2016.
    kl_div_weight = generalized_logistic_function(
        iteration, center=kl_weight_center, B=kl_weight_growth_rate,
    )
    # apply weight threshold
    kl_div_weight = min(kl_div_weight, kl_threshold)
    return kl_div, kl_div_weight

def compute_mse(input, reconstr):
    return F.mse_loss(reconstr, input)











###########################################
# Style Decoder
###########################################

#class StyleDecoder(nn.Module):
#    def __init__(self, style_embedding_size, hidden_size, final_output_size, length, type="gru", use_vae=True):
#        super(StyleDecoder, self).__init__()
#        self.use_vae = use_vae
#        #input_size = 2 * style_embedding_size if use_vae else style_embedding_size
#        input_size = style_embedding_size
#        if type == "gru":
#            self.decoder = StyleDecoderGRU(input_size, hidden_size, final_output_size, length)
#        else:
#            raise ValueError("Unknown encoder type: {}".format(type))
#        
#    def forward(self, input):
#        output = self.decoder(input)
#        return output
#            
#class StyleDecoderGRU(nn.Module):
#    def __init__(self, input_size, hidden_size, final_output_size, length):
#        super(StyleDecoderGRU, self).__init__()
#
#        self.length = length
#
#        self.projection_layer = LinearNorm(
#            input_size, hidden_size, w_init_gain="linear"
#        )
#
#        self.rnn_layer = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
#
#        self.convs = nn.Sequential(
#            ConvTransposeNorm1D(
#                hidden_size,
#                hidden_size,
#                kernel_size=3,
#                stride=1,
#                padding=int((3 - 1) / 2),
#                dilation=1,
#                w_init_gain="relu",
#                output_padding=0,
#            ),
#            nn.ReLU(),
#            # AvgPoolNorm1D(kernel_size=2),
#            ConvTransposeNorm1D(
#                hidden_size,
#                final_output_size,
#                kernel_size=3,
#                stride=1,
#                padding=int((3 - 1) / 2),
#                dilation=1,
#                w_init_gain="relu",
#                output_padding=0,
#            ),
#            nn.ReLU(),
#        )
#        
#    def forward(self, input):
#        #print('decoder')
#        #print('input', input.shape)
#        input = self.projection_layer(input)
#        #print('after fc', input.shape)
#        input = input.unsqueeze(1)
#        input = input.repeat(1, self.length, 1)
#        #print('after expand', input.shape)
#        output, _ = self.rnn_layer(input)
#        #print('after gru', output.shape)
#        output = output[:, :, :int(output.shape[-1]/2)]
#        #print('used after gru',output.shape)
#        output = self.convs(output)
#        #print('after convs',output.shape)
#        return output
#
#        #input = self.convs(input)
#        #output, _ = self.rnn_layer(input)
#        #style_embedding = self.projection_layer(output[:, -1])
#        #return style_embedding
#
#class ConvTransposeNorm1D(nn.Module):
#    """ Conv Norm 1D Module:
#        - Conv 1D
#    """
#
#    def __init__(
#            self,
#            in_channels,
#            out_channels,
#            kernel_size=1,
#            stride=1,
#            padding=None,
#            dilation=1,
#            bias=True,
#            w_init_gain="linear",
#            output_padding=1,
#    ):
#        super(ConvTransposeNorm1D, self).__init__()
#        self.conv = nn.ConvTranspose1d(
#            in_channels,
#            out_channels,
#            kernel_size=kernel_size,
#            stride=stride,
#            padding=padding,
#            dilation=dilation,
#            bias=bias,
#            output_padding=output_padding,
#        )
#        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
#
#    def forward(self, x):
#        """ Forward function of Conv Norm 1D
#            x = (B, L, in_channels)
#        """
#        x = x.transpose(1, 2)  # (B, in_channels, L)
#        x = self.conv(x)  # (B, out_channels, L)
#        x = x.transpose(1, 2)  # (B, L, out_channels)
#
#        return x