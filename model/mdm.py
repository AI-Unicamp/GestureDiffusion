import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz

class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        # General Configs        
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = self.njoints * self.nfeats
        self.latent_dim = latent_dim
        self.cond_mode = kargs.get('cond_mode', 'no_cond')

        # Unused?
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.num_actions = num_actions
        self.action_emb = kargs.get('action_emb', None)

        # Text Encoder 
        self.use_text = kargs.get('use_text', False)
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.clip_dim = clip_dim
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

        # Audio Encoder
        self.use_audio = kargs.get('use_audio', False)
        self.mfcc_input = kargs.get('mfcc_input', False)
        self.use_wav_enc = kargs.get('use_wav_enc', False) 
        self.mfcc_dim = 26 if self.mfcc_input else 0
        self.wav_enc_dim = 32 if self.use_wav_enc else 0
        self.augmented_input_feats = self.input_feats+self.mfcc_dim+self.wav_enc_dim
        if use_audio:
            print('Using Audio Features:')
            if self.mfcc_input:
                print('Selected Features: MFCCs')
            if self.use_wav_enc:
                print('Selected Features: WavEncoder Representations')
                self.wav_encoder = WavEncoder()

        # Input Linear
        self.data_rep = data_rep
        self.input_process = InputProcess(self.data_rep, self.augmented_input_feats, self.latent_dim)

        # Denoiser Network
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.seqTransEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                                                        d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation),
                                                        num_layers=self.num_layers)

        # Timestep Network
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Sinusoidal Encoder
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # Output Linear
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        # Unused?
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # Get Sizes
        bs, njoints, nfeats, nframes = x.shape # [BS, POSE_DIM, 1, CHUNK_LEN]
        
        # Get Timesteps Embeddings
        emb = self.embed_timestep(timesteps)  # [1, BS, LAT_DIM]

        # Text Conditioning
        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            if self.use_text:
                emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)) # [1, BS, LAT_DIM]

        # Audio Conditioning
        if self.use_audio:
            if self.mfcc_input:
                mfccs = y['mfcc']                               # [BS, MFCC_DIM, 1, CHUNK_LEN]
                x = torch.cat((x, mfcss), axis=1)               # [BS, POSE_DIM + MFCC_DIM, 1, CHUNK_LEN]
            if self.use_wav_enc:
                audio_representation = self.wav_encoder(y['audio']) # [BS, WAV_ENC_DIM, 1, CHUNK_LEN]
                x = torch.cat((x, audio_representation), axis=1)    # [BS, POSE_DIM + WAV_ENC_DIM, 1, CHUNK_LEN]

        # Linear Input Feature Pass
        x = self.input_process(x)               # [CHUNK_LEN, BS, LAT_DIM]

        # Cat 0th-conditioning-token
        xseq = torch.cat((emb, x), axis=0)      # [CHUNK_LEN+1, BS, LAT_DIM]  
        
        # Apply Positional Sinusoidal Embeddings
        xseq = self.sequence_pos_encoder(xseq)  # [CHUNK_LEN+1, BS, LAT_DIM]  

        # Transformer Encoder Pass
        output = self.seqTransEncoder(xseq)     # [CHUNK_LEN+1, BS, LAT_DIM]

        # Ignore First Token
        output = output[1:]                     # [CHUNK_LEN, BS, LAT_DIM]

        # Linear Output Feature Pass
        output = self.output_process(output)    # [BS, POSE_DIM, 1, CHUNK_LEN]
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class WavEncoder(nn.Module):
    '''
    Taken from https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context/
    '''
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600, dilation = 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=5, dilation = 4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=5, dilation = 7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=5, dilation = 13),
        )

    def forward(self, wav_data):            # [B, 147000]
        wav_data = wav_data.unsqueeze(1)    # [B, 1, 147000]
        out = self.feat_extractor(wav_data) # [B, 32, 200] 
        return out.unsqueeze(2)             # [B, 32, 1, 200]
    
    def layer_output_size(self,l_in, padding, kernel_size, dilation, stride):
        l_out = int(np.floor((l_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1))
        return l_out

class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep == 'genea_vec':
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        else:
            raise NotImplementedError

class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep == 'genea_vec':
            output = self.poseFinal(output) # [CHUNK_LEN, BS, POSE_DIM]
        else:
            raise NotImplementedError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats) # [CHUNK_LEN, BS, POSE_DIM, 1]
        output = output.permute(1, 2, 3, 0)  
        return output

class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output