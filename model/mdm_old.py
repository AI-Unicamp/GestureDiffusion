import numpy as np
import torch
import torch.nn as nn
from model.rotation2xyz import Rotation2xyz

class MDM_Old(nn.Module):
    def __init__(self, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", data_rep='rot6d', dataset='amass', **kargs):
        super().__init__()
        print('Using MDM V1')
        
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

        # Seed Pose Encoder
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.seed_poses = kargs.get('seed_poses', 0)
        print('Using {} Seed Poses.'.format(self.seed_poses))
        assert self.seed_poses > 0
        self.seed_pose_encoder = SeedPoseEncoder(self.njoints, self.seed_poses, self.latent_dim)

        # Audio Encoder
        self.mfcc_dim = 26
        print('Using Audio Features:')
        print('Selected Features: MFCCs')

        # Input Linear
        self.data_rep = data_rep
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.mfcc_dim, self.latent_dim)

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

        # Sinusoidal Encoder
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # Timestep Network
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Output Linear
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        # Unused?
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    
    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # Get Sizes
        bs, njoints, nfeats, nframes = x.shape # [BS, POSE_DIM, 1, CHUNK_LEN]
        force_mask = y.get('uncond', False)     # TODO: UNDERSTAND MASK

        # Get Timesteps Embeddings
        emb = self.embed_timestep(timesteps)  # [1, BS, LAT_DIM]

        # Seed Poses Embeddings
        flat_seed = y['seed'].squeeze(2).reshape(bs, -1)        # [BS, POSE_DIM, 1, SEED_POSES] -> [BS, POSE_DIM, SEED_POSES] -> [BS, POSE_DIM * SEED_POSES] 
        emb_seed = self.seed_pose_encoder(self.mask_cond(flat_seed, force_mask=force_mask)).unsqueeze(0) # [1, BS, LAT_DIM]
        emb += emb_seed

        # Audio Conditioning
        mfccs = y['mfcc']                               # [BS, MFCC_DIM, 1, CHUNK_LEN]
        x = torch.cat((x, mfccs), axis=1)               # [BS, POSE_DIM + MFCC_DIM, 1, CHUNK_LEN]

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

class SeedPoseEncoder(nn.Module):
    def __init__(self, njoints, seed_poses, latent_dim):
        super().__init__()
        self.njoints = njoints
        self.seed_poses = seed_poses
        self.latent_dim = latent_dim
        self.seed_embed = nn.Linear(self.njoints * self.seed_poses, self.latent_dim)

    def forward(self, x):
        x = self.seed_embed(x)
        return x