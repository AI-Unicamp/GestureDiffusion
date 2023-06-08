import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.local_attention.transformer import LocalTransformer

class MDM(nn.Module):
    def __init__(self, njoints, nfeats, pose_rep, data_rep, latent_dim=256, ff_size=1024,
                  num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                 dataset='amass', clip_dim=512, clip_version=None, **kargs):
        super().__init__()
        print('Using MDM FiLM')

        # General Configs        
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.data_rep = data_rep
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = self.njoints * self.nfeats
        self.latent_dim = latent_dim
        self.dropout = dropout

        # Timestep Network
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Text Encoder 
        self.use_text = kargs.get('use_text', False)
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.clip_dim = clip_dim
        if self.use_text:
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            print('Using Text')
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        # VAD
        self.use_vad = kargs.get('use_vad', False)
        if self.use_vad:
            self.vad_lookup = nn.Embedding(2, self.latent_dim)
            print('Using VAD')

        # Seed Pose Encoder
        self.seed_poses = kargs.get('seed_poses', 0)
        print('Using {} Seed Poses.'.format(self.seed_poses))
        assert self.seed_poses > 0
        self.seed_pose_encoder = SeedPoseEncoder(self.njoints, self.latent_dim)

        # Audio Encoder
        self.mfcc_input = kargs.get('mfcc_input', False)
        self.use_wavlm = kargs.get('use_wavlm', False)
        print('Using Audio Features:')
        if self.mfcc_input:
            self.mfcc_dim = 26
            self.audio_feat_dim = self.mfcc_dim
            print('Selected Features: MFCCs')
        if self.use_wavlm:
            self.wavlm_proj_dim = 64
            self.audio_feat_dim = self.wavlm_proj_dim
            self.wavlm_encoder = nn.Linear(768, self.audio_feat_dim)
            print('Selected Features: WavLM Representations')

        # Pose Encoder
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)

        # Global Context Encoder
        self.gce = GlobalContextEncoder(latent_dim)

        # Feature Projection
        self.project_to_lat = nn.Linear(self.latent_dim * 2 + self.audio_feat_dim, self.latent_dim)

        # Local Self-Attention
        self.local_film = LocalFilm()

        # Global Self-Attention
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.activation = activation
        self.num_layers = num_layers        
        self.global_film = GlobalFilm(num_heads, ff_size, activation, num_layers, dropout, latent_dim)
 

        # Project Representation to Output Pose
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        # Unused?
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def forward(self, x, timesteps, y=None):

        # Sizes
        _, _, _, nframes = x.shape                              # [BS, POSE_DIM, 1, CHUNK_LEN]

        #############################
        #### FEATURE CALCULATION ####
        #############################

        # Text Embeddings
        if self.use_text:
            enc_text = self.encode_text(y['text'])
            emb_text = self.embed_text(enc_text)                # [1, BS, LAT_DIM]
            emb_text = emb_text.squeeze(0)                      # [BS, LAT_DIM]


        # Seed Poses Embeddings
        seed = y['seed'].squeeze(2).permute(0,2,1)              # [BS, POSE_DIM, 1, SEED_POSES] -> [BS, POSE_DIM, SEED_POSES] -> [BS, SEED_POSES, POSE_DIM] 
        embs_seed = self.seed_pose_encoder(seed)                # [BS, SEED_POSES, LAT_DIM]
        embs_seed = embs_seed.permute(1,0,2)                    # [SEED_POSES, BS, LAT_DIM]

        # VAD Embeddings
        vad_vals = y['vad']                                     # [BS, CHUNK_LEN]
        emb_vad = self.vad_lookup(vad_vals)                     # [BS, CHUNK_LEN, LAT_DIM]
        emb_vad = emb_vad.permute(1, 0, 2)                      # [CHUNK_LEN, BS, LAT_DIM]

        # Timesteps Embeddings
        emb_t = self.embed_timestep(timesteps)                  # [1, BS, LAT_DIM]

        # Audio Embeddings
        if self.mfcc_input:                                     # TODO: is it actually the raw mfccs? 
            emb_audio = y['audio_rep']                          # [BS, MFCC_DIM, 1, CHUNK_LEN]
        elif self.use_wavlm:
            interp_reps = y['audio_rep']                        # [BS, 768, 1, CHUNK_LEN]
            interp_reps = interp_reps.permute(0, 3, 2, 1)       # [BS, CHUNK_LEN, 1, 768]
            emb_audio = self.wavlm_encoder(interp_reps)         # [BS, CHUNK_LEN, 1, WAVLM_PROJ_DIM]
            emb_audio = emb_audio.permute(0, 3, 2, 1)           # [BS, WAVLM_PROJ_DIM, 1, CHUNK_LEN]
        else:
            raise NotImplementedError
        emb_audio = emb_audio.squeeze(2)                        # [BS, AUDIO_DIM, CHUNK_LEN], (AUDIO_DIM = MFCC_DIM or WAVLM_PROJ_DIM)
        emb_audio = emb_audio.permute((2, 0, 1))                # [CHUNK_LEN, BS, AUDIO_DIM]

        # Pose Embeddings
        emb_pose = self.input_process(x)                        # [CHUNK_LEN, BS, LAT_DIM]

        #############################
        #### FEATURE AGGREGATION ####
        #############################

        # Global Embeddings
        global_embs = (emb_text + emb_t).permute(1,0,2)         # [BS, 1, LAT_DIM]
        gammas, betas = self.gce(global_embs)                   #  

        # Cat Pose w/ Audio and VAD(Fine-Grained) Embeddings
        fg_embs=torch.cat((emb_pose,emb_audio,emb_vad),axis=2)  # [CHUNK_LEN, BS, LAT_DIM + AUDIO_DIM + LAT_DIM]

        # Project to Latent Dim
        xseq = self.project_to_lat(fg_embs)                     # [CHUNK_LEN, BS, LAT_DIM]

        ######################
        ##### LOCAL PASS #####
        ######################

        # Apply Positional Sinusoidal Embeddings
        xseq = self.sequence_pos_encoder(xseq)                  # [CHUNK_LEN, BS, LAT_DIM]  

        # Local Attention
        
        xseq = self.local_film(gammas, betas, xseq)             # [CHUNK_LEN, BS, LAT_DIM]

        #############################
        #### FEATURE AGGREGATION ####
        #############################

        # Concat Past Information
        xseq = torch.cat((embs_seed, xseq), axis=0)             # [CHUNK_LEN+N_SEED, BS, LAT_DIM]   

        #######################
        ##### GLOBAL PASS #####
        #######################

        # Apply Positional Sinusoidal Embeddings
        xseq = self.sequence_pos_encoder(xseq)                  # [CHUNK_LEN+N_SEED, BS, LAT_DIM]

        # Global Attention
        output = self.global_film(gammas, betas, xseq)          # [CHUNK_LEN+N_SEED, BS, LAT_DIM] 

        # Ignore First Tokens
        output = output[self.seed_poses:]                       # [CHUNK_LEN, BS, LAT_DIM]

        # Linear Output Feature Pass
        output = self.output_process(output)                    # [BS, POSE_DIM, 1, CHUNK_LEN]

        return output

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

        if self.data_rep in ['genea_vec', 'genea_vec+']:
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
        if self.data_rep in ['genea_vec', 'genea_vec+']:
            output = self.poseFinal(output) # [CHUNK_LEN, BS, POSE_DIM]
        else:
            raise NotImplementedError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats) # [CHUNK_LEN, BS, POSE_DIM, 1]
        output = output.permute(1, 2, 3, 0)  
        return output
    
class SeedPoseEncoder(nn.Module):
    def __init__(self, njoints, latent_dim):
        super().__init__()
        self.njoints = njoints
        self.latent_dim = latent_dim
        self.seed_embed = nn.Linear(self.njoints, self.latent_dim)

    def forward(self, x):       # [BS, SEED_POSES, POSE_DIM]
        x = self.seed_embed(x)  # [BS, SEED_POSES, LAT`DIM]
        return x
    
class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x
    
class GlobalContextEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(GlobalContextEncoder, self).__init__()
        # self.global_processor
        self.latent_dim = latent_dim
        self.feature_wise_affine = LinearNorm(self.latent_dim, 2 * self.latent_dim)

    def forward(self, x):
        x = self.feature_wise_affine(x)
        gammas, betas = torch.split(x, self.latent_dim, dim=-1)
        return gammas, betas
    
class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    , extended to 'TADAM: Task dependent adaptive metric for improved few-shot learning'
    """
    def __init__(self):
        super(FiLM, self).__init__()
        self.s_gamma = nn.Parameter(torch.ones(1,), requires_grad=True)
        self.s_beta = nn.Parameter(torch.ones(1,), requires_grad=True)

    def forward(self, x, gammas, betas):
        """
        x -- [B, T, H]
        gammas -- [B, 1, H]
        betas -- [B, 1, H]
        """
        gammas = self.s_gamma * gammas.expand_as(x)
        betas = self.s_beta * betas.expand_as(x)
        return (gammas + 1.0) * x + betas

      
class LocalFilm(nn.Module):
    def __init__(self):
        super().__init__()

        self.local_transformer = LocalTransformer()
        self.film = FiLM()

    def forward(self,  gammas, betas, xseq):
        xseq = xseq.permute(1, 0, 2)                            # [BS, CHUNK_LEN, LAT_DIM]
        xseq = self.local_transformer(xseq)                     # [BS, CHUNK_LEN, LAT_DIM]
        xseq = self.film(xseq, gammas, betas)                   # [BS, CHUNK_LEN, LAT_DIM]
        xseq = xseq.permute(1, 0, 2)                            # [CHUNK_LEN, BS, LAT_DIM]
        return xseq
    
class GlobalFilm(nn.Module):
    def __init__(self, num_heads, ff_size, activation, num_layers, dropout, latent_dim):
        super().__init__()
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.seqTransEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                                                        d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation),
                                                        num_layers=4)  
        self.film = FiLM()

    def forward(self,  gammas, betas, x):
        x = self.seqTransEncoder(x)      # [CHUNK_LEN+SEED_POSES, BS, LAT_DIM] 
        x = x.permute(1, 0, 2)           # [BS, CHUNK_LEN+SEED_POSES, LAT_DIM] 
        x = self.film(x, gammas, betas)  # [BS, CHUNK_LEN+SEED_POSES, LAT_DIM] 
        x = x.permute(1, 0, 2)           # [CHUNK_LEN+SEED_POSES, BS, LAT_DIM] 
        return x