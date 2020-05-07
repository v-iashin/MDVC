######################################################################################
# The implementation relies on http://nlp.seas.harvard.edu/2018/04/03/attention.html #
######################################################################################

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class VocabularyEmbedder(nn.Module):
    
    def __init__(self, voc_size, d_model):
        super(VocabularyEmbedder, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model
        self.embedder = nn.Embedding(voc_size, d_model)
        
    def forward(self, x): # x - tokens (B, seq_len)
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        
        return x # (B, seq_len, d_model)
    
class FeatureEmbedder(nn.Module):
    
    def __init__(self, d_feat, d_model):
        super(FeatureEmbedder, self).__init__()
        self.d_model = d_model
        self.embedder = nn.Linear(d_feat, d_model)
        
    def forward(self, x): # x - tokens (B, seq_len, d_feat)
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        
        return x # (B, seq_len, d_model)
    
class PositionalEncoder(nn.Module):
    
    def __init__(self, d_model, dout_p, seq_len=3660): # 3651 max feat len for c3d
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)
        
        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))
        
        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)
        
    def forward(self, x): # x - embeddings (B, seq_len, d_model)
        B, S, d_model = x.shape
        # torch.cuda.FloatTensor torch.FloatTensor
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        
        return x # same as input

def subsequent_mask(size):
    mask = torch.ones(1, size, size)
    mask = torch.tril(mask, 0)
    
    return mask.byte() # ([1, size, size])

def mask(src, trg, pad_idx):
    # masking the padding. src shape: (B, S') -> (B, 1, S')
    src_mask = (src != pad_idx).unsqueeze(1)
    
    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)

        return src_mask, trg_mask
    
    else:
        return src_mask

def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

def attention(Q, K, V, mask):
    # Q, K, V are # (B, *(H), seq_len, d_model//H = d_k)
    d_k = Q.size(-1)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)
    
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))
    
    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)
    
    return out # (B, *(H), seq_len, d_model//H = d_k)

class MultiheadedAttention(nn.Module):
    
    def __init__(self, d_model, H):
        super(MultiheadedAttention, self).__init__()
        assert d_model % H == 0
        self.d_model = d_model
        self.H = H
        self.d_k = d_model // H
        # Q, K, V, and after-attention layer (4). out_features is d_model
        # because we apply linear at all heads at the same time
        self.linears = clone(nn.Linear(d_model, d_model), 4) # bias True??
        
    def forward(self, Q, K, V, mask): # Q, K, V are of size (B, seq_len, d_model)
        B, seq_len, d_model = Q.shape
        
        Q = self.linears[0](Q) # (B, *, in_features) -> (B, *, out_features)
        K = self.linears[1](K)
        V = self.linears[2](V)
        
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2) # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        
        if mask is not None:
            # the same mask for all heads
            mask = mask.unsqueeze(1)
        
        # todo: check whether they are both calculated here and how can be 
        # serve both.
        att = attention(Q, K, V, mask) # (B, H, seq_len, d_k)
        att = att.transpose(-3, -2).contiguous().view(B, seq_len, d_model)
        att = self.linears[3](att)
        
        return att # (B, H, seq_len, d_k)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)
        
    def forward(self, x, sublayer): # [(B, seq_len, d_model), attention or feed forward]
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        
        return x + res
    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # todo dropout?
        
    def forward(self, x): # x - (B, seq_len, d_model)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x # x - (B, seq_len, d_model)
    
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.self_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
    def forward(self, x, src_mask): # x - (B, seq_len, d_model) src_mask (B, 1, S)
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs 
        # the output of the self attention
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        
        return x # x - (B, seq_len, d_model)
    
class Encoder(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff), N)
        
    def forward(self, x, src_mask): # x - (B, seq_len, d_model) src_mask (B, 1, S)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        
        return x # x - (B, seq_len, d_model) which will be used as Q and K in decoder
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff):
        super(DecoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 3)
        self.self_att = MultiheadedAttention(d_model, H)
        self.enc_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        # TODO: 
        # should all multiheaded and feed forward
        # attention be the same, check the parameter number
        
    def forward(self, x, memory, src_mask, trg_mask): # x, memory - (B, seq_len, d_model) src_mask (B, 1, S) trg_mask (B, S, S)
        # a comment regarding the motivation of the lambda function 
        # please see the EncoderLayer
        sublayer0 = lambda x: self.self_att(x, x, x, trg_mask)
        sublayer1 = lambda x: self.enc_att(x, memory, memory, src_mask)
        sublayer2 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x = self.res_layers[2](x, sublayer2)
        
        return x # x, memory - (B, seq_len, d_model)
    
class Decoder(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Decoder, self).__init__()
        self.dec_layers = clone(DecoderLayer(d_model, dout_p, H, d_ff), N)
        
    def forward(self, x, memory, src_mask, trg_mask): # x (B, S, d_model) src_mask (B, 1, S) trg_mask (B, S, S)
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, trg_mask)
        # todo: norm?
        return x # (B, S, d_model)
    
class SubsAudioVideoGeneratorConcatLinearDoutLinear(nn.Module):
    
    def __init__(self, d_model_subs, d_model_audio, d_model_video, voc_size, dout_p):
        super(SubsAudioVideoGeneratorConcatLinearDoutLinear, self).__init__()
        self.linear = nn.Linear(d_model_subs + d_model_audio + d_model_video, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)
        print('using SubsAudioVideoGeneratorConcatLinearDoutLinear')
        
    # ?(B, seq_len, d_model_audio), ?(B, seq_len, d_model_audio), ?(B, seq_len, d_model_video)
    def forward(self, subs_x, audio_x, video_x):
        x = torch.cat([subs_x, audio_x, video_x], dim=-1)
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        
        return F.log_softmax(x, dim=-1) # (B, seq_len, voc_size)


class SubsAudioVideoTransformer(nn.Module):
    
    def __init__(self, trg_voc_size, src_subs_voc_size,
                 d_feat_audio, d_feat_video,
                 d_model_audio, d_model_video, d_model_subs,
                 d_ff_audio, d_ff_video, d_ff_subs,
                 N_audio, N_video, N_subs,
                 dout_p, H, use_linear_embedder):
        super(SubsAudioVideoTransformer, self).__init__()
        self.src_emb_subs = VocabularyEmbedder(src_subs_voc_size, d_model_subs)
        if use_linear_embedder:
            self.src_emb_audio = FeatureEmbedder(d_feat_audio, d_model_audio)
            self.src_emb_video = FeatureEmbedder(d_feat_video, d_model_video)
        else:
            assert d_feat_video == d_model_video and d_feat_audio == d_model_audio
            self.src_emb_audio = Identity()
            self.src_emb_video = Identity()
        
        self.trg_emb_subs  = VocabularyEmbedder(trg_voc_size, d_model_subs)
        self.trg_emb_audio = VocabularyEmbedder(trg_voc_size, d_model_audio)
        self.trg_emb_video = VocabularyEmbedder(trg_voc_size, d_model_video)
        self.pos_emb_subs  = PositionalEncoder(d_model_subs, dout_p)
        self.pos_emb_audio = PositionalEncoder(d_model_audio, dout_p)
        self.pos_emb_video = PositionalEncoder(d_model_video, dout_p)
        self.encoder_subs =  Encoder(d_model_subs,  dout_p, H, d_ff_subs,  N_subs)
        self.encoder_audio = Encoder(d_model_audio, dout_p, H, d_ff_audio, N_audio)
        self.encoder_video = Encoder(d_model_video, dout_p, H, d_ff_video, N_video)
        self.decoder_subs =  Decoder(d_model_subs,  dout_p, H, d_ff_subs,  N_subs)
        self.decoder_audio = Decoder(d_model_audio, dout_p, H, d_ff_audio, N_audio)
        self.decoder_video = Decoder(d_model_video, dout_p, H, d_ff_video, N_video)
        
        # late fusion
        self.generator = SubsAudioVideoGeneratorConcatLinearDoutLinear(
            d_model_subs, d_model_audio, d_model_video, trg_voc_size, dout_p
        )
        
        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    # src_subs (B, Ss2, d_feat_subs), src_audio (B, Ss, d_feat_audio) src_video (B, Ss, d_feat_video) 
    # trg (B, St) src_mask (B, 1, Ss) src_sub_mask (B, 1, Ssubs) trg_mask (B, St, St)
    def forward(self, src, trg, mask):
        src_video, src_audio, src_subs = src
        src_mask, trg_mask, src_subs_mask = mask

        # embed
        src_subs = self.src_emb_subs(src_subs)
        src_audio = self.src_emb_audio(src_audio)
        src_video = self.src_emb_video(src_video)
        
        trg_subs = self.trg_emb_subs(trg)
        trg_audio = self.trg_emb_audio(trg)
        trg_video = self.trg_emb_video(trg)
        
        src_subs = self.pos_emb_subs(src_subs)
        src_audio = self.pos_emb_audio(src_audio)
        src_video = self.pos_emb_video(src_video)
        
        trg_subs = self.pos_emb_subs(trg_subs)
        trg_audio = self.pos_emb_audio(trg_audio)
        trg_video = self.pos_emb_video(trg_video)
        
        # encode and decode
        memory_subs = self.encoder_subs(src_subs, src_subs_mask)
        memory_audio = self.encoder_audio(src_audio, src_mask)
        memory_video = self.encoder_video(src_video, src_mask)
        
        out_subs = self.decoder_subs(trg_subs, memory_subs, src_subs_mask, trg_mask)
        out_audio = self.decoder_audio(trg_audio, memory_audio, src_mask, trg_mask)
        out_video = self.decoder_video(trg_video, memory_video, src_mask, trg_mask)
        
        # generate
        out = self.generator(out_subs, out_audio, out_video)
        
        return out # (B, St, voc_size)
