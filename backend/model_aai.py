import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence
import os
from layers import ConvReLUNorm
from transformer_aai import FFTransformer
device = 'cpu'


class FastPitch(nn.Module):
    def __init__(self, n_mel_channels, max_seq_len, n_symbols, padding_idx,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 p_dur_predictor_dropout, dur_predictor_n_layers,
                 pitch_predictor_kernel_size, pitch_predictor_filter_size,
                 p_pitch_predictor_dropout, pitch_predictor_n_layers,
                 pitch_embedding_kernel_size, n_speakers, speaker_emb_weight):
        super(FastPitch, self).__init__()
        del max_seq_len

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=41,
            padding_idx=padding_idx).float()


        self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim)
        self.proj = nn.Linear(symbols_embedding_dim, n_mel_channels, bias=True)
        self.proj_decin2 = nn.Linear(symbols_embedding_dim, symbols_embedding_dim, bias=True)
        self.proj_decin = nn.Linear(13, symbols_embedding_dim, bias=True)
    def forward(self, inputs, dur_tgt, speaker=None, use_gt_durations=True, use_gt_pitch=False,
                pace=1.0, max_duration=75, return_attn=False):
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)
        inputs = self.proj_decin(inputs)
        inputs = self.proj_decin2(inputs)

        enc_out = self.encoder(inputs, conditioning=spk_emb, return_attn=return_attn, use_c=False, use_pos=False)
        dec_out = self.decoder(enc_out, return_attn=return_attn)
        mel_out = self.proj(dec_out)
        return mel_out, None
