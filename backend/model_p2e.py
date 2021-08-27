import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence

from layers import ConvReLUNorm
from utils import mask_from_lens
from transformer_p2e import FFTransformer
from typing import Optional
from torch.nn.utils.rnn import pad_sequence

def regulate_len(durations, enc_out, pace=1, mel_max_len=None, training=False):
    pace=0.0265
    """If target=None, then predicted durations are applied"""
    reps = torch.round(durations.float() / pace).long()
    dec_lens = reps.sum(dim=1)
    enc_rep = pad_sequence([torch.repeat_interleave(o, r, dim=0)
                            for o, r in zip(enc_out, reps)],
                           batch_first=True, padding_value=40.0)
    if mel_max_len:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens, reps.sum(dim=1)


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.fc = nn.Linear(filter_size, 1, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out.squeeze(-1)
40

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
        del max_seq_len  # unused

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
        # n_speakers = 10

        self.speaker_emb = None
        # self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

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

        self.proj = nn.Linear(out_fft_output_size, n_mel_channels, bias=True)

    def forward(self, inputs, dur_tgt, speaker=None, use_gt_durations=True, use_gt_pitch=False,
                pace=1.0, max_duration=75):
        mel_max_len = 400

        if self.speaker_emb is None:
            spk_emb = 0
        else:
            # print(speaker)
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)
        pred_enc_out, pred_enc_mask = enc_out, enc_mask
        log_dur_pred = self.duration_predictor(pred_enc_out, pred_enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)
        len_regulated, dec_lens, actual_lens = regulate_len(
            dur_tgt if use_gt_durations else dur_pred,
            enc_out, pace, mel_max_len, training=True)
        # print(len_regulated.shape, dec_lens.shape)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        return mel_out, dec_mask, dur_pred, log_dur_pred, dec_lens

    def infer(self, inputs, input_lens, pace=1.0, dur_tgt=None, pitch_tgt=None,
              pitch_transform=None, max_duration=400, speaker=0):
        del input_lens  # unused
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT

        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Embedded for predictors
        pred_enc_out, pred_enc_mask = enc_out, enc_mask

        # Predict durations
        log_dur_pred = self.duration_predictor(pred_enc_out, pred_enc_mask)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)
        # print(dur_pred)
        len_regulated, dec_lens, _ = regulate_len(
            dur_pred if dur_tgt is None else dur_tgt,
            enc_out, pace, mel_max_len=None)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        dec_out = dec_out*dec_mask
        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        phon_lens = enc_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py

        return mel_out, dec_lens, dur_pred, phon_lens
