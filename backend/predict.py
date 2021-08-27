import warnings
warnings.filterwarnings("ignore")
import logging
import asr
import numpy as np
import scipy.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import keras
import sys
import numpy
import gc
from scipy.spatial.distance import euclidean
import librosa
import tensorflow as tf
# if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.get_logger().setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import HTK
from keras.preprocessing.sequence import pad_sequences
from scipy.stats import pearsonr
from matplotlib import cm
from nnmnkwii.preprocessing import delta_features
import pickle
#from nnmnkwii.baseline.gmm import MLPG
from nnmnkwii.paramgen import mlpg
from Utils_GMM import GMM_M
from KalmanSmoother import *
from scipy.signal import butter, filtfilt
from scipy import signal
from matplotlib import gridspec
from nnmnkwii.util import trim_zeros_frames, remove_zeros_frames
from keras.backend.tensorflow_backend import set_session
import infer_aai, infer_p2e
from g2p_en import G2p
from tacotron.infer_taco import infer_taco
from numpy import linalg as LA
from fastdtw import fastdtw
from scipy.io import wavfile
import wavio

class Model():
    def __init__(self):
        self.NoUnits = 32
        self.n_mfcc = 13
        self.inputDim = self.n_mfcc*3
        self.OutDim = 128
        self.windows = [(0, 0, np.array([1.0])),
                   (1, 1, np.array([-0.5, 0.0, 0.5])),
                 (1, 1, np.array([1.0, -2.0, 1.0])),]
        self.rate = 22050
        self.hop_length = int(0.010*self.rate)
        self.n_fft = int(0.020*self.rate)
        self.fb, self.fa = signal.cheby2(10,40,7.0/(100/2),'low', analog=False)
        self.blstm_model = 'Models/BLSTM_SD_AAI_Batch5_Tanaya_LSTMunits_32__.h5'
        self.cnn_model = 'Models/CNN_6H_64F_SD_AAI_Batch5_Tanaya_LSTMunits_256__.h5'
        self.dnn_model = 'Models/DNN_H1s_SD_AAI_Batch5_Tanaya_LSTMunits_128__.h5'
        self.gmm_model = 'Models/Tanaya32_full_finalized_model.sav'
        self.transformer_model = 'Models/aai_34sub_relative.pth'
        # self.transformer_model = 'Models/aai_ft_Vignesh_relative.pth'
        # self.fastspeech_model = 'Models/text2ema_small_pooled_34sub.pth'
        self.fastspeech_model = 'Models/text2ema_small_Vignesh_refine.pth'
        self.tacotron_model = 'tacotron/outdir/checkpoint_17200'
        self.allow_title = False
        self.model_name = {'lstm':self.blstm_model,
                            'cnn':self.cnn_model,
                            'dnn':self.dnn_model,
                            'gmm':self.gmm_model,
                            'transformer':self.transformer_model,
                            'fastspeech':self.fastspeech_model,
                            'tacotron2':self.tacotron_model}

        self.device = 'cpu'
        # self.device = 'cpu'
        self.plot12_savename = '../test_data/12_re.png'
        self.plotfull_savename = '../test_data/full_re.jpg'
        self.plot12row_savename = '../test_data/demo.jpg'
        self.labels = ['ULx', 'ULy', 'LLx', 'LLy', 'Jawx', 'Jawy', 'TTx', 'TTy', 'TBx', 'TBy', 'TDx', 'TDy']
        self.std_frac = 0.25
        self.normalise_mfcc = True
        self.delta_features = True
        self.babble_noise_path = 'noise/speech_babble_16k.wav'
        self.hfchannel_noise_path = 'noise/speech_babble_16k.wav'
        self.pink_noise_path = 'noise/speech_babble_16k.wav'
        self.mean = mean = [10.13867092, 8.35774779 , 11.6490379 ,  -3.47554931,  -1.32870194,
         -42.22969279,  -8.96651629,   4.41072592, -18.06819539,   8.16558618,
         -24.21579873,  11.80944184]
        self.std = [0.19126671, 0.53748644, 0.31821568, 1.81918336, 0.44596135, 0.70038397,
         1.44398989, 1.13724842, 1.97694945, 1.05962662, 2.06794848, 1.67578123]
        self.deNorm = True

    def insert_noise(self, signal, noise):
        print(noise['source'])
        if noise['source'] == 'Gaussian':
            samples = np.random.normal(0, 1, len(signal))
        elif noise['source'] == 'babble':
            samples = librosa.load(self.babble_noise_path, sr=22050)[0].squeeze()
            start_idx = np.random.randint(0, len(samples)-len(signal)-1)
            samples = samples[start_idx:start_idx+len(signal)]
        elif noise['source'] == 'hfchannel':
            samples = librosa.load(self.hfchannel_noise_path, sr=22050)[0].squeeze()
            start_idx = np.random.randint(0, len(samples)-len(signal)-1)
            samples = samples[start_idx:start_idx+len(signal)]
        elif noise['source'] == 'pink':
            samples = librosa.load(self.pink_noise_path, sr=22050)[0].squeeze()
            start_idx = np.random.randint(0, len(samples)-len(signal)-1)
            samples = samples[start_idx:start_idx+len(signal)]
        else:
            raise Exception('Noise source not defined')

        alpha = LA.norm(signal)/(LA.norm(samples)*10**(0.05*noise['SNR'])); # samples: Noise signal
        return np.add(signal, alpha*samples)


    def preprocess(self, audio, noise):
        signal, rate = librosa.load(audio, sr=22050)
        # rate = wavio.read(audio)
        if noise['source'] is not None:
            signal = self.insert_noise(signal.squeeze(), noise)
        M_t = librosa.feature.mfcc(signal, self.rate, n_mfcc=self.n_mfcc, hop_length=self.hop_length, n_fft=self.n_fft).T
        if self.delta_features:
            M_t = delta_features(M_t, self.windows)

        if self.normalise_mfcc:
            mean_G = np.mean(M_t, axis=0)
            std_G = np.std(M_t, axis=0)
            M_t = self.std_frac*(M_t-mean_G)/std_G
        return M_t

    def preprocess_text(self, text):
        g2p = G2p()
        phonemes = g2p(text)
        phoneme_list = []
        for phone in phonemes:
            if phone != ' ':
                p_list = ''
                for p in phone:
                    if not p.isdigit():
                        p_list += p.lower()
                phoneme_list.append(p_list)
        return phoneme_list


    def predict_aai(self, audio, select_model, noise):
        M_t = self.preprocess(audio, noise)
        if select_model not in self.model_name.keys():
            raise  Exception('Model name not found.. select dnn, cnn, lstm or gmm')

        if select_model == 'gmm':
            gmm = pickle.load(open(self.model_name[select_model], 'rb'))
            paramgen = GMM_M(gmm, windows=self.windows)
            pred, D, W = paramgen.transform(M_t)

        elif select_model == 'transformer':
            return infer_aai.predict(m_t=M_t,
                                    device=self.device,
                                    checkpoint=self.model_name['transformer'])

        else:
            model = keras.models.load_model(self.model_name[select_model])
            M_t = np.expand_dims(M_t, 0)
            pred = np.squeeze(model.predict(M_t))
            D = np.tile(np.var(pred,axis=0),(pred.shape[0],1))
            W = self.windows

        pred_wom = pred
        pred_wm = mlpg(pred, D, W) #MLPG
        pred_kf = kalmansmooth(pred.transpose()).transpose()
        pred_lpf = filtfilt(self.fb, self.fa, pred.transpose()).transpose()
        keras.backend.clear_session()
        return  {'raw':pred_wom, 'mlpg':pred_wm, 'kf':pred_kf, 'lpf':pred_lpf}

    def predict_pta(self, text, select_model):
        phonemes = self.preprocess_text(text)

        if select_model == 'fastspeech':
            ema, dur_pred = infer_p2e.predict(phonemes, self.model_name[select_model], self.device)

        elif select_model == 'tacotron2':
            ema, dur_pred = infer_taco(phonemes, self.model_name[select_model], self.device)

        else:
            raise Exception('PTA Model not found')

        return ema,dur_pred, phonemes

    def plot_separate_subplots(self, p, select_model, aud):
        fig, ax = plt.subplots(3, 4)
        for i in range(3):
            for j in range(4):
                ax[i][j].plot(p[:, i*4+j]   )
        plt.tight_layout()
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        plt.savefig(self.plot12_savename)
        return self.plot12_savename

    def plot_12rows(self, p, select_model, aud, boundaries):
        ymin = np.min(p, axis=0)
        ymax = np.max(p, axis=0)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        numRows = 12 if aud is None else 13
        offset = 0
        if select_model == 'tacotron2':
            offset = 1
            numRows = 13
        fig, ax = plt.subplots(numRows, 1, figsize=(7, 6))
        color=iter(cm.plasma(np.linspace(0,1,12)))
        fig.text(0.015, 0.5, 'displacement (mm)', ha='center', rotation='vertical')

        if aud is not None:
            ax[0].specgram(aud[0])
            # fig.text(0.9, 0.9, 'Spec')
            ax[0].get_xaxis().set_visible(False)
            offset = 1
            ax[0].spines['bottom'].set_color('#dddddd')
            ax[0].spines['right'].set_color('#dddddd')
        if boundaries is not None:
            phones, dur = boundaries
            if select_model == 'fastspeech':
                dur = [sum(dur[:i]) for i in range(len(dur)+1)]
            else:
                dur = dur.squeeze()
        for i in range(12):
            c=next(color)

            ax[i+offset].plot(p[:, i], label=self.labels[i], color=c)
            ax[i+offset].set_xlim(0, len(p))
            ax[i+offset].legend( bbox_to_anchor=(1, 1), loc='upper left')

            ax[i+offset].spines['top'].set_visible(False)
            # ax[i+offset].spines['right'].set_visible(False)
            ax[i+offset].get_xaxis().set_visible(False)
            ax[i+offset].spines['right'].set_color('#dddddd')
            if i != 11:

                ax[i+offset].spines['bottom'].set_color('#dddddd')
        ax[0].spines['top'].set_visible(True)
        ax[0].spines['top'].set_color('#dddddd')
        ax[11+offset].get_xaxis().set_visible(True)
        if boundaries is not None:
            track = {}
            color=iter(cm.rainbow(np.linspace(0,1,len(np.array(phones).ravel()))))
            if select_model == 'fastspeech':
                for i in range(len(dur)-1):
                    c=next(color)
                    if phones[i] in track:
                        c = track[phones[i]]
                    else:
                        track[phones[i]] = c
                    ax[11].text(dur[i]+1, ymin[11]+0.1, phones[i])
                    for j in range(12):
                        ax[j].vlines(dur[i], ymin=ymin[j], ymax=ymax[j], linestyles='dotted', color='black')
            elif select_model == 'tacotron2':
                for i in range(dur.shape[1]):
                    ax[0].plot(dur[:, i])
                ax[0].set_xlim(0, len(dur))
        if self.allow_title:
            ax[0].set_title(f'{select_model.upper()} prediction')
        plt.xticks(ticks=[i for i in range(len(p)) if i%100==0], labels=[int(i/100) for i in range(len(p)) if i%100==0])
        if len(p) < 100:
            plt.xticks(ticks=[0, len(p)], labels=[0, len(p)/100])
        plt.xlabel('time (sec)')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0.07)
        plt.savefig(self.plot12row_savename)
        return self.plot12row_savename

    def plot_in_one(self, p, select_model, aud):
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 3])
        ax0 = fig.add_subplot(gs[0])
        plt.title(f'{select_model.upper()} prediction')
        color=iter(cm.plasma(np.linspace(0,1,12)))
        ax0.specgram(aud[0])
        ax0.get_xaxis().set_visible(False)
        ax1 = fig.add_subplot(gs[1])
        for i in range(12):
            c=next(color)
            ax1.plot(p[:, i], label=self.labels[i], color=c)
        ax1.legend(bbox_to_anchor=(1., 1), loc='upper left')
        ax1.set_xlim(0, len(p))
        plt.xticks(ticks=[i for i in range(len(p)) if i%100==0], labels=[int(i/100) for i in range(len(p)) if i%100==0])
        plt.xlabel('time (sec)')
        plt.ylabel('displacement (mm)')
        plt.savefig(self.plotfull_savename, bbox_inches='tight')
        return self.plotfull_savename

    def correlate(self, textEma, audEma):
        textEma = textEma.squeeze()
        audEma = audEma.squeeze()
        if textEma.shape[0] == 12:
             textEma = textEma.T
        if audEma.shape[0] == 12:
             audEma = audEma.T
        if textEma.shape[1] !=12 or audEma.shape[1] != 12:
            raise Exception('check ema dimensions')
        X = textEma
        Y = audEma
        coefficients_ = []
        dis, pth = fastdtw(X,Y , dist=euclidean)
        for artic in range(0,12):
            out, gt = [], []
            for i in range(0,len(pth)):
                out.append(Y[pth[i][1]][artic])
                gt.append(X[pth[i][0]][artic])
            coef=pearsonr(out,gt)[0]
            coefficients_.append(coef)
        cc_mean = np.mean(coefficients_)
        cc_std = np.std(coefficients_)
        return round(cc_mean, 3), round(cc_std, 3)
    # break


    def make_plot(self,
                audio=None,
                text=None,
                select_model=None,
                display='mlpg',
                mode='aai',
                allow_asr=False,
                noise={'source':None, 'SNR':None}):

        if mode == 'aai':
            if audio is None or select_model is None:
                raise Exception('no input for aai')
            if select_model == 'transformer':
                self.normalize = False
                self.delta_features = False
                ema = self.predict_aai(audio, select_model, noise)
            else:
                ema = self.predict_aai(audio, select_model, noise)[display]
            audio_info = librosa.load(audio)
            boundaries = None
            if allow_asr:
                text = asr.predict(audio, 'spinx')
                print(text)
                asr_ema, asr_durations, asr_phonemes = self.predict_pta(text, 'fastspeech')
                asr_boundaries = [asr_phonemes, asr_durations]
                asr_audio_info = None
        elif mode=='p2e':
            if text is None:
                raise Exception('no input for p2e')
            ema, durations, phonemes = self.predict_pta(text, select_model)
            audio_info = None
            boundaries = [phonemes, durations]

        else:
            raise NotImplementedError
        if self.deNorm:
            ema = ema[:, ]* self.std + self.mean
        print(select_model, np.mean(ema), np.max(ema), np.min(ema))
        plotname = self.plot_12rows(ema, select_model, audio_info, boundaries=boundaries)
        if allow_asr:
            asr_plotname = self.plot_12rows(asr_ema, 'ASR+FASTSPEECH', asr_audio_info, boundaries=asr_boundaries)
        gc.collect()
        return plotname, ema

# modelClass = Model()
# modelClass.make_plot(audio='../test_data/record.wav', select_model='lstm', allow_asr=False)
# modelClass.make_plot(mode='p2e', text='this is a recording to test the webpage', select_model='tacotron2')
# p = modelClass.predict('../test_data/test_model_with_mfcc.wav', 'gmm')
# with open('../test_data/dummy_data.npy', 'wb') as f:
#     p = np.save(f, p['mlpg'])
