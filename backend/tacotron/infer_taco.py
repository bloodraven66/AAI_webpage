import numpy as np
import torch
import sys
sys.path.append('tacotron')
from model import Tacotron2
from hparams import create_hparams
from text import text_to_sequence
import sys
from phoneme_to_seq import *
import os

def infer_taco(text_path, checkpoint_path, device):
    hparams=create_hparams()
    # checkpoint_path = os.getcwd()+"/tacotron/outdir/checkpoint_17200"
    model = Tacotron2(hparams).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
    _ = model.to(device).eval().half()
    dct=prep_dct()
    sequence = np.array([list(map(dct.get,text_path))]) #np.array(to_sequence(text_path,dct))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
    torch.manual_seed(1234)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    return mel_outputs_postnet.detach().cpu().numpy().squeeze().T ,alignments.detach().cpu().numpy()

if __name__=="__main__":
    out,align = infer_taco(sys.argv[1], sys.argv[2], sys.argv[3])
    print(out.shape, align.shape)
