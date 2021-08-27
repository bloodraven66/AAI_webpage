import speech_recognition as sr
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

def predict(audio_file, model='wav2vec2'):
    if model == 'spinx':
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        return r.recognize_sphinx(audio)

    elif model == 'wav2vec2':
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        speech, _ = sf.read(audio_file)
        input_values = tokenizer(speech, return_tensors="pt", padding="longest").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)
        return transcription[0]
    else:
        raise NotImplementedError
