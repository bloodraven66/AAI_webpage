from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
import io
import soundfile as sf
import base64
import wave
import scipy.io.wavfile as wavf
import numpy as np
from fastapi.responses import FileResponse
import uvicorn
import base64
import predict

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class getaData(BaseModel):
    file: bytes
    key: bytes

@app.get("/")
def read_root():
    return {"hello":"word"}

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelParams():
    def __init__(self):
        self.model_name = 'blstm'
        self.ASRStatus = False
        self.TTSStatus = False
        self.cc_status = {'active':False,
                        'text':None,
                        'audio':None,
                        'noise':None,
                        'snr':None,
                        'audEma':None,
                        'textEma':None}
        self.return_cc = False
        self.mean = None
        self.std = None
        self.Noise_Dist = 'none'
        self.Noise_SNR = 'none'
        self.multi_user_data = {}

def key_timeout():
    for key in args.multi_user_data:
        if time.time() - args.multi_user_data[key][0]> 60*10:
            args.multi_user_data.pop(key, None)

def insert_to_dict(key, tag, value):
    vals = {'time':0, 'audio_name':1, 'aai_model':2, 'text':3, 'pta_model':4, 'aai_ema':5, 'pta_ema':6, 'cc_status':7, 'mean':8, 'std':9}
    if key not in args.multi_user_data:
        args.multi_user_data[key] = [False]*len(vals)
    args.multi_user_data[key][0] = time.time()
    args.multi_user_data[key][vals[tag]] = value
    key_timeout()

def extract_from_dct(key, tag):
    vals = {'time':0, 'audio_name':1, 'aai_model':2, 'text':3, 'pta_model':4, 'aai_ema':5, 'pta_ema':6, 'cc_status':7, 'mean':8, 'std':9}
    args.multi_user_data[key][0] = time.time()
    key_timeout()
    return args.multi_user_data[key][vals[tag]]

def reset_key(key):
    vals = {'time':0, 'audio_name':1, 'aai_model':2, 'text':3, 'pta_model':4, 'aai_ema':5, 'pta_ema':6, 'cc_status':7, 'mean':8, 'std':9}
    if key not in args.multi_user_data:
        args.multi_user_data[key] = [False]*len(vals)
    args.multi_user_data[key][0] = time.time()
    key_timeout()




def pred_ema(key):
    audio_name = extract_from_dct(key, "audio_name")
    model_name = extract_from_dct(key, "aai_model")
    modelClass = predict.Model()
    if args.model_name == None: return 'Model not selected'
    plotname, ema = modelClass.make_plot(audio=audio_name, select_model=model_name.lower(),
                                    noise={'source':None, 'SNR':None},  allow_asr=args.ASRStatus)
    insert_to_dict(key, 'aai_ema', ema)
    if extract_from_dct(key, 'pta_ema') is not False:
        m, std = modelClass.correlate( extract_from_dct(key, 'pta_ema'), extract_from_dct(key, 'aai_ema'))
        insert_to_dict(key, 'mean', m)
        insert_to_dict(key, 'std', std)
        insert_to_dict(key, 'cc_status', True)
    return plotname

def pred_ema_from_text(key):
    text = extract_from_dct(key, "text")
    model_name = extract_from_dct(key, "pta_model")
    modelClass = predict.Model()
    plotname, ema = modelClass.make_plot(text = text, mode='p2e', select_model=model_name.lower())
    insert_to_dict(key, 'pta_ema', ema)
    if extract_from_dct(key, 'aai_ema') is not False:
        m, std = modelClass.correlate( extract_from_dct(key, 'pta_ema'), extract_from_dct(key, 'aai_ema'))
        insert_to_dict(key, 'mean', m)
        insert_to_dict(key, 'std', std)
        insert_to_dict(key, 'cc_status', True)
    return plotname


@app.post("/reset_params/")
def receive_selected_model(file: getaData):
    if file.file.decode("utf-8") == 'reset':
        key = file.key.decode("utf-8")
        reset_key(key)
        return {'reset activated'}
    else:
        return {'No action performed'}

@app.post("/sendModelName/")
def receive_selected_model(file: getaData):
    model_name = file.file.decode("utf-8")
    key = file.key.decode("utf-8")
    insert_to_dict(key, 'aai_model', model_name)
    return {'model selection received'}

@app.post("/sendModelNamePTA/")
def receive_selected_model(file: getaData):
    model_name = file.file.decode("utf-8")
    key = file.key.decode("utf-8")
    insert_to_dict(key, 'pta_model', model_name)
    return {'PTA model selection received'}

@app.post("/sendASRStatus/")
def receive_selected_model(file: getaData):
    args.ASRStatus = file.file.decode("utf-8").lower()
    return {'PTA model selection received'}

@app.post("/sendTTSStatus/")
def receive_selected_model(file: getaData):
    args.TTSStatus = file.file.decode("utf-8").lower()
    return {'PTA model selection received'}


@app.post("/NoiseParams/")
def receive_selected_model(file: getaData):
    args.Noise_Dist = file.file.decode("utf-8").lower()
    print(args.Noise_Dist)
    return {'PTA model selection received'}

@app.post("/sendNoiseSNR/")
def receive_selected_model(file: getaData):
    args.Noise_SNR = file.file.decode("utf-8").lower()
    print(args.Noise_SNR)
    return {'PTA model selection received'}

@app.post("/sendText/")
def receive_text(file: getaData):
    key = file.key.decode("utf-8")
    text = file.file.decode("utf-8")
    insert_to_dict(key, 'text', text)
    return {'done'}

@app.post("/plot_p2e/")
def fetch_preds_fastspeech(file: getaData):
    key = file.key.decode("utf-8")

    img = pred_ema_from_text(key)
    with open(img, 'rb') as image_file:
        encoded_image_string = base64.b64encode(image_file.read()).decode('utf-8')
    payload = {
       "mime" : "image/jpg",
       "image": encoded_image_string,
       "some_other_data": None
   }
    print('payload sent')
    return payload

@app.post("/plot/")
def fetch_preds(file: getaData):
    key = file.key.decode("utf-8")

    img = pred_ema(key)
    with open(img, 'rb') as image_file:
        encoded_image_string = base64.b64encode(image_file.read()).decode('utf-8')
    payload = {
       "mime" : "image/jpg",
       "image": encoded_image_string,
       "some_other_data": None
   }
    print('payload sent')
    return payload


@app.post("/getcc/")
def fetch_preds_fastspeech(file: getaData):
    key = file.key.decode("utf-8")
    return_cc = extract_from_dct(key, 'cc_status')
    if return_cc is not False:
        payload = {
           "mime" : "str",
           "mean": str(extract_from_dct(key, 'mean')),
           "std": str(extract_from_dct(key, 'std'))
       }
        print('cc sent')
        insert_to_dict(key, 'cc_status', False)
    else:
        payload = {
           "mime" : "str",
           "mean": 'none',
           "std": 'none'
       }
    return payload


@app.post("/upload/")
async def fetch(key: str = Form(...), myFile: UploadFile = Form(...)):
    print(key)
    audio_name = f'../audio_data/{key}.wav'
    with open(audio_name, "wb+") as file_object:
        file_object.write(myFile.file.read())
    insert_to_dict(key, 'audio_name', audio_name)
    return {'done'}

@app.post("/uploadbytes/")
async def fetch2(key: str = Form(...), file: UploadFile = File(...)):
    data = file.file.read()
    audio_name = f'../audio_data/{key}.wav'
    with open(audio_name, 'wb+') as file_object:
        file_object.write(data)
    insert_to_dict(key, 'audio_name', audio_name)
    return {'done'}


if __name__ == "__main__":
    args = ModelParams()
    uvicorn.run(app,
                host="0.0.0.0",
                port=3001,
                ssl_keyfile="./key.pem",
                ssl_certfile="./cert.pem"
                )
