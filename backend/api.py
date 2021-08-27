from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import io
import soundfile as sf
import base64
import wave
import scipy.io.wavfile as wavf
import numpy as np
from fastapi.responses import FileResponse
import uvicorn
import base64
# import predict

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class getaData(BaseModel):
    file: bytes

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
        self.save_name = 'interspeech_demo2.wav'
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

def pred_ema():
    pass
    # exit()
    # modelClass = predict.Model()
    # if args.model_name == None: return 'Model not selected'
    # print(args.model_name.lower())
    # if args.Noise_Dist != 'none' and args.Noise_SNR != 'none':
    #     noise = {'source':args.Noise_Dist, 'SNR':float(args.Noise_SNR)}
    # else:
    #     noise = {'source':None, 'SNR':None}
    # plotname, ema = modelClass.make_plot(audio=args.save_name, select_model=args.model_name.lower(),
    #                                 noise=noise,  allow_asr=args.ASRStatus)
    # args.cc_status['active'] = True
    # args.cc_status['audio'] = args.model_name
    # args.cc_status['noise'] = args.Noise_Dist
    # args.cc_status['snr'] = args.Noise_SNR
    # args.cc_status['audEma'] = ema
    # if args.cc_status['text'] is not None and args.cc_status['active']  == True:
    #     m, std = modelClass.correlate(args.cc_status['textEma'], args.cc_status['audEma'])
    #     print(m, std, args.cc_status['audio'], args.cc_status['text'])
    #     args.mean = m
    #     args.std = std
    #     args.return_cc = True
    # return plotname

def pred_ema_from_text():


    modelClass = predict.Model()
    plotname, ema = modelClass.make_plot(text = args.text, mode='p2e', select_model=args.PTA_model_name)
    args.cc_status['active'] = True
    args.cc_status['text'] = args.PTA_model_name
    args.cc_status['textEma'] = ema
    if args.cc_status['audio'] is not None and args.cc_status['active']  == True:
        m, std = modelClass.correlate(args.cc_status['textEma'], args.cc_status['audEma'])
        print(m, std, args.cc_status['audio'], args.cc_status['text'])
        args.mean = m
        args.std = std
        args.return_cc = True
    return plotname


@app.post("/reset_params/")
def receive_selected_model(file: getaData):
    if file.file.decode("utf-8") == 'reset':
        print('reseting state')
        for key in args.cc_status:
            args.cc_status[key] = None
        args.cc_status['active'] = False
        return {'reset activated'}
    else:
        return {'No action performed'}

@app.post("/sendModelName/")
def receive_selected_model(file: getaData):
    args.model_name = file.file.decode("utf-8")

    return {'model selection received'}

@app.post("/sendModelNamePTA/")
def receive_selected_model(file: getaData):
    args.PTA_model_name = file.file.decode("utf-8").lower()
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
    print('recieved')
    args.text = file.file.decode("utf-8")
    return {'done'}

@app.post("/plot_p2e/")
def fetch_preds_fastspeech(file: getaData):
    # img = '../test_data/full.png'
    img = pred_ema_from_text()
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
    pass
    # img = '../test_data/full.png'
   #  img = pred_ema()
   #  with open(img, 'rb') as image_file:
   #      encoded_image_string = base64.b64encode(image_file.read()).decode('utf-8')
   #  payload = {
   #     "mime" : "image/jpg",
   #     "image": encoded_image_string,
   #     "some_other_data": None
   # }
   #  print('payload sent')
   #  return payload


@app.post("/getcc/")
def fetch_preds_fastspeech(file: getaData):

    if args.return_cc:
        payload = {
           "mime" : "str",
           "mean": str(args.mean),
           "std": str(args.std)
       }
        print('cc sent')
        args.return_cc = False
    else:
        payload = {
           "mime" : "str",
           "mean": 'none',
           "std": 'none'
       }
    return payload


@app.post("/upload/")
async def fetch(myFile: UploadFile = Form(...)):
    print('uploaded')
    with open(args.save_name, "wb+") as file_object:
        file_object.write(myFile.file.read())
    return {'done'}

@app.post("/uploadbytes/")
async def fetch2(file: UploadFile = File(...)):
    print('trying')
    # import wave
    # import base64
    # audio_bytes = myFile.file.read()

    # signal, sr = sf.read(io.BytesIO(audio_bytes))
    # print(sr)
    # exit()
    import librosa
    # print(myFile.file.read()[:10])
    data = file.file.read()
    with open(args.save_name, 'wb+') as file_object:
        # print()
        file_object.write(data)
    y, sr = librosa.load(args.save_name)
    print(len(y)/sr)
    # myFile

    # import soundfile as sf
    # data, samplerate = sf.read(io.BytesIO(file.file.read()))
    # print(samplerate)
    return {'done'}


if __name__ == "__main__":
    args = ModelParams()
    uvicorn.run(app,
                host="0.0.0.0",
                port=3001,
                ssl_keyfile="./key.pem",
                ssl_certfile="./cert.pem"
                )
