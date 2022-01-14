# Codes for INTERSPEECH 2021 show and tell paper - Webpage interface for articulatory movement estimation for speech prediction

<h4>INFO:</h2>

The webpage is now offline. You can host it locally. The steps shown below are for Ubuntu.

Front end:

- clone this repository
- install npm
- cd frontend
- npm init (first time only)
- npm install (first time only)
- npm start (to start frontend, it will open in localhost)

Backend:

- Install docker with 'snap install docker'
- Backend files are shared <a href='https://zenodo.org/deposit/5849094'>here</a>. You can either
   1. download the saved docker image (5.2gb) with 'webpage_docker_backup.tar' or
   2. download the files (535mb) at 'aai_pta_viz.zip'
- After 1, run 'sudo docker load -i webpage_docker_backup.tar'
- After 2, unzip aai_pta_viz.zip, followed by 'sudo docker build -t aai_pta_viz aai_pta_viz'
- To start backend, run 'sudo docker run --net=host aai_pta_viz'

Tasks:

- Acoustic to articulatory inversion (AAI)
- Phoneme to articulatory estimation (PTA)

Models:

- Hidden Markov Model (HMM)
- Gaussian Mixture Model (GMM)
- Deep Neural Networks (DNN)
- Convolutional Neural Network (CNN)
- Long Short Term Memory networks (LSTM)
- Transformers
- Tacotron
- FastSpeech

Frameworks:

- Frontend: ReactJS
- Backend Inference: Pytorch & Keras
- Backend Hosting: Python Fastapi + gunicorn

Todo:
- Saving live recording is broken with docker currently, will have a workaround for it soon.
