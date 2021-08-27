import React, { Component } from "react";
import { BrowserRouter, Route } from "react-router-dom";
// import RecordingPage from '/home/sathvik/Documents/AAI_webpage/frontend/src/recordingPage'
import RecordingPage from '/data1/Code/Sathvik/AAI_webpage/frontend/src/recordingPage'
import PlotPage from '/data1/Code/Sathvik/AAI_webpage/frontend/src/plotPage'
var jsonResponseMean = 0;
var jsonResponseSTD = 0;
class LandingPage extends Component {

  state = {
    sentenceId: 0,
    showDoneModal: false,
    seconds: 0,
    totalDurationRecorded: 0,
    totalRecordingsRecorded: 0,
    recording: false,
    selectedFile: null,
    mean: 0,
    std: 0
  };

  snapshot = null;
  totalSentences = null;
  userDoc = null;


  onFileChange = event => {
    this.setState({ selectedFile: event.target.files[0] });
    document.getElementById("uploadFileButton").disabled = false;
  };

  // <button id='playUploaded' class='btn-design' disabled={!this.state.value}>
  //   Play
  // </button>
  async componentDidMount() {
    document.title = "Articulatory Estimation"
    window.scrollTo(-100, -100)
    let recordAudio = () =>
      new Promise(async (resolve) => {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        const mediaRecorder = new MediaRecorder(stream);
        let audioChunks = [];

        mediaRecorder.addEventListener("dataavailable", (event) => {
          audioChunks.push(event.data);
        });

        const start = () => {
          audioChunks = [];
          mediaRecorder.start();
        };

        const stop = () =>
          new Promise((resolve) => {
            mediaRecorder.addEventListener("stop", () => {
              const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
              const audioUrl = URL.createObjectURL(audioBlob);
              const audio = new Audio(audioUrl);
              const play = () => audio.play();
              resolve({ audioChunks, audioBlob, audioUrl, play });
            });

            mediaRecorder.stop();
          });

        resolve({ start, stop });
      });

    const recordButton = document.getElementById("start");
    const playUploaded = document.getElementById("playUploaded");
    const stopButton = document.getElementById("stop");
    const playButton = document.getElementById("play");
    const saveButton = document.getElementById("save");

    let recorder;
    let audio;
    let interval;

    recordButton.addEventListener("click", async () => {
      recordButton.setAttribute("disabled", true);

      if (!recorder) {
        recorder = await recordAudio();
      }
      recorder.start();

      this.setState({ seconds: 0, recording: true });
      interval = setInterval(
        () => this.setState({ seconds: this.state.seconds + 1 }),
        1000
      );
      stopButton.removeAttribute("disabled");
      playButton.setAttribute("disabled", true);
      saveButton.setAttribute("disabled", true);
    });

    stopButton.addEventListener("click", async () => {
      recordButton.removeAttribute("disabled");
      stopButton.setAttribute("disabled", true);
      playButton.removeAttribute("disabled");
      saveButton.removeAttribute("disabled");
      audio = await recorder.stop();
      this.setState({ recording: false });
      clearInterval(interval);
    });
    // onFileChange = event => {
    //   this.state.playUploaded.removeAttribute("disabled");
    //   this.setState({ selectedFile: event.target.files[0] });
    // };

    playButton.addEventListener("click", () => {
      audio.play();
      });
    // playUploaded.addEventListener("click", () => {
    //     this.state.selectedFile.play();
    //     });
    saveButton.addEventListener("click", () => {
    const XHRrecord = new XMLHttpRequest();
    const FD = new FormData()
    FD.append('file', audio.audioBlob);
    XHRrecord.open('POST', "http://10.64.26.89:3001/predict/");
    XHRrecord.send(FD);
    });


  }

  handleChange(e) {
  let isChecked = e.target.checked;
  const XHRasr = new XMLHttpRequest();
  XHRasr.open('POST', "http://10.64.26.89:3001/sendASRStatus/");
  XHRasr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  XHRasr.send(JSON.stringify({ "file": isChecked}))


}
  handleChangeTTS(e) {
  let isChecked = e.target.checked;
  const XHRasr = new XMLHttpRequest();
  XHRasr.open('POST', "http://10.64.26.89:3001/sendTTSStatus/");
  XHRasr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  XHRasr.send(JSON.stringify({ "file": isChecked}))


  }



  onFileUpload = () => {
    console.log('button pressed')
    this.setState({ showStore: true });
    const formData = new FormData();
    formData.append(
      "myFile",
      this.state.selectedFile
    );

    var e = document.getElementById("ddlViewBy");
    var strUser = e.options[e.selectedIndex].text;

      const XHR2 = new XMLHttpRequest();
      XHR2.open('POST', "http://10.64.26.89:3001/sendModelName/");
      XHR2.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
      XHR2.send(JSON.stringify({ "file": strUser}))

    var e2 = document.getElementById("ddlViewByNoise");
    var strUser2 = e2.options[e2.selectedIndex].text;

        const XHR3 = new XMLHttpRequest();
        XHR3.open('POST', "http://10.64.26.89:3001/NoiseParams/");
        XHR3.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        XHR3.send(JSON.stringify({ "file": strUser2}))
    var noiseSNR = document.getElementById("inputNoise");
    console.log(noiseSNR)
    var strUser3 = noiseSNR.options[noiseSNR.selectedIndex].text;
    const XHR4 = new XMLHttpRequest();
    XHR4.open('POST', "http://10.64.26.89:3001/sendNoiseSNR/");
    XHR4.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    XHR4.send(JSON.stringify({ "file": strUser3}))


    const XHR = new XMLHttpRequest();
    XHR.open('POST', "http://10.64.26.89:3001/upload/");
    XHR.send(formData);

    XHR.onload  = function() {
      console.log('here')
   		var jsonResponse = JSON.parse(XHR.responseText);
      console.log(jsonResponse)
		if (jsonResponse == 'done'){

        const XHR = new XMLHttpRequest();
        XHR.open('POST', "http://10.64.26.89:3001/plot/");
        XHR.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        XHR.send(JSON.stringify({ "file": "ping"}))
        XHR.onload  = function() {
       		var jsonResponse = JSON.parse(XHR.responseText).image;
          document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;

          const XHRcc = new XMLHttpRequest();
          XHRcc.open('POST', "http://10.64.26.89:3001/getcc/");
          XHRcc.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
          XHRcc.send(JSON.stringify({ "file": "returnCC"}))
          XHRcc.onload  = function() {
            console.log(XHRcc.responseText)

            jsonResponseMean = JSON.parse(XHRcc.responseText).mean
            jsonResponseSTD = JSON.parse(XHRcc.responseText).std

            // document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;
            if (jsonResponseMean != 'none') {
              console.log(jsonResponseMean, jsonResponseSTD)
                document.getElementById("ccDiv").innerHTML = 'Correlation Coefficient: '+ jsonResponseMean+'('+jsonResponseSTD + ')';
                // this.setState({ showCCStore: true });

                  document.getElementById('ccDiv').style.display = 'block';
            }
            else {
            }
        };
      };







		}else{
    }
  };

  };


  onTextUpload = () => {
    var textentry = document.getElementById('inputText').value
      console.log(textentry)
      this.setState({ showStore: true });

      var e2 = document.getElementById("ddlViewByPTA");
      var strUser2 = e2.options[e2.selectedIndex].text;

        const XHR3 = new XMLHttpRequest();
        XHR3.open('POST', "http://10.64.26.89:3001/sendModelNamePTA/");
        XHR3.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        XHR3.send(JSON.stringify({ "file": strUser2}))

      const XHR2 = new XMLHttpRequest();
      XHR2.open('POST', "http://10.64.26.89:3001/sendText/");
      XHR2.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
      XHR2.send(JSON.stringify({ "file": textentry}))

      XHR2.onload  = function() {

     		var jsonResponse = JSON.parse(XHR2.responseText);
        console.log(jsonResponse)
  		if (jsonResponse == 'done'){

          const XHR = new XMLHttpRequest();
          XHR.open('POST', "http://10.64.26.89:3001/plot_p2e/");
          XHR.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
          XHR.send(JSON.stringify({ "file": "ping"}))
          XHR.onload  = function() {
         		var jsonResponse = JSON.parse(XHR.responseText).image;
            document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;

            const XHRcc = new XMLHttpRequest();
            XHRcc.open('POST', "http://10.64.26.89:3001/getcc/");
            XHRcc.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
            XHRcc.send(JSON.stringify({ "file": "returnCC"}))
            XHRcc.onload  = function() {
              console.log(XHRcc.responseText)
              jsonResponseMean = JSON.parse(XHRcc.responseText).mean
              jsonResponseSTD = JSON.parse(XHRcc.responseText).std
              // document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;
              if (jsonResponseMean != 'none') {
                // console.log(jsonResponseMean, jsonResponseSTD)
                document.getElementById("ccDiv").innerHTML = 'Correlation Coefficient: '+jsonResponseMean+'('+jsonResponseSTD + ')';
                  // this.setState({ showCCStore: true });
                  document.getElementById('ccDiv').style.display = 'block';
              }
              else {
                console.log('error')
              }
              console.log('shown')
        };



          // var jsonResponseMean = JSON.parse(XHRcc.responseText).mean
          // var jsonResponseSTD = JSON.parse(XHRcc.responseText).std
          // document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;
          // console.log(jsonResponseMean, jsonResponseSTD)
      };

  		}else{
      }
    }

  };

  disableImage = () => {
    document.getElementById('disable').setAttribute('disabled','disabled');
    const XHR = new XMLHttpRequest();
    XHR.open('POST', "http://10.64.26.89:3001/reset_params/");
    XHR.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    XHR.send(JSON.stringify({ "file": "reset"}))


  };





  render() {
    return (
      <div style={{overflowx: 'hidden', position:'relative', left:'10px'}}

      >
      <style>

      </style>

      <h2>Articulatory movement estimation in Speech Production</h2>
      <BrowserRouter>
      <Route path="/record" component={RecordingPage} />
      <Route path="/plot" component={PlotPage} />
      </BrowserRouter>


      <hr  class="dashed"></hr>

      <h3>Acoustic-to-Articulatory Inversion (AAI)</h3>

      <div>
      <input type="checkbox" defaultChecked={true}  onChange={e => this.handleChange(e)} />

        Perform ASR and estimate articulators from phonemes.<br></br>
      <h4>Upload an audio file</h4>
          <input type="file" onChange={this.onFileChange} accept="audio/wav, audio/mp3" id='uploadingFile' hidden/>
          <label class='btn-design-upload' for="uploadingFile">Select file</label>


          <button onClick={this.onFileUpload} class='btn-design' id='uploadFileButton' >
            Upload
          </button>  </div>

      <div style={{position:'relative', left:'50px'}}><h4>OR</h4></div>
      <div style={{position:'relative'}} >
      <h4>Record your audio</h4>
      </div>


      <script type="text/javascript">
           document.body.innerHTML = '';
       </script>
        <div className="flex-container">
          <button id="start" class='btn-design'>
            <span style={{ position: "relative" }}>Start</span>
          </button>
          <button
            id="stop"
            class='btn-design'
            disabled={true}
          >
            <span style={{ position: "relative" }}>Stop</span>
          </button>
          <button id="play" class='btn-design' disabled={true}>
            <span style={{ position: "relative" }}>Play</span>
          </button>
          <button id="save" class='btn-design' disabled={true}>
            <span style={{ position: "relative" }}>Upload</span>
          </button>
        </div>


        <div className="flex-container" style={{position:'relative', left:'450px',  top:'-295px'}}>
        <label for="ddlViewBy">Select Model</label><br></br>
        <select id="ddlViewBy">
        <option value='1'>GMM</option>
        <option value='2'>DNN</option>
        <option value='3'>CNN</option>
        <option value='4'>LSTM</option>
        <option value='5'>Transformer</option>
        </select>


        <div style={{position:'relative', left:'140px',  top:'-40px'}}>
        <label for="ddlViewByNoise">Select Noise</label><br></br>
        <select id="ddlViewByNoise">
        <option value='1'>None</option>
        <option value='2'>Gaussian</option>
        <option value='3'>Pink</option>
        <option value='4'>Babble</option>
        <option value='5'>HFChannel</option>
        </select>

        <div style={{position:'relative',  top:'-40px', left:'130px'}}>


        <label for="inputNoise">SNR(dB)</label><br></br>
        <select id="inputNoise">
        <option value='1'>None</option>
        <option value='2'>5</option>
        <option value='3'>10</option>
        <option value='4'>15</option>
        <option value='5'>20</option>
        </select>
        </div></div></div>

          <div style={{position:'relative',  top:'-100px'}}>
      <hr class='dashed'></hr>
      </div>

      <div style={{position:'relative',  top:'-100px'}}>
      <h3>Phoneme-to-Articulatory Estimation(PTA)</h3>
      <input type="checkbox" defaultChecked={true} onChange={e => this.handleChangeTTS(e)}/>
      Perform TTS!
      <br></br>
      <input name="searchTxt" type="text"  id="inputText"/>
      <br></br><br></br>


        <button onClick={this.onTextUpload} class='btn-design'>
          Submit Text
        </button>
          <div style={{position:'relative',  top:'30px'}}>
        <a  href='/'><button class='btn-design' id='disable' style={{display: this.state.showStore ? 'block' : 'none' }} onClick={this.disableImage}>Reset</button> </a>
      </div></div>


    <div style={{position:'absolute', left:'450px',  top:'405px'}}>
    <label for="ddlViewByPTA">Select Model</label><br></br>
    <select id="ddlViewByPTA">
    <option value='1'>FastSpeech  </option>
    <option value='2'>Tacotron2</option>
    </select>
    </div>

    <div style={{position:'absolute', right:'10px', top:'0px'}}>
    <img id="imageBox"/>
    <div id='ccDiv' style={{display: this.state.showCCStore ? 'block' : 'none', position:'relative', left:'40px', }} > Showing cc</div>

      </div>






      </div>

    );
  }

}

export default LandingPage;
