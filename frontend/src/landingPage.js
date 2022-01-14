import React, { Component } from "react";
import { BrowserRouter, Route } from "react-router-dom";
// import RecordingPage from '/home/sathvik/Documents/AAI_webpage/frontend/src/recordingPage'
import RecordingPage from 'recordingPage'
import PlotPage from 'plotPage'
var jsonResponseMean = 0;
var jsonResponseSTD = 0;

const rand=()=>Math.random(0).toString(36).substr(2);
const token=(length)=>(rand()+rand()+rand()+rand()).substr(0,length);
const user_id=token(20);
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




  onChangeIcon = event => {
    document.getElementById("start1").disabled = true;
    document.getElementById("stop1").disabled = false;
    // document.getElementById("save1").disabled = false;
  };
  onChangeIcon2 = event => {
    document.getElementById("start1").disabled = false;
    document.getElementById("stop1").disabled = true;
    document.getElementById("save1").disabled = false;
  };
  onChangeIcon3 = event => {
  };

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

    const recordButton = document.getElementById("start1");
    const stopButton = document.getElementById("stop1");
    const saveButton = document.getElementById("save1");

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
      saveButton.setAttribute("disabled", true);
    });

    stopButton.addEventListener("click", async () => {
      recordButton.removeAttribute("disabled");
      stopButton.setAttribute("disabled", true);
      saveButton.removeAttribute("disabled");
      audio = await recorder.stop();
      this.setState({ recording: false });
      clearInterval(interval);
    });

    saveButton.addEventListener("click", () => {
    const XHR_upload = new XMLHttpRequest();
    const FD = new FormData()
    console.log(audio.audioBlob)
    FD.append('file', audio.audioBlob);
    FD.append('key', user_id);
    console.log(user_id)
    XHR_upload.open('POST', "http://0.0.0.0:3001/uploadbytes/");
    var e = document.getElementById("ddlViewBy");
    var strUser = e.options[e.selectedIndex].text;
    const XHR2 = new XMLHttpRequest();
    XHR2.open('POST', "http://0.0.0.0:3001/sendModelName/");
    XHR2.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    XHR2.send(JSON.stringify({ "file": strUser, "key": user_id}))
    XHR_upload.send(FD);
    XHR_upload.onload  = function() {
      console.log('here')
   		var jsonResponse = JSON.parse(XHR_upload.responseText);
      console.log(jsonResponse)
		if (jsonResponse === 'done'){

        const XHR_upload = new XMLHttpRequest();
        XHR_upload.open('POST', "http://0.0.0.0:3001/plot/");
        XHR_upload.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        XHR_upload.send(JSON.stringify({"file":"ping","key": user_id}))
        XHR_upload.onload  = function() {
       		var jsonResponse = JSON.parse(XHR_upload.responseText).image;
          document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;

          const XHRcc = new XMLHttpRequest();
          XHRcc.open('POST', "http://0.0.0.0:3001/getcc/");
          XHRcc.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
          XHRcc.send(JSON.stringify({"key": user_id, "file": "returnCC"}))
          XHRcc.onload  = function() {
            console.log(XHRcc.responseText)

            jsonResponseMean = JSON.parse(XHRcc.responseText).mean
            jsonResponseSTD = JSON.parse(XHRcc.responseText).std

            if (jsonResponseMean !== 'none') {
              console.log(jsonResponseMean, jsonResponseSTD)
                document.getElementById("ccDiv").innerHTML = 'Correlation Coefficient: '+ jsonResponseMean+'('+jsonResponseSTD + ')';

                  document.getElementById('ccDiv').style.display = 'block';
            }
            else {
            }
        };
      };







		}else{
    }
  };
    });


    }





  onFileChange = event => {
    this.setState({ selectedFile: event.target.files[0] });
  };
  onFileUpload = () => {
    console.log('button pressed')
    const formData = new FormData();
    formData.append(
      "myFile",
      this.state.selectedFile
    );
    formData.append('key', user_id);
    var e = document.getElementById("ddlViewBy");
    var strUser = e.options[e.selectedIndex].text;

      const XHR2 = new XMLHttpRequest();
      XHR2.open('POST', "http://0.0.0.0:3001/sendModelName/");
      XHR2.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
      XHR2.send(JSON.stringify({ "file": strUser, "key": user_id}))


    const XHR = new XMLHttpRequest();
    XHR.open('POST', "http://0.0.0.0:3001/upload/");

    XHR.send(formData);
    XHR.onload  = function() {
      console.log('here')
   		var jsonResponse = JSON.parse(XHR.responseText);
      console.log(jsonResponse, )
		if (jsonResponse == 'done'){
      console.log('received')
        const XHR = new XMLHttpRequest();
        XHR.open('POST', "http://0.0.0.0:3001/plot/");
        XHR.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        XHR.send(JSON.stringify({ "file":"ping", "key": user_id}))
        XHR.onload  = function() {
       		var jsonResponse = JSON.parse(XHR.responseText).image;
          document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;

          const XHRcc = new XMLHttpRequest();
          XHRcc.open('POST', "http://0.0.0.0:3001/getcc/");
          XHRcc.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
          XHRcc.send(JSON.stringify({"key": user_id, "file": "returnCC"}))
          XHRcc.onload  = function() {
            console.log(XHRcc.responseText)

            jsonResponseMean = JSON.parse(XHRcc.responseText).mean
            jsonResponseSTD = JSON.parse(XHRcc.responseText).std

            // document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;
            if (jsonResponseMean !== 'none') {
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
        XHR3.open('POST', "http://0.0.0.0:3001/sendModelNamePTA/");
        XHR3.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        XHR3.send(JSON.stringify({ "file": strUser2, "key": user_id}))

      const XHR2 = new XMLHttpRequest();
      XHR2.open('POST', "http://0.0.0.0:3001/sendText/");
      XHR2.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
      XHR2.send(JSON.stringify({ "file": textentry, "key": user_id}))

      XHR2.onload  = function() {

     		var jsonResponse = JSON.parse(XHR2.responseText);
        console.log(jsonResponse)
  		if (jsonResponse == 'done'){

          const XHR = new XMLHttpRequest();
          XHR.open('POST', "http://0.0.0.0:3001/plot_p2e/");
          XHR.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
          XHR.send(JSON.stringify({ "file": "ping", "key": user_id}))
          XHR.onload  = function() {
         		var jsonResponse = JSON.parse(XHR.responseText).image;
            document.getElementById('imageBox2').src = "data:image/png;base64," + jsonResponse;

            const XHRcc = new XMLHttpRequest();
            XHRcc.open('POST', "http://0.0.0.0:3001/getcc/");
            XHRcc.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
            XHRcc.send(JSON.stringify({"key": user_id, "file": "returnCC"}))
            XHRcc.onload  = function() {
              console.log(XHRcc.responseText)
              jsonResponseMean = JSON.parse(XHRcc.responseText).mean
              jsonResponseSTD = JSON.parse(XHRcc.responseText).std
              // document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;
              if (jsonResponseMean !== 'none') {
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
    XHR.open('POST', "http://0.0.0.0:3001/reset_params/");
    XHR.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    XHR.send(JSON.stringify({"key":user_id, "file": "reset"}))


  };





  render() {
    return (
      <div style={{overflowx: 'hidden', position:'relative', left:'10px'}}>

      <h2><center>Articulatory movement estimation in Speech Production</center></h2>
      <BrowserRouter>
      <Route path="/record" component={RecordingPage} />
      <Route path="/plot" component={PlotPage} />
      </BrowserRouter>
      <div style={{position:'relative', top:'160px'}}><hr class='dashed'></hr></div>
      <hr  class="dashed"></hr>
      <div class='leftHalf'>
      <h3 class='subtitle'>Acoustic-to-Articulatory Inversion (AAI)</h3>
      <div style={{position:'relative', left:'100px'}}>
      <b>Upload an audio file:  </b>
      <div style={{position:'relative', top:'-24px', left:'190px'}}>
      <input type="file" onChange={this.onFileChange} accept="audio/wav, audio/mp3" id='uploadingFile' hidden/>
      <label class='btn-design-upload' for="uploadingFile">Select file</label>


      <button onClick={this.onFileUpload} class='btn-design' id='uploadFileButton' >
        Upload  </button>
      </div></div>
      <div style={{position:'relative', top:'-20px', left:'250px'}}>
      <h4>OR</h4>
      </div>
      <div style={{position:'relative', top:'-15px', left:'100px'}}>
      <b>Record your audio:    </b>
      <script type="text/javascript">
           document.body.innerHTML = '';
       </script>
      <div style={{position:'relative', top:'-15px', left:'190px'}}>
          <button  id="start1" onClick={this.onChangeIcon} class='btn-design'>Start  </button>
          <button id="stop1" onClick={this.onChangeIcon2} class='btn-design' >Stop </button>
          <button id="save1" class='btn-design'>Upload</button>
      </div></div>
    <div style={{position:'absolute', top:'40px', left:'625px'}}>
        <label for="ddlViewBy">Select Model</label><br></br>
        <select id="ddlViewBy">
        <option value='1'>GMM</option>
        <option value='2'>DNN</option>
        <option value='3'>CNN</option>
        <option value='4'>LSTM</option>
        <option value='5'>Transformer</option>
        </select>
        </div>
        <div style={{position:'relative', top:'-10px'}}>
          <img class='center' id="imageBox"/>
      </div></div>

      <div class='rightHalf'>

      <h3 class='subtitle'>Phoneme-to-Articulatory Estimation(PTA)</h3>
      <div style={{position:'relative', left:'120px'}}>
      <b>Enter text:    </b>
      <div style={{position:'relative', left:'100px', top:'-18px'}}>
      <input name="searchTxt" type="text"  id="inputText"/>

      <div style={{position:'relative', left:'230px', top:'-27px'}}>
        <button onClick={this.onTextUpload} class='btn-design'>
          Upload
        </button>
        </div> </div></div>
        <a  href='/'><button class='btn-design' id='disable' style={{display: this.state.showStore ? 'block' : 'none' }} onClick={this.disableImage}>Reset</button> </a>



    <div style={{position:'absolute',left:'625px', top:'40px'}}>
    <label for="ddlViewByPTA">Select Model</label><br></br>
    <select id="ddlViewByPTA">
    <option value='1'>FastSpeech  </option>
    <option value='2'>Tacotron2</option>
    </select>
    </div>
      <div style={{position:'relative', top:'15px'}}>
    <img class='center' id="imageBox2"/></div>
    </div>

    <div id='ccDiv' style={{display: this.state.showCCStore ? 'block' : 'none', position:'absolute', top:'160px', left:'1400px', }} > Showing cc</div>


    </div>

    );
  }

}

export default LandingPage;
