import React, { Component } from "react";


class RecordingPage extends Component {
  state = {
    sentenceId: 0,
    showDoneModal: false,
    seconds: 0,
    totalDurationRecorded: 0,
    totalRecordingsRecorded: 0,
    recording: false,
  };

  snapshot = null;
  totalSentences = null;
  userDoc = null;

  async componentDidMount() {
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
      // audio.exportWAV(createDownloadLink)
    });

    playButton.addEventListener("click", () => {
      audio.play();
    });
  

    saveButton.addEventListener("click", () => {
    const XHRrecord = new XMLHttpRequest();
    const FD = new FormData()
    FD.append('file', audio.audioBlob);
    XHRrecord.open('POST', "http://10.64.26.89:3001/predict/");
    XHRrecord.send(FD);
    });
  }




  render() {
    return (

      <div
        style={{

        }}

      >
      <script type="text/javascript">
           document.body.innerHTML = '';
       </script>
        <div className="flex-container">
          <button id="start" className="btn btn-circle btn-xl mr-4 mb-4">
            <span style={{ position: "relative" }}>Start</span>
          </button>
          <button
            id="stop"
            className="btn btn-circle btn-xl mr-4 mb-4"
            disabled={true}
          >
            <span style={{ position: "relative" }}>Stop</span>
          </button>
          <button id="play" className="btn btn-circle btn-xl mr-4 mb-4" disabled={true}>
            <span style={{ position: "relative" }}>Play</span>
          </button>
          <button id="save" className="btn btn-circle btn-xl mr-4 mb-4" disabled={true}>
            <span style={{ position: "relative" }}>Upload</span>
          </button>
        </div>
      </div>
    );
  }

}

export default RecordingPage;
