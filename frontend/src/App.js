import React from "react";
// import RecordingPage from '/home/sathvik/Documents/AAI_webpage/frontend/src/recordingPage'
// import LandingPage from '/home/sathvik/Documents/AAI_webpage/frontend/src/landingPage'
import LandingPage from 'landingPage'
import RecordingPage from 'recordingPage'
import PlotPage from 'plotPage'
import { Component } from "react";
import {BrowserRouter, Route, Switch } from "react-router-dom";

class App extends Component {
  render() {
    return (
    <BrowserRouter>
        <div>
          <Switch>
          <Route exact path="/record" component={RecordingPage} />
            <Route exact path="/plot" component={PlotPage} />
          <Route component={LandingPage} />
          </Switch>
        </div>
      </BrowserRouter>
    );
  }
}

export default App;
