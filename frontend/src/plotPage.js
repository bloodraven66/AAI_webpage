import React, { Component } from "react";
import './styles.css'
class PlotPage extends Component {
  async componentDidMount() {
    const XHR = new XMLHttpRequest();
    XHR.open('POST', "http://10.64.26.89:3001/plot/");
    XHR.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    XHR.send(JSON.stringify({ "file": "ping"}))
    XHR.onload  = function() {
   		var jsonResponse = JSON.parse(XHR.responseText).image;
      document.getElementById('imageBox').src = "data:image/png;base64," + jsonResponse;
  };

}

  render() {
    return(

      <div class="center">
  <img id="imageBox"/>
  <a style={{position:'relative', top:'-90px', left:'300px'}} href='/'><button>Go back</button> </a>
</div>

  )
  };

}
export default PlotPage;
