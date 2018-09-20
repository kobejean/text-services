'use strict';

var spawn = require('child_process').spawn;

exports.get_text_similarity = function(req, res) {
  py   = spawn('python3', ['../machine_learning/similarity.py']);
  data = [strings1, strings2];
  dataString = '';

  /*Here we are saying that every time our node application receives data from the python process output stream(on 'data'), we want to convert that received data into a string and append it to the overall dataString.*/
  py.stdout.on('data', function(data){
    dataString += data.toString();
  });

  /*Once the stream is done (on 'end') we want to simply log the received data to the console.*/
  py.stdout.on('end', function(){
    console.log('Sum of numbers=', dataString);
  });

  /*We have to stringify the data first otherwise our python process wont recognize it*/
  py.stdin.write(JSON.stringify(data));
  py.stdin.end();

  res.json({
    similarity: [0.5, 0.6]
  });
};
