# nNetJs
Neural Networks for javaScript; a self-contained library

Written for general use in web-based aplications.

###Example 1

We construct a 5-layers deep neural network, outputing a 3-vector, and train it for a 1000 steps.

```javascript
    //generate Random data for demo
    //input -> 5 x 50 matrix, 1x50 is one case
    var input=getRandomBatchIn(5,50)
    //output -> 5x3 matrix, 1x3 is one case
    var output=getRandomBatchOut(5,3,{mean:10,variance:5});
    var data={in:input,out:output};
    //declare the net
    var n=new net();
    //specify its' structure
    n.add({shape:[1,50],type:"input"});
    n.add({shape:[50,10],type:"fc",activation:"sigmoid"});
    n.add({shape:[10,10],type:"fc",activation:"reLu"});
    n.add({shape:[10,10],type:"fc",activation:"sigmoid"});
    n.add({shape:[10,10],type:"fc",activation:"sigmoid"});
    n.add({shape:[10,3],type:"fc",activation:"id"});
    //initialize neurons (uses variance normalization by layer length)
    n.init(gaussian(0, 1));
    //train for 1000 steps
    for(var k=0;k<1000;k++) n.trainStep({data:data,cost:"meanSquare",stepSize:0.1});
    //evaluate on all 5 cases
    var prediction=[];
    for(k in input){
        prediction[k]=n.eval(input[k]);
        forAll(prediction[k],function(x){return x.toFixed(2)});
    }
    //display results
    console.log("Correct predictions");
    console.log(output);
    console.log("\nPredictions:");
    console.log(prediction);
```

###Example 2

We construct a 5-layers deep neural network, outputing classification probability, and train it for a 1000 steps.

```javascript
    //generate Random data for demo
    //input -> 5 x 50 matrix, 1x50 is one case
    var input=getRandomBatchIn(5,50)
    //output -> 5x3 matrix, 1x3 is one case
    var output=getRandomClassBatch(5,3);
    var data={in:input,out:output};
    //declare the net
    var n=new net();
    //specify its' structure
    n.add({shape:[1,50],type:"input"});
    n.add({shape:[50,10],type:"fc",activation:"sigmoid"});
    n.add({shape:[10,10],type:"fc",activation:"reLu"});
    n.add({shape:[10,10],type:"fc",activation:"sigmoid"});
    n.add({shape:[10,10],type:"fc",activation:"reLu"});
    n.add({shape:[10,3],type:"fc",activation:"sigmoid"});
    //initialize neurons (uses variance normalization by layer length)
    n.init(gaussian(0, 1));
    //train for 1000 steps
    for(var k=0;k<1000;k++) n.trainStep({data:data,cost:"entropyBinary",stepSize:0.1});
    //evaluate on all data
    var classification=[];
    for(k in input){
        classification[k]=n.eval(input[k]);
        forAll(classification[k],function(x){return x.toFixed(2)});
    }
    //display results
    console.log("Correct predictions");
    console.log(output);
    console.log("\nPredictions:");
    console.log(classification);
```

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">nNetJs</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://si.linkedin.com/in/zigasajovic" property="cc:attributionName" rel="cc:attributionURL">Žiga Sajovic</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
