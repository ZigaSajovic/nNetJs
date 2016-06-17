/**
 * Žiga Sajovic
 */


var activations={
    id:{
        f:function(x){
            return x;
        },
        df:function(x){
            return 1;
        }
    },
    reLu:{
        f:function(x){
            if(x>0)return x;
            else return 0;
        },
        df:function(x){
            if(x>0)return 1;
            else {
                return 0;
            }
        }
    },
    reLuLeak:{
        f:function(x){
            if(x>0)return x;
            else return 0.0001;
        },
        df:function(x){
            if(x>0)return 1;
            else {
                return 0.0001;
            }
        }
    },
    sigmoid:{
        f:function(x){
            return 1/(1+Math.exp(-x));
        },
        df:function(x){
            return (1/(1+Math.exp(-x)))*(1-1/(1+Math.exp(-x)));
        }
    }
};

var costs={
    meanSquare:{
        f:function(x,y){
            var sum=0;
            for(i in x){
                sum+=(x[i]-y[i])*(x[i]-y[i])/x.length;
            }
            return sum;
        },
        df:function(x,y){
            var out=[];
            for(i in x)out[i]=2*(x[i]-y[i])/x.length;
            return out;
        }
    },
    entropyBinary:{
        f:function(x,y){
            var sum=0;
            for(i in x)sum-=(y[i]*Math.log(x[i])+(1-y[i])*Math.log(1-x[i]))/x.length;
            return sum;
        },
        df:function(x,y){
            var out=[];
            for(i in x)out[i]=-(y[i]/x[i] - (1-y[i])/(1-x[i]) )/x.length;
            return out;
        }
    },
    entropy:{
        f:function(x,y){
            var sum=0;
            for(i in x)sum-=(y[i]*Math.log(x[i]))/x.length;
            return sum;
        },
        df:function(x,y){
            var out=[];
            for(i in x)out[i]=-(y[i]/x[i])/x.length;
            return out;
        }
    }
};

var layers={
    fc:{
        eval:function(layer,result){
            result.result = matMulLayer(result.result, layer);
            forAll(result.result, activations[layer.activation].f);
        },
        evalForGrad:function(layer,lastOut){
            var matMul=matMulLayer(lastOut,layer);
            var output;
            forAllCopy(output=[],matMul,activations[layer.activation].f);
            return{
                matMul:matMul,
                output:output
            }
        },
        grad:function(layer,result,resultLast,grad){
            forAll(result.matMul,activations[layer.activation].df);
            ocProd(grad.grad,result.matMul);
            var db=[];
            forAllCopy(db,grad.grad,function(x){return x;});
            var dw=dLdW(resultLast.output,grad.grad,layer.par.w);

            var grads={
                dw: dw,
                db: db
            };
            grad.grad=matMulTrans(grad.grad,layer);
            return grads;
        }
    },
    softmax:{
        eval:function(layer,result){
            var sum=0;
            for(i in result.result)sum+=Math.exp(result.result[i]);
            for(i in result.result)result.result[i]=Math.exp(result.result[i])/sum;
        },
        evalForGrad:function(layer,lastOut){
            var tmp=[];
            var sum=0;
            for(i in lastOut)sum+=Math.exp(lastOut[i]);
            for(i in lastOut)tmp[i]=Math.exp(lastOut[i])/sum;
            return{
                output:tmp,
                sum:sum
            }
        },
        grad:function(layer,result,resultLast,grad){
            var gradient=[];
            for(i in resultLast.output){
                gradient[i]=0;
                for(j in resultLast.output){
                    if(i==j){
                        gradient[i]+=(result.output[i]-result.output[i]*result.output[i])*grad.grad[j];
                    }
                    else {
                        gradient[i] -= result.output[i] * result.output[j]*grad.grad[j];
                    }
                }
            }
            grad.grad=gradient;
        }
    }
};

function net(){
    this.count=0;
    this.layers=[];
    this.init=init;
    this.add=add;
    this.grad=gradients;
    this.gradBatch=gradientsBatch;
    this.trainStep=trainStepBatch;
    this.eval=eval;
}

function add(layer){
    if(this.count==0&&layer.type=="input"&&layer.shape[0]==1||this.count>0&&this.layers[this.count-1].shape[1]==layer.shape[0]){
        this.count++;
        this.layers.push(layer);
    }
    else{}//error
}

function init(distribution){
    for(var i=1;i< this.layers.length;i++){
        this.layers[i].par=initializeWeights(this.layers[i].shape,distribution);
    }
}

function initializeWeights(shape,distribution){
    var out=[];
    var out2=[];
    for(var i=0;i<shape[1];i++){
        var tmp=[];
        for(var j=0;j<shape[0];j++){
            tmp[j]=distribution()/Math.sqrt(shape[1]);
        }
        out2[i]=0;
        out[i]=tmp;
    }
    return {w:out,b:out2};
}


function eval(input){
    var result={result:input};
    for(var i=1;i<this.layers.length;i++){
        layers[this.layers[i].type].eval(this.layers[i],result);
    }
    return result.result;
}

function gradients(cost, data){
    var results=[];
    var k=1;
    results[0]={output:data.in};
    for(var i=1;i<this.layers.length;i++){
        results[i]=layers[this.layers[i].type].evalForGrad(this.layers[i],results[i-1].output);
        k++;
    }
    var grad={grad:costs[cost].df(results[k-1].output,data.out)};
    var grads=[];
    for(var i=this.layers.length-1;i>0;i--){
        grads[i]=layers[this.layers[i].type].grad(this.layers[i],results[i],results[i-1],grad);

        if(grads[i]==null)grads[i]={none:true};
        grads[i].layer=i;
    }
    return grads;
}

function gradientsBatch(details){
    var batch=[];
    var ks=[];
    for(var a=0;a<details.data.in.length;a++){
        var results=[];
        var k=1;
        results[0]={output:details.data.in[a]};
        for(var i=1;i<this.layers.length;i++){
            results[i]=layers[this.layers[i].type].evalForGrad(this.layers[i],results[i-1].output);
            k++;
        }
        batch[a]=results;
        ks[a]=k;
    }

    var grad=[];
    var grads=[];
    for(i in batch)grad[i]={grad:costs[details.cost].df(batch[i][ks[i]-1].output,details.data.out[i])};

    for(var i=this.layers.length-1;i>0;i--){
        grads[i]=[];
        for(a in batch){
            grads[i][a]=layers[this.layers[i].type].grad(this.layers[i],batch[a][i],batch[a][i-1],grad[a]);
            grads[i].layer=i;
        }
    }
    return grads;
}

function trainStepBatch(details){
    var batch=[];
    var ks=[];
    for(var a=0;a<details.data.in.length;a++){
        var results=[];
        var k=1;
        results[0]={output:details.data.in[a]};
        for(var i=1;i<this.layers.length;i++){
            results[i]=layers[this.layers[i].type].evalForGrad(this.layers[i],results[i-1].output);
            k++;
        }
        batch[a]=results;
        ks[a]=k;
    }

    var grad=[];
    for(i in batch)grad[i]={grad:costs[details.cost].df(batch[i][ks[i]-1].output,details.data.out[i])};

    for(var i=this.layers.length-1;i>0;i--){
        var grads=[];
        var test=true;
        for(a in batch){
            grads[a]=layers[this.layers[i].type].grad(this.layers[i],batch[a][i],batch[a][i-1],grad[a]);
            if(grads[a]==null)test=false;
            else grads[a].layer=i;
        }
        if(test)stepBatch(this.layers[i].par,grads,details.stepSize);
    }
}

function stepBatch(params,grads, stepSize){
    for(i in params.w){
        for(j in params.w[i]){
            for(a in grads){
                params.w[i][j]-=stepSize*grads[a].dw[i][j];
            }
        }
    }
    for(i in params.b){
        for(a in grads){
            params[a]-=stepSize*grads[a].db[i];
        }
    }
}

function dLdW(res,grad,w){
    var tmp=[];
    for(i in w){
        var t=[];
        for(j in w[i]){
            t[j]=res[j]*grad[i];
        }
        tmp[i]=t;
    }
    return tmp;
}

function ocProd(vec1,vec2){
    for(i in vec1){
        vec1[i]=vec1[i]*vec2[i];
    }
}

function forAll(vec,fun){
    for(var i in vec)vec[i]=fun(vec[i]);
}

function forAllCopy(res,vec,fun){
    for(var i in vec)res[i]=fun(vec[i]);
}

function matMulTrans(layer1,layer2){
    var out=[];
    for(var i=0;i<layer2.shape[0];i++){
        out[i]=0;
        for(var j=0;j<layer2.shape[1];j++){
            out[i]+=layer1[j]*layer2.par.w[j][i];

        }
    }
    return out;
}

function matMulLayer(layer1, layer2){
    var out=[];
    for(var i=0;i<layer2.shape[1];i++){
        out[i]=0;
        for(var j=0;j<layer2.shape[0];j++){
            out[i]+=layer1[j]*layer2.par.w[i][j];
        }
        out[i]+=layer2.par.b[i];
    }
    return out;
}

function checkBatchGradient(cost){
    var t=[0.310,0.02,0.03];
    var d=[0.2,0.1,0.7];
    var data={in:t,out:d};
    var n=new net();
    n.add({shape:[1,3],type:"input"});
    n.add({shape:[3,3],type:"fc",activation:"sigmoid"});
    n.add({shape:[3,3],type:"fc",activation:"sigmoid"});
    n.add({shape:[3,3],type:"fc",activation:"sigmoid"});
    n.init(gaussian(0, 1));
    n.layers[1].par.b[1]=5;
    var g=n.grad(cost,data);
    console.log("grad");
    console.log(g[1]);
    var s=n.eval(t);

    var H=costs[cost].f(s,d);
    console.log("\n\n"+H);
    n.layers[1].par.b[1]=5.0000001;
    var r=n.eval(t);

    var T=costs[cost].f(r,d);
    console.log("\n\n"+T);
    var e=(T-H)/0.0000001;
    console.log("numeric");
    console.log(e);
}

function gaussian(mean, stdev) {
    var y2;
    var use_last=false;
    return function(){
        var y1;
        if(use_last){
            y1=y2;
            use_last=false;
        }
        else{
            var x1,x2,w;
            do{
                x1=2.0*Math.random()-1.0;
                x2=2.0*Math.random()-1.0;
                w =x1 * x1 + x2 * x2;
            }while( w>=1.0);
            w=Math.sqrt((-2.0 * Math.log(w))/w);
            y1=x1*w;
            y2=x2*w;
            use_last=true;
        }

        return mean+stdev*y1;
    }
}

function getRandomTest(k){
    var st=gaussian(0,1);
    var out=[];
    for(var i=0;i<k;i++)out[i]=st();
    return out;
}
function getRandomBatchIn(n,k){
    var out=[];
    for(var i=0;i<n;i++)out[i]=getRandomTest(k);
    return out;
}

function getRandomDat(k,data){
    var st=gaussian(data.mean,data.variance);
    var out=[];
    for(var i=0;i<k;i++)out[i]=st();
    return out;
}

function getRandomBatchOut(n,k,data){
    var out=[];
    for(var i=0;i<n;i++)out[i]=getRandomDat(k,data);
    return out;
}
function getRandomClassBatch(n,k){
    var st=gaussian(0,1);
    var out=[];
    for(var i=0;i<n;i++){
        var tmp=[];
        var x=parseInt(Math.abs(st())*(k+1))%k;
        for(var j=0;j<k;j++)tmp[j]=0;
        tmp[x]=1;
        out[i]=tmp;
    }
    return out;
}

function normalize(mat){
    var avgs=[];
    var a=mat.length;
    var b=mat[0].length;
    for(var i=0;i<b;i++){
        avgs[i]=0;
        for(var j=0;j<a;j++){
            avgs[i]+=mat[j][i];
        }
        avgs[i]/=a;
    }
    var vari=[];
    for(var i=0;i<b;i++){
        vari[i]=0;
        for(var j=0;j<a;j++){
            vari[i]+=(mat[j][i]-avgs[i])*(mat[j][i]-avgs[i]);
        }
        avgs[i]=Math.sqrt(avgs[i]/a);
    }
    for(var i=0;i<b;i++){
        for(var j=0;j<a;j++){
            mat[j][i]=(mat[j][i]-avgs[i])/vari[i];
        }
    }
}

function example1(){
    /*
     Example shows training of a neural net outputting a numeric value
     Final layer is linear, as indicated by "id" activation function
     */
    //generate Random data for demo
    //input -> 5 x 50 matrix, 1x50 is one case
    var input=getRandomBatchIn(5,50)
    //output -> 5x2 matrix, 1x2 is one case
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
    //define cost function
    var cost="meanSquare";
    //train for 1000 steps
    for(var k=0;k<1000;k++) n.trainStep({data:data,cost:cost,stepSize:0.1});
    //evaluate on all 5 cases
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
}

function example2(){
    //generate Random data for demo
    //input -> 5 x 50 matrix, 1x50 is one case
    var input=getRandomBatchIn(5,50)
    //output -> 5x2 matrix, 1x2 is one case
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
    //define cost function
    var cost="entropyBinary";
    //train for 1000 steps
    for(var k=0;k<1000;k++) n.trainStep({data:data,cost:cost,stepSize:0.1});
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
}

//example1();
example2();
