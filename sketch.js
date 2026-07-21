//LINEAR REGRESSION with p5 & tensorflow
// By Fiorie Rousselot-Barbe, 2026

//Our X and Y value points are set as plain arrays
let x_vals =[];
let y_vals = [];

//y=mx+b for slope intercept formula
//using b or m.print(); in the console to check the values
let m;
let b;

const learningRate = 0.35;
// sgd = stochastic gradient descent
// the optimizer helps the model improve itself
const optimizer = tf.train.sgd(learningRate);

function setup(){

// canvas is 400 x 400 px
createCanvas(400, 400);

/*
 NOTES: i'm using p5.js random to get a number between 0 and 1
 tensor = main data structure in tensorflow.js
 represent multi-dimensional arrays of numbers ; holds data for ML models
 it can run on your GPU to do quicker mathematics
 tf scalar = a simple number ; tf variable = can change/learn
*/
//these are my trainable variables
m = tf.variable(tf.scalar(random(1)));
b = tf.variable(tf.scalar(random(1)));

}

function loss(pred, labels){

    // we'll train the model to minimize the LOSS function with the OPTIMIZER ; adjusting m and b
    return pred.sub(labels).square().mean();
    
}

/*
using this function to predict Y based off where our X values are (y= mx + b)

*/
function predict(x){

    // 1 dimensional tensor
    const xs = tf.tensor1d(x);
    //y=mx+b
    const ys = xs.mul(m).add(b);
    return ys;

}

//When the mouse is pressed...
function mousePressed(){

    //map(value, oldMin, oldMax, newMin, newMax)
    // we are converting messy coordinates into normalized mathematic data
    // example: if X=200, it becomes 0.5
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    //push to add values of the mouse position to our arrays
    x_vals.push(x);
    y_vals.push(y);
}

function draw(){

    //tidy to clean up memory data
    tf.tidy(() => {
    if (x_vals.length >0){
    const ys = tf.tensor1d(y_vals);
    optimizer.minimize(() => loss (predict(x_vals), ys));
    }
})

//black background
background (0);

//setting the color and weight of the points we want to draw
stroke (255);
strokeWeight (7);

//earlier we did screen -> math so the AI can learn better
//now we're un-normalizing those values to draw them properly

for (let i = 0; i < x_vals.length; i++){
    // px py = point of x / y
    //un-normalizing our x and y values
    let px = map (x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    //drawing a point at our x and y values
    point (px, py);

}

//line
//also tidying up the tensor count
tf.tidy(() => {
const xs = [0,1];
const ys = predict(xs);
let lineY = ys.dataSync();


let x1 = map(xs[0], 0, 1, 0, width);
let x2 = map(xs[1], 0, 1, 0, width);


let y1 = map(lineY[0], 0, 1, height, 0);
let y2 = map(lineY[1], 0, 1, height, 0);

//drawing the line
strokeWeight (2);
line (x1, y1, x2, y2);
});
}