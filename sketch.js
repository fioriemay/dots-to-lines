// LINEAR REGRESSION teaching tool — p5.js & TensorFlow.js
// By Fiorie Rousselot-Barbe, 2026

// Our X and Y value points, normalized 0..1, plus the frame each was added
// (used to animate them popping in).
let x_vals = [];
let y_vals = [];
let pointAddedFrame = [];

// y = mx + b
let m, b;

// Smoothly-animated stand-ins for m/b so the line "chases" its real target
// instead of snapping — this is what makes each step visible.
let displayedM, displayedB;
let targetM, targetB;

let learningRate = 0.35;
let optimizer;

// 'auto' trains continuously every frame; 'step' only trains when the
// Step button is clicked.
let mode = 'auto';
let isPlaying = true;
let speed = 1; // iterations per frame in auto mode

let iteration = 0;
let currentLoss = null;
let lossHistory = [];

let lossChartCanvas, lossChartCtx;

// Cached DOM elements
let modeAutoBtn, modeStepBtn, autoControls, stepControls;
let playPauseBtn, stepBtn, speedSlider, lrSlider, lrValueEl;
let randomDataBtn, clearBtn, resetWeightsBtn;
let statM, statB, statLoss, statIter, explanationEl;

function setup() {
  const size = canvasSize();
  const cnv = createCanvas(size, size);
  cnv.parent('canvas-container');

  // Trainable variables — same random start as the original sketch.
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  displayedM = m.dataSync()[0];
  displayedB = b.dataSync()[0];
  targetM = displayedM;
  targetB = displayedB;

  optimizer = tf.train.sgd(learningRate);

  lossChartCanvas = document.getElementById('loss-chart');
  lossChartCtx = lossChartCanvas.getContext('2d');
  sizeLossChart();

  cacheDom();
  wireEvents();
  updateStatsBar();
}

function cacheDom() {
  modeAutoBtn = document.getElementById('mode-auto');
  modeStepBtn = document.getElementById('mode-step');
  autoControls = document.getElementById('auto-controls');
  stepControls = document.getElementById('step-controls');

  playPauseBtn = document.getElementById('play-pause-btn');
  stepBtn = document.getElementById('step-btn');
  speedSlider = document.getElementById('speed-slider');
  lrSlider = document.getElementById('lr-slider');
  lrValueEl = document.getElementById('lr-value');

  randomDataBtn = document.getElementById('random-data-btn');
  clearBtn = document.getElementById('clear-btn');
  resetWeightsBtn = document.getElementById('reset-weights-btn');

  statM = document.getElementById('stat-m');
  statB = document.getElementById('stat-b');
  statLoss = document.getElementById('stat-loss');
  statIter = document.getElementById('stat-iter');
  explanationEl = document.getElementById('explanation-text');
}

function wireEvents() {
  modeAutoBtn.addEventListener('click', () => setMode('auto'));
  modeStepBtn.addEventListener('click', () => setMode('step'));
  playPauseBtn.addEventListener('click', togglePlay);
  stepBtn.addEventListener('click', () => trainStep());

  speedSlider.addEventListener('input', (e) => {
    speed = parseInt(e.target.value, 10);
  });

  lrSlider.addEventListener('input', (e) => {
    learningRate = parseFloat(e.target.value);
    lrValueEl.textContent = learningRate.toFixed(2);
    optimizer = tf.train.sgd(learningRate);
  });

  randomDataBtn.addEventListener('click', generateRandomData);
  clearBtn.addEventListener('click', clearPoints);
  resetWeightsBtn.addEventListener('click', resetWeights);
}

function setMode(newMode) {
  mode = newMode;
  modeAutoBtn.classList.toggle('active', mode === 'auto');
  modeStepBtn.classList.toggle('active', mode === 'step');
  autoControls.classList.toggle('hidden', mode !== 'auto');
  stepControls.classList.toggle('hidden', mode !== 'step');
}

function togglePlay() {
  isPlaying = !isPlaying;
  playPauseBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
}

function loss(pred, labels) {
  // Mean squared error — average squared distance between line and points.
  return pred.sub(labels).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  return xs.mul(m).add(b);
}

// Runs exactly one gradient descent update and reports what changed to the
// line, in plain language, in the sidebar.
function trainStep() {
  if (x_vals.length === 0) {
    explanationEl.innerHTML = `<p>Add at least one point on the grid first — the line needs something to aim for.</p>`;
    return null;
  }

  const oldM = m.dataSync()[0];
  const oldB = b.dataSync()[0];

  // Average signed error using the *old* weights — used for the plain
  // language explanation of which way the line is being pulled.
  let meanResidual = 0;
  for (let i = 0; i < x_vals.length; i++) {
    const predY = oldM * x_vals[i] + oldB;
    meanResidual += (y_vals[i] - predY);
  }
  meanResidual /= x_vals.length;

  const lossValue = tf.tidy(() => {
    const ys = tf.tensor1d(y_vals);
    const costTensor = optimizer.minimize(() => loss(predict(x_vals), ys), true);
    return costTensor.dataSync()[0];
  });

  const newM = m.dataSync()[0];
  const newB = b.dataSync()[0];

  iteration++;
  currentLoss = lossValue;
  lossHistory.push(lossValue);
  if (lossHistory.length > 300) lossHistory.shift();

  targetM = newM;
  targetB = newB;

  updateExplanation(oldM, oldB, newM, newB, meanResidual, lossValue);

  return { oldM, oldB, newM, newB, lossValue };
}

function tiltPhrase(oldM, newM, epsilon = 0.0005) {
  const diff = newM - oldM;
  if (Math.abs(diff) < epsilon) return 'stayed about as steep as it was';
  return diff > 0 ? 'got a little steeper' : 'got a little flatter';
}

function shiftPhrase(oldB, newB, epsilon = 0.0005) {
  const diff = newB - oldB;
  if (Math.abs(diff) < epsilon) return 'barely moved';
  return diff > 0 ? 'shifted up a touch' : 'shifted down a touch';
}

function updateExplanation(oldM, oldB, newM, newB, meanResidual, lossValue) {
  const tilt = tiltPhrase(oldM, newM);
  const shift = shiftPhrase(oldB, newB);

  let residualSentence;
  if (Math.abs(meanResidual) < 0.002) {
    residualSentence = 'On average, the line already runs right through your points.';
  } else if (meanResidual > 0) {
    residualSentence = 'On average, your points sit above the line, so it shifts upward to reach them.';
  } else {
    residualSentence = 'On average, your points sit below the line, so it shifts downward to reach them.';
  }

  explanationEl.innerHTML = `
    <p><strong>Step ${iteration}</strong> — the line is off by ${lossValue.toFixed(4)} on average (smaller means a closer fit).</p>
    <ul class="step-list">
      <li>The line's <strong>steepness (m)</strong> ${tilt} — ${oldM.toFixed(3)} → ${newM.toFixed(3)}</li>
      <li>The line's <strong>starting point (b)</strong> ${shift} — ${oldB.toFixed(3)} → ${newB.toFixed(3)}</li>
    </ul>
    <p>${residualSentence}</p>
    <p class="muted">Every step, the program checks how far the line is from your points, then nudges its steepness and position a little closer.</p>
  `;
}

function generateRandomData() {
  x_vals = [];
  y_vals = [];
  pointAddedFrame = [];

  const trueM = random(0.3, 1.4) * (random() < 0.5 ? -1 : 1);
  const trueB = random(0.1, 0.6);
  const n = 14;

  for (let i = 0; i < n; i++) {
    const x = random(1);
    const noise = randomGaussian(0, 0.06);
    const y = constrain(trueM * x + trueB + noise, 0, 1);
    x_vals.push(x);
    y_vals.push(y);
    pointAddedFrame.push(frameCount);
  }

  explanationEl.innerHTML = `<p>Dropped ${n} points roughly along a hidden line. Press play (or step through) and watch the line find it.</p>`;
}

function clearPoints() {
  x_vals = [];
  y_vals = [];
  pointAddedFrame = [];
  iteration = 0;
  currentLoss = null;
  lossHistory = [];
  explanationEl.innerHTML = `<p>Points cleared. Click the grid to add new ones.</p>`;
}

function resetWeights() {
  m.dispose();
  b.dispose();
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  displayedM = m.dataSync()[0];
  displayedB = b.dataSync()[0];
  targetM = displayedM;
  targetB = displayedB;
  iteration = 0;
  currentLoss = null;
  lossHistory = [];
  explanationEl.innerHTML = `<p>The line jumped to a new random starting spot and steepness. Watch it find your points again.</p>`;
}

function mousePressed() {
  if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) return;

  const x = map(mouseX, 0, width, 0, 1);
  const y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
  pointAddedFrame.push(frameCount);
}

function draw() {
  if (mode === 'auto' && isPlaying) {
    for (let i = 0; i < speed; i++) trainStep();
  }

  // Chase the real m/b so every change — auto or single-stepped — animates.
  displayedM = lerp(displayedM, targetM, 0.15);
  displayedB = lerp(displayedB, targetB, 0.15);

  drawBackground();
  drawGrid();
  drawPoints();
  drawRegressionLine();
  drawHoverPreview();
  updateStatsBar();
  drawLossChart();
}

function drawBackground() {
  const ctx = drawingContext;
  const hueShift = (frameCount * 0.15) % 360;
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, `hsl(${230 + hueShift * 0.2}, 45%, 10%)`);
  grad.addColorStop(1, `hsl(${265 + hueShift * 0.2}, 55%, 16%)`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);
}

function drawGrid() {
  push();
  stroke(255, 255, 255, 10);
  strokeWeight(1);
  const spacing = 40;
  for (let x = spacing; x < width; x += spacing) line(x, 0, x, height);
  for (let y = spacing; y < height; y += spacing) line(0, y, width, y);
  pop();
}

function easeOutBack(t) {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

function drawPoints() {
  noStroke();
  for (let i = 0; i < x_vals.length; i++) {
    const age = frameCount - pointAddedFrame[i];
    const t = constrain(age / 18, 0, 1);
    const scale = easeOutBack(t);
    const px = map(x_vals[i], 0, 1, 0, width);
    const py = map(y_vals[i], 0, 1, height, 0);
    const r = 8 * scale;

    fill(79, 209, 255, 40);
    circle(px, py, r * 3.2);
    fill(79, 209, 255, 90);
    circle(px, py, r * 2);
    fill(230, 245, 255);
    circle(px, py, r * 1.15);
  }
}

function drawRegressionLine() {
  const y1 = map(displayedB, 0, 1, height, 0);
  const y2 = map(displayedM + displayedB, 0, 1, height, 0);

  // Layered glow, then a crisp core line on top.
  strokeWeight(8);
  stroke(255, 110, 199, 35);
  line(0, y1, width, y2);

  strokeWeight(4);
  stroke(255, 110, 199, 90);
  line(0, y1, width, y2);

  strokeWeight(2);
  stroke(255, 255, 255, 230);
  line(0, y1, width, y2);
}

function drawHoverPreview() {
  if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) return;
  const pulse = 6 + 3 * Math.sin(frameCount * 0.15);
  noFill();
  stroke(255, 255, 255, 90);
  strokeWeight(1.5);
  circle(mouseX, mouseY, 18 + pulse);
}

function updateStatsBar() {
  statM.textContent = displayedM.toFixed(3);
  statB.textContent = displayedB.toFixed(3);
  statLoss.textContent = currentLoss === null ? '–' : currentLoss.toFixed(4);
  statIter.textContent = iteration;
}

function drawLossChart() {
  const ctx = lossChartCtx;
  const w = lossChartCanvas.width;
  const h = lossChartCanvas.height;
  ctx.clearRect(0, 0, w, h);
  if (lossHistory.length < 2) return;

  const maxLoss = Math.max(...lossHistory);
  const n = lossHistory.length;

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * w;
    const norm = maxLoss > 0 ? lossHistory[i] / maxLoss : 0;
    const y = h - norm * (h - 10) - 5;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  const strokeGrad = ctx.createLinearGradient(0, 0, w, 0);
  strokeGrad.addColorStop(0, '#4fd1ff');
  strokeGrad.addColorStop(1, '#ff6ec7');
  ctx.strokeStyle = strokeGrad;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = 'rgba(79, 209, 255, 0.08)';
  ctx.fill();
}

function canvasSize() {
  const container = document.getElementById('canvas-container');
  const w = container.clientWidth || 480;
  return Math.max(280, Math.min(560, w));
}

function sizeLossChart() {
  lossChartCanvas.width = lossChartCanvas.clientWidth;
  lossChartCanvas.height = 90;
}

function windowResized() {
  const size = canvasSize();
  resizeCanvas(size, size);
  sizeLossChart();
}
