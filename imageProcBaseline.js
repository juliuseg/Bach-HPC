let gridSize = 20;
let cols, rows;
let active = [];
let radius = 8;
let endpoints = [];

function setup() {
  createCanvas(600, 600);
  cols = floor(width / gridSize);
  rows = floor(height / gridSize);

  for (let i = 0; i < cols; i++) {
    active[i] = [];
    for (let j = 0; j < rows; j++) {
      active[i][j] = false;
    }
  }
}

function draw() {
  background(255);
  stroke(220);

  // Draw grid
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      rect(i * gridSize, j * gridSize, gridSize, gridSize);
    }
  }

  // Draw or erase on mouse drag
  if (mouseIsPressed) {
    let i = floor(mouseX / gridSize);
    let j = floor(mouseY / gridSize);
    if (i >= 0 && i < cols && j >= 0 && j < rows) {
      if (keyIsDown(SHIFT)) {
        active[i][j] = false;
      } else {
        active[i][j] = true;
      }
    }
  }

  // --- CONNECTION DRAWING ---
  stroke(0);
  strokeWeight(radius * 2);
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      if (!active[i][j]) continue;

      let hasNeighbor = false;

      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;

          let ni = i + dx;
          let nj = j + dy;

          if (ni >= 0 && ni < cols && nj >= 0 && nj < rows && active[ni][nj]) {
            if (Math.abs(dx) === 1 && Math.abs(dy) === 1) {
              let side1 = active[i + dx]?.[j] || false;
              let side2 = active[i]?.[j + dy] || false;
              if (side1 || side2) continue;
            }

            let x1 = i * gridSize + gridSize / 2;
            let y1 = j * gridSize + gridSize / 2;
            let x2 = ni * gridSize + gridSize / 2;
            let y2 = nj * gridSize + gridSize / 2;
            line(x1, y1, x2, y2);
            hasNeighbor = true;
          }
        }
      }

      if (!hasNeighbor) {
        let x = i * gridSize + gridSize / 2;
        let y = j * gridSize + gridSize / 2;
        point(x, y);
      }
    }
  }

  // --- ENDPOINT DETECTION ---
  endpoints = [];
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      if (!active[i][j]) continue;

      let neighbors = [];
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          let ni = i + dx;
          let nj = j + dy;
          if (ni >= 0 && ni < cols && nj >= 0 && nj < rows && active[ni][nj]) {
            neighbors.push([ni, nj, dx, dy]);
          }
        }
      }

      let isEndpoint = false;
      let arrowDX = 0, arrowDY = 0;

      if (neighbors.length === 1) {
        isEndpoint = true;
        [, , arrowDX, arrowDY] = neighbors[0];
      } else if (neighbors.length === 2) {
        let [a, b] = neighbors;
        let dx = Math.abs(a[0] - b[0]);
        let dy = Math.abs(a[1] - b[1]);
        let touching = (dx === 1 && dy === 0) || (dx === 0 && dy === 1);
        if (touching) {
          isEndpoint = true;
          if (Math.abs(a[2]) + Math.abs(a[3]) === 1) {
            [arrowDX, arrowDY] = [a[2], a[3]];
          } else {
            [arrowDX, arrowDY] = [b[2], b[3]];
          }
        }
      }

      if (isEndpoint) {
        let x = i * gridSize + gridSize / 2;
        let y = j * gridSize + gridSize / 2;
        endpoints.push({ x, y, dx: arrowDX, dy: arrowDY });
      }
    }
  }

  // --- DRAW ENDPOINTS ---
  noFill();
  stroke(255, 0, 150);
  strokeWeight(2);
  for (let ep of endpoints) {
    ellipse(ep.x, ep.y, radius * 2 + 4);
  }

  // --- DRAW ENDPOINT ARROWS ---
  stroke(0, 0, 255);
  strokeWeight(2);
  fill(0, 0, 255);
  for (let ep of endpoints) {
    let len = 30;
    let x1 = ep.x;
    let y1 = ep.y;
    let x2 = x1 - ep.dx * len;
    let y2 = y1 - ep.dy * len;

    line(x1, y1, x2, y2);
    push();
    translate(x2, y2);
    rotate(atan2(-ep.dy, -ep.dx));
    triangle(-4, -3, -4, 3, 0, 0);
    pop();
  }

  // --- MATCH ENDPOINTS ---
  let searchRadius = 5 * gridSize;
  stroke(0, 200, 0);
  strokeWeight(1.5);

  let used = new Set();

  for (let i = 0; i < endpoints.length; i++) {
    if (used.has(i)) continue;
    let a = endpoints[i];
    let dirA = createVector(a.dx, a.dy);
    if (dirA.mag() === 0) continue;
    dirA.normalize();
    let posA = createVector(a.x, a.y);

    for (let j = i + 1; j < endpoints.length; j++) {
      if (used.has(j)) continue;
      let b = endpoints[j];
      let dirB = createVector(b.dx, b.dy);
      if (dirB.mag() === 0) continue;
      dirB.normalize();
      let posB = createVector(b.x, b.y);

      let between = p5.Vector.sub(posB, posA);
      let d = between.mag();
      if (d > searchRadius) continue;

      let betweenNorm = between.copy().normalize();
      let dotDirs = dirA.dot(dirB);
      let dotAB = dirA.dot(betweenNorm);
      let dotBA = dirB.dot(p5.Vector.sub(posA, posB).normalize());

      if (dotDirs <= 0.0 && dotAB < 0 && dotBA < 0) {
        line(posA.x, posA.y, posB.x, posB.y);
        used.add(i);
        used.add(j);
        break;
      }
    }
  }
  
  noFill();
}
