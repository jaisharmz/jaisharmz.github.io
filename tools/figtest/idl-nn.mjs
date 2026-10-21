// Prints every number the idl-nn caption quotes, at the figure's defaults.
import { load } from "./_load.mjs";

const spec = load("idl-nn");
const C = spec.compute, D = spec.defaults;
console.log(`defaults: H = ${D.H}, m = ${D.m}, alpha = ${D.alpha}, epochs = ${D.epochs}, seed = ${D.seed}`);

for (const kind of ["circles", "xor"]) {
  console.log(`\n${kind}`);
  for (const act of ["sigmoid", "tanh", "relu", "linear"]) {
    const r = C.train(kind, D.H, act, D.epochs, D.alpha, D.seed, D.m);
    let line = `  ${act.padEnd(7)} H = ${D.H}: cost J = ${r.J.toFixed(4)}, accuracy = ${(100 * r.acc).toFixed(1)}%`;
    if (act === "linear") {
      const lr = C.collapse(r.P);
      line += `, collapsed logistic regression theta = (${lr.b.toFixed(3)}, ${lr.w1.toFixed(3)}, ${lr.w2.toFixed(3)})`;
    }
    console.log(line);
  }
  // how the number of hidden units changes the sigmoid network
  const hs = [];
  for (let H = 1; H <= 8; H++) {
    const r = C.train(kind, H, "sigmoid", D.epochs, D.alpha, D.seed, D.m);
    hs.push(`H=${H}: ${(100 * r.acc).toFixed(1)}%`);
  }
  console.log(`  sigmoid by H: ${hs.join(", ")}`);
}

// check: with a linear activation the network output equals the collapsed logistic regression
const r = C.train("circles", D.H, "linear", D.epochs, D.alpha, D.seed, D.m);
const lr = C.collapse(r.P);
let maxdiff = 0;
for (let i = 0; i < r.d.m; i++) {
  const a = C.predict(r.P, "linear", r.d.x1[i], r.d.x2[i]);
  const b = 1 / (1 + Math.exp(-(lr.b + lr.w1 * r.d.x1[i] + lr.w2 * r.d.x2[i])));
  maxdiff = Math.max(maxdiff, Math.abs(a - b));
}
console.log(`\nlinear network vs collapsed logistic regression, largest gap in h over the data: ${maxdiff.toExponential(2)}`);
