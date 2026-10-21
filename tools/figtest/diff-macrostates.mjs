// node tools/figtest/diff-macrostates.mjs
// Prints every number the diff-macrostates caption quotes, at the figure's defaults.
import { load } from "./_load.mjs";

const spec = load("diff-macrostates");
const C = spec.compute;
const d = spec.defaults;

function show(n) {
  const t0 = Date.now();
  const e = C.enumerate(n);
  const ms = Date.now() - t0;
  const lag = C.lagrange(n);
  const fit = C.fit(e.mean);
  const cf = C.closedForm(n);
  let maxErr = 0;
  for (let j = 0; j <= n; j++) maxErr = Math.max(maxErr, Math.abs(e.mean[j] - cf.mean[j]));
  console.log(`n = ${n}${n === d.n ? " (default)" : ""}`);
  console.log(`  macrostates ${e.count}, microstates ${e.total.toExponential(4)} (stars and bars ${cf.total.toExponential(4)}), enumerate took ${ms} ms`);
  console.log(`  mean occupancy <n_j>/n, j = 0..6: ${Array.from(e.mean.slice(0, 7)).map(x => x.toFixed(4)).join(" ")}`);
  console.log(`  max |enumerated - closed form| over j: ${maxErr.toExponential(2)}`);
  console.log(`  lambda_2 fitted ${fit.lambda.toFixed(4)}, from the constraints ${lag.lambda.toFixed(4)}, ln 2 = ${Math.LN2.toFixed(4)}`);
  console.log(`  ratio <n_1>/<n_0> = ${(e.mean[1] / e.mean[0]).toFixed(4)}, <n_6>/<n_5> = ${(e.mean[6] / e.mean[5] || 0).toFixed(4)}`);
  console.log(`  macrostates tied for the largest W: ${e.tied}`);
  if (n >= 10) {
    const j = 10, curve = lag.c * Math.exp(-lag.lambda * j);
    console.log(`  tail at j = ${j}: bar ${e.mean[j].toExponential(3)}, Lagrange curve ${curve.toExponential(3)}, bar/curve ${(e.mean[j] / curve).toFixed(3)}`);
  }
  e.top.forEach((t, i) => {
    let last = 0; for (let j = 0; j <= n; j++) if (t.occ[j]) last = j;
    console.log(`  top ${i + 1}: (${t.occ.slice(0, Math.max(last, n <= 5 ? n : 0) + 1).join(",")}) W = ${t.W.toPrecision(4)} share ${(100 * t.share).toFixed(2)}%`);
  });
}

console.log(`k_B T implied by lambda_2 = ln 2 (text: lambda_2 = 1/(k_B T)): ${(1 / Math.LN2).toFixed(4)}`);
show(3);
show(d.n);
show(10);
show(30);
show(d.nmax);
