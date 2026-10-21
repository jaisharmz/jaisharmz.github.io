// Prints the numbers the diff-equipartition caption quotes (units m = 1, k_B T = 1).
import { load } from "./_load.mjs";
const spec = load("diff-equipartition");
const C = spec.compute, D = spec.defaults;
for (const [kappa, gamma] of [[1, 1], [0.25, 1], [4, 1], [1, 0.25], [1, 4], [2, 1], [0.5, 1]]) {
  const p = { ...D, kappa, gamma };
  const t0 = Date.now();
  const r = C.simulate(p, D.seed);
  const tm = p.m / p.gamma;
  console.log(`kappa=${kappa} gamma=${gamma}: C=${r.C.toFixed(3)}  <v^2> measured ${r.stationary.toFixed(3)} (t in ${r.window[0]}..${r.window[1]})  C/(2 gamma m) ${r.predicted.toFixed(3)}  kT/m ${r.target.toFixed(3)}  acf(m/gamma=${tm.toFixed(2)}) ${C.acfAt(r, tm).toFixed(3)} vs e^-1 ${Math.exp(-1).toFixed(3)}  acf(1) ${C.acfAt(r, 1).toFixed(3)} vs ${Math.exp(-gamma).toFixed(3)}  (${Date.now() - t0} ms)`);
}
// time to reach half the plateau from rest: ln 2 / (2 gamma / m)
console.log("half-plateau time at gamma=1:", (Math.log(2) / 2).toFixed(3), " at gamma=0.25:", (Math.log(2) / 0.5).toFixed(3));
// sampling noise of the measured plateau at the defaults: 20 other seeds
{
  const vals = [];
  for (let seed = 101; seed <= 120; seed++) vals.push(C.simulate(D, seed).stationary);
  const mean = vals.reduce((a, b) => a + b) / vals.length;
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (vals.length - 1));
  console.log(`plateau over seeds 101..120 at defaults: mean ${mean.toFixed(3)}  sd ${sd.toFixed(3)}  min ${Math.min(...vals).toFixed(3)}  max ${Math.max(...vals).toFixed(3)}`);
  const r = C.simulate({ ...D, gamma: 0.25 }, D.seed);
  console.log("gamma=0.25 acf at tau=4 (=m/gamma):", C.acfAt(r, 4).toFixed(3));
}
