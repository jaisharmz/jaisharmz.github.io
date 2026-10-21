// Prints the numbers the diff-langevin caption quotes: each noise setting at the
// default step size, after the default number of steps, from the default seed.
import { load } from "./_load.mjs";
const spec = load("diff-langevin");
const C = spec.compute, D = spec.defaults;
const f = (a, d = 3) => a.map((x) => x.toFixed(d)).join(" / ");
const tP = C.targetStats(1), tP2 = C.targetStats(2);
console.log("target P   weights", f(tP.w), " variance", f(tP.v));
console.log("target P^2 weights", f(tP2.w), " variance", f(tP2.v), " P^2 var / P var", f(tP2.v.map((v, k) => v / tP.v[k])));
const s2 = C.mix.s * C.mix.s;
const ula = (c2, eps) => c2 / (2 / s2 - eps / (s2 * s2)) / s2; // stationary var / s^2 of x <- x - eps x/s^2 + sqrt(c2 eps) z
console.log(`one Gaussian mode, eps=${D.eps}: ULA variance / s^2 = ${ula(2, D.eps).toFixed(3)} (sqrt(2 eps)), ${ula(1, D.eps).toFixed(3)} (sqrt(eps))`);
const init = C.init(D.n, D.seed, D.cloud); const s0 = C.snapshot(init);
console.log("initial cloud weights (basin masses)", f(s0.w));
for (const eps of [D.eps, 0.02, 0.2]) {
  for (const noise of ["2", "1", "0"]) {
    const t0 = Date.now();
    const r = C.simulate({ ...D, eps, noise }, D.seed);
    const ratio = r.avg.v.map((v, k) => v / tP.v[k]);
    console.log(`eps=${eps} noise=${noise} steps=${D.steps}: weights ${f(r.avg.w)}  variance/target ${f(ratio, 2)}  (avg of last ${r.avg.count * D.every} steps, ${Date.now() - t0} ms)`);
  }
}
