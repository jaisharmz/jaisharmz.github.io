// Caption numbers for diff-fokker-planck: L1 gap between the finite-volume Fokker-Planck
// density and a particle histogram, PDE mass, and means, for each switch setting and
// particle count, starting from 0.4 N(-2.5, 0.35^2) + 0.6 N(2, 0.35^2).
import { load } from "./_load.mjs";
const C = load("diff-fokker-planck").compute, P = C.defaults;
const times = [0, 0.5, 1, 3];
const show = (label, rows) => rows.forEach((r) => console.log(
  `${label} t=${r.t.toFixed(2)} L1=${r.L1.toFixed(4)} mass=${r.mass.toFixed(10)} meanPDE=${r.meanPDE.toFixed(4)} meanPart=${r.meanParticles.toFixed(4)}` +
  (r.peak !== undefined ? ` peak=${r.peak.toFixed(3)} minDensity=${r.minDensity.toExponential(2)}` : "")));
let t0 = Date.now();
show("both N=2000", C.run({ ...P }, times));
console.log("ms:", Date.now() - t0);
for (const N of [500, 8000]) show(`both N=${N}`, C.run({ ...P, N }, [3]));
show("drift only N=2000", C.run({ ...P, diff: false }, times));
show("diffusion only N=2000", C.run({ ...P, drift: false }, times));
// theory: mean m(t) = m(0) e^{-t} under f = -x (both on or drift only)
const m0 = 0.4 * -2.5 + 0.6 * 2;
console.log("m(0) =", m0, " m(0) e^-1 =", (m0 * Math.exp(-1)).toFixed(4), " m(0) e^-3 =", (m0 * Math.exp(-3)).toFixed(4));
// seed spread of the L1 gap at t = 3, N = 2000
const gaps = [];
for (let seed = 1; seed <= 6; seed++) gaps.push(C.run({ ...P, seed }, [3])[0].L1.toFixed(3));
console.log("L1 at t=3, N=2000, seeds 1..6:", gaps.join(" "));
