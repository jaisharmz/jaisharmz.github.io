// Caption numbers for diff-noising: mean distance from the origin and the share of
// 400 particles within 2 s of a data mode at t = 0, 0.2, 0.5, and 1, for the forward
// VP SDE and for the reverse SDE run with the exact score.
import { load } from "./_load.mjs";
const C = load("diff-noising").compute, P = C.defaults;
const t0 = Date.now();
const runs = { forward: C.run("forward", P), reverse: C.run("reverse", P) };
console.log("compute ms (both runs):", Date.now() - t0);
for (const dir of ["forward", "reverse"]) {
  const R = runs[dir];
  for (const t of [0, 0.1, 0.2, 0.3, 0.5, 1]) {
    const k = Math.round(t * P.steps);
    console.log(`${dir} t=${t} meanDist=${R.meanDist[k].toFixed(3)} share=${R.share[k].toFixed(3)} faces=${R.near[0][k].toFixed(3)} cars=${R.near[1][k].toFixed(3)}`);
  }
  const lab = Array.from(R.label);
  console.log(`${dir} labels: faces=${lab.filter((v) => v === 0).length} cars=${lab.filter((v) => v === 1).length}`);
}
console.log("alpha(1) =", C.alpha(1).toFixed(5), " data share within 2s for a 2D Gaussian = 1 - e^-2 =", (1 - Math.exp(-2)).toFixed(4));
console.log("E|z| for z ~ N(0, I_2) = sqrt(pi/2) =", Math.sqrt(Math.PI / 2).toFixed(4));
console.log("mode distances from origin:", C.modes.map((m) => Math.hypot(...m.mu).toFixed(3)).join(", "));
