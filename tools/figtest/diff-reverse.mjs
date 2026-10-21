// Caption numbers for diff-reverse: mode shares and within-mode spread of 300
// reverse-time samples (exact score) for each sampler, against 0.30 / 0.70 and 0.30.
import { load } from "./_load.mjs";
const C = load("diff-reverse").compute, P = C.defaults;
for (const mode of ["sde", "drift", "ode"]) {
  const r = C.sample({ ...P, mode });
  const s = r.stats;
  console.log(`${mode.padEnd(5)} left n=${s.nLeft} share=${s.shareLeft.toFixed(3)} sd=${s.sdLeft.toFixed(3)} | right n=${s.nRight} share=${s.shareRight.toFixed(3)} sd=${s.sdRight.toFixed(3)}`);
}
// how many distinct end points the no-noise run produces (rounded to 0.01)
const d = C.sample({ ...P, mode: "drift" });
const uniq = new Set(Array.from(d.final, (v) => v.toFixed(2)));
console.log("drift-only distinct end points at 0.01 resolution:", uniq.size, "of", P.nPart);
const lo = Math.min(...d.final.filter((v) => v > C.split)), hi = Math.max(...d.final.filter((v) => v > C.split));
console.log("drift-only right-mode range:", lo.toFixed(4), "to", hi.toFixed(4));
// the starting share left of the split for N(0,1)
const s0 = Array.from({ length: P.nPart }, (_, i) => d.traj[i * d.K]);
console.log("starting points left of -0.25:", s0.filter((v) => v < C.split).length);
// seed spread for the SDE: shares over seeds 1..10
const shares = [];
for (let seed = 1; seed <= 10; seed++) shares.push(C.sample({ ...P, mode: "sde", seed }).stats.shareLeft.toFixed(3));
console.log("sde left share over seeds 1..10:", shares.join(" "));
