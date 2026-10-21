// Caption numbers for diff-random-walk: sample variance of 4000 walkers against
// 2Dt at t = 1 (the figure's default time) and t = 2, for each step size, D = 0.5.
import { load } from "./_load.mjs";
const C = load("diff-random-walk").compute, P = C.defaults;
for (const t of [1, 2]) {
  for (const dx of C.steps) {
    const r = C.run({ ...P, dx, t });
    console.log(`t=${t} dx=${dx} dt=${r.dt.toFixed(4)} steps=${r.n} t_actual=${r.t.toFixed(4)} mean=${r.mean.toFixed(4)} variance=${r.variance.toFixed(4)} 2Dt=${r.theory.toFixed(4)} ratio=${(r.variance / r.theory).toFixed(4)}`);
  }
}
// batching check: 25 steps at once equals 25 single steps
const a = C.init(P.N, P.seed); C.step(a, 25);
const b = C.init(P.N, P.seed); for (let i = 0; i < 25; i++) C.step(b, 1);
console.log("batch-independent:", a.k.every((v, i) => v === b.k[i]));
