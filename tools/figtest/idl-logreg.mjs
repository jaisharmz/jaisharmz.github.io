// Prints every number the idl-logreg caption quotes, at the figure's defaults.
import { load } from "./_load.mjs";

const spec = load("idl-logreg");
const C = spec.compute, D = spec.defaults;
const at = (run, it) => run.rec.find(r => r.iter === it);

for (const kind of ["separable", "overlapping"]) {
  const d = C.data(kind, D.seed, D.perClass);
  const t0 = Date.now();
  const run = C.run(d, D.alpha, D.maxIter);
  const ms = Date.now() - t0;
  console.log(`\n${kind}: m = ${d.m}, alpha = ${D.alpha}, run of ${D.maxIter} iterations took ${ms} ms, ${run.rec.length} checkpoints`);
  console.log(`  every point correct from iteration: ${run.firstAll}`);
  for (const it of [0, 10, 100, 1000, 10000, 100000]) {
    const r = at(run, it);
    if (!r) continue;
    console.log(`  iteration ${String(it).padStart(6)}: J = ${r.J.toFixed(4)}, accuracy = ${(100 * r.acc).toFixed(1)}%, ` +
      `|theta| = ${r.norm.toFixed(2)}, theta = (${r.th.map(v => v.toFixed(2)).join(", ")})`);
  }
}
