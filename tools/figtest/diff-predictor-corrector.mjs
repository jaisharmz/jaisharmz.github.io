// Prints the numbers the diff-predictor-corrector caption quotes, at the default seed.
import { load } from "./_load.mjs";
const spec = load("diff-predictor-corrector");
const C = spec.compute, D = spec.defaults;
const L = C.level(D.tEnd);
console.log(`alpha(1) = ${C.alpha(1).toFixed(4)}, p_1 component variance ${C.level(1).v0.toFixed(4)}; at tEnd=${D.tEnd}: component variance ${L.v0.toFixed(5)} (data ${0.15 ** 2})`);
console.log(`corrector step at tEnd: eps = ${(D.c * L.vmin).toFixed(5)}; at t=0.5: ${(D.c * C.level(0.5).vmin).toFixed(4)}; at t=1: ${(D.c * C.level(1).vmin).toFixed(4)}`);
console.log(`true heavier-mode weight left of boundary ${C.boundary(D.tEnd).toFixed(4)}: ${C.cdf(C.boundary(D.tEnd), L).toFixed(4)}`);
console.log(`W1 floor (expected, 2000 exact draws): ${C.w1Floor(D.n, D.tEnd).toFixed(4)}`);
// check the floor against real exact draws
{
  const vals = [];
  for (let seed = 1; seed <= 20; seed++) {
    const r = spec.compute; const R = (await import("./_load.mjs")).rng(1000 + seed);
    const x = new Float64Array(D.n);
    for (let j = 0; j < D.n; j++) { const k = R.uniform() < 0.7 ? 0 : 1; x[j] = (k ? L.m1 : L.m0) + Math.sqrt(k ? L.v1 : L.v0) * R.normal(); }
    vals.push(r.w1(x, D.tEnd));
  }
  console.log(`W1 of exact draws over 20 seeds: mean ${(vals.reduce((a, b) => a + b) / 20).toFixed(4)}  min ${Math.min(...vals).toFixed(4)}  max ${Math.max(...vals).toFixed(4)}`);
}
const t0 = Date.now();
console.log("N      " + [0, 1, 2, 3, 5].map((K) => `K=${K}: W1 (left weight, evals)`.padEnd(30)).join(""));
for (const N of [3, 4, 5, 6, 8, 10, 16, 20, 32, 64, 200]) {
  let line = String(N).padEnd(7);
  for (const K of [0, 1, 2, 3, 5]) {
    const r = C.run({ ...D, N, K }, D.seed);
    line += `${r.w1.toFixed(3)} (${r.wLeft.toFixed(3)}, ${r.evals}${r.outside > 0 ? ", out " + (100 * r.outside).toFixed(1) + "%" : ""})`.padEnd(30);
  }
  console.log(line);
}
console.log(`(${Date.now() - t0} ms)`);
// Large-N bias check with 20000 samples (not the figure's 2000), to separate the
// corrector's own bias from sampling noise.
{
  console.log("20000-sample check, seed default:");
  for (const N of [10, 200]) {
    let line = `  N=${N}`;
    for (const K of [0, 1, 2]) { const r = C.run({ ...D, N, K, n: 20000 }, D.seed); line += `  K=${K}: W1 ${r.w1.toFixed(3)} weight ${r.wLeft.toFixed(3)}`; }
    console.log(line);
  }
  // Unadjusted Langevin on one Gaussian with eps = c * variance settles at variance / (1 - c/2):
  console.log(`  ULA variance inflation at c=${D.c}: ${(1 / (1 - D.c / 2)).toFixed(3)}; weights tempered by it: ${(0.7 ** (1 - D.c / 2) / (0.7 ** (1 - D.c / 2) + 0.3 ** (1 - D.c / 2))).toFixed(3)}`);
}
