import { load } from "./_load.mjs";
const spec = load("cx-total-correlation");
const C = spec.compute, D = spec.defaults;
const f = (v, d = 4) => v.toFixed(d);

// closed form against log det of the explicit matrix
for (const [n, r] of [[2, 0.5], [5, 0.5], [20, 0.3], [10, -0.1]]) {
  const R = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : r)));
  console.log(`n=${n} r=${r}: closed form ${f(C.tc(n, r), 10)}, -1/2 log det R by Cholesky ${f(-0.5 * C.logdet(R), 10)}`);
}

const e = C.entropies(D.n, D.r, D.s);
const sim = C.simulate(D.n, D.r, D.s, D.days, D.seed, D.scatter);
console.log(`\nDefaults n = ${D.n}, r = ${D.r}, return sd s = ${D.s}, ${D.days} simulated days, seed ${D.seed}`);
console.log(`each marginal entropy ${f(e.marginal)} nats; sum of marginals ${f(e.sumMarginal)}; joint ${f(e.joint)}; metric ${f(e.tc)} nats (${f(e.tc / Math.LN2)} bits)`);
console.log(`estimated from simulated days ${f(sim.tcHat)}; sample corr of tokens 1, 2 = ${f(sim.r12, 3)}`);

console.log("\nmetric against r and n:");
for (const n of [2, 5, 10, 20]) {
  console.log(`  n=${String(n).padStart(2)}: ` + [0, 0.1, 0.3, 0.5, 0.9, 0.99].map((r) => `r=${r}: ${f(C.tc(n, r), 3)}`).join("  ") + `   min r ${f(C.rmin(n), 4)}`);
}
console.log(`n=2: r=-0.5 gives ${f(C.tc(2, -0.5))}, r=0.5 gives ${f(C.tc(2, 0.5))}`);
console.log(`n=20 at the lowest slider r (${Math.max(-0.99, Math.ceil((C.rmin(20) + 0.002) * 1000) / 1000)}): ${f(C.tc(20, Math.max(-0.99, Math.ceil((C.rmin(20) + 0.002) * 1000) / 1000)), 3)}`);
console.log(`per-token metric at r = 0.3: n=2 ${f(C.tc(2, 0.3) / 2)}, n=20 ${f(C.tc(20, 0.3) / 20)}`);

console.log("\nplug-in estimate at independence (true value 0), 2000 days, bias approx n(n-1)/(4N):");
for (const n of [2, 5, 20]) {
  const s = C.simulate(n, 0, D.s, D.days, D.seed, 0);
  console.log(`  n=${n}: estimate ${f(s.tcHat)}, n(n-1)/(4N) = ${f(n * (n - 1) / (4 * D.days))}`);
}
const s20 = C.simulate(20, 0.5, D.s, D.days, D.seed, 0);
console.log(`n=20, r=0.5: exact ${f(C.tc(20, 0.5))}, estimate ${f(s20.tcHat)}`);
