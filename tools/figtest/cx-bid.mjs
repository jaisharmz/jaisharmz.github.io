import { load } from "./_load.mjs";
const spec = load("cx-bid");
const C = spec.compute, D = spec.defaults;

// Regularized lower incomplete gamma against known values (scipy.special.gammainc,
// and closed forms: P(n, x) = 1 - e^-x sum_{j<n} x^j / j!, P(1/2, x) = erf(sqrt x)).
const known = [
  [3, 2, 0.32332358381693654], [0.5, 1, 0.8427007929497151], [10, 10, 0.5420702855281478],
  [5, 3, 0.18473675547622787], [4.5, 7.3, 0.8974743197859523], [12, 3, 7.138662897420658e-05],
  [1.5, 0.01, 0.0007477553393911979], [20, 35, 0.9976754933921579], [1, 2.5, 1 - Math.exp(-2.5)]
];
let worst = 0;
for (const [a, x, v] of known) {
  const got = C.gammaP(a, x), rel = Math.abs(got - v) / v;
  worst = Math.max(worst, rel);
  console.log(`P(${a}, ${x}) = ${got.toPrecision(15)}   expected ${v.toPrecision(15)}   rel err ${rel.toExponential(1)}`);
}
console.log(`lnGamma(10) = ${C.lnGamma(10)} (ln 9! = ${Math.log(362880)}); lnGamma(0.5) = ${C.lnGamma(0.5)} (ln sqrt pi = ${0.5 * Math.log(Math.PI)})`);
console.log(`worst relative error ${worst.toExponential(2)}`);
if (worst > 1e-10) throw new Error("incomplete gamma is off");

// density integrates to one and matches the CDF
let s = 0; const k = D.k, lam = D.lam, h = 1e-4;
for (let x = h / 2; x < 20; x += h) s += C.pdf(x, k, lam) * h;
console.log(`integral of the density = ${s.toFixed(6)}; F(2) = ${C.cdf(2, k, lam).toFixed(6)}`);

const p = { k: D.k, lam: D.lam, C: D.C };
const o = C.optimum(p);
const sim = C.auctions(p, o.B, D.auctions, D.seed);
console.log("\nDefaults k = %s, lambda = %s, C = %s", D.k, D.lam, D.C);
console.log(`mean of B* = ${o.mean.toFixed(3)}, sd = ${o.sd.toFixed(3)}`);
console.log(`best ask B = ${o.B.toFixed(3)}, win prob = ${(o.win * 100).toFixed(1)}%, expected profit = ${o.profit.toFixed(3)}`);
console.log(`margin B - C = ${o.margin.toFixed(4)}, inverse hazard (1-F)/f = ${o.invHazard.toFixed(4)}`);
console.log(`simulated: win rate ${(sim.winRate * 100).toFixed(1)}%, mean profit ${sim.meanProfit.toFixed(3)} over ${D.auctions}`);
// profit at nearby asks for the caption
for (const B of [1.5, 2.0, 2.5, 3.0]) console.log(`  B = ${B}: profit ${C.profit(B, p).toFixed(3)}, win ${((1 - C.cdf(B, k, lam)) * 100).toFixed(1)}%`);

for (const q of [{ k: 4, lam: 2, C: 0 }, { k: 4, lam: 2, C: 2 }, { k: 4, lam: 2, C: 3 }, { k: 12, lam: 6, C: 1 }, { k: 1, lam: 0.5, C: 1 }, { k: 12, lam: 2, C: 1 }]) {
  const r = C.optimum(q);
  console.log(`k=${q.k} lam=${q.lam} C=${q.C}: best B ${r.B.toFixed(3)}, win ${(r.win * 100).toFixed(1)}%, profit ${r.profit.toFixed(3)}, mean B* ${r.mean.toFixed(2)}, margin ${r.margin.toFixed(3)} vs 1/h ${r.invHazard.toFixed(3)}`);
}
