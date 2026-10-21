import { load } from "./_load.mjs";
const spec = load("cx-sampling");
const C = spec.compute, D = spec.defaults;

// 1. The text's discrete sum: left endpoints t_i = (i - 1) dt with n dt = T.
console.log("Check the derivation (mu = 0.5, R0 = 1, T = 1):");
for (const n of [1, 4, 8, 64]) {
  const q = C.parts(0.5, 1, 1, n);
  const dt = 1 / n;
  console.log(`  n = ${n}: continuous ${q.continuous.toFixed(8)}, discrete direct ${q.discreteDirect.toFixed(8)}, closed form ${q.discreteClosed.toFixed(8)}, ` +
    `difference ${(q.continuous - q.discreteDirect).toFixed(8)} vs exactLoss ${C.exactLoss(0.5, 1, 1, dt).toFixed(8)}`);
}
// second-order term: 1/mu - dt/(e^{mu dt} - 1) = dt/2 - mu dt^2/12 + O(dt^4)
const dt0 = 1 / 8;
console.log(`  bracket at dt = 1/8: ${(1 / 0.5 - dt0 / Math.expm1(0.5 * dt0)).toFixed(8)} vs dt/2 - mu dt^2/12 = ${(dt0 / 2 - 0.5 * dt0 * dt0 / 12).toFixed(8)}`);
console.log(`  mu = -0.5 gives E[Loss] = ${C.exactLoss(-0.5, 1, 1, 1 / 8).toFixed(5)} (negative: the provider over-bills when utilization falls)`);

// 2. Monte Carlo at the defaults
const p = { mu: D.mu, sigma: D.sigma, R0: D.R0, T: D.T, M: D.M, paths: D.paths, maxLog2n: D.maxLog2n };
const t0 = Date.now();
const s = C.simulate(p, D.seed);
console.log(`\nDefaults mu = ${D.mu}, sigma = ${D.sigma}, R0 = ${D.R0}, T = ${D.T}, ${D.paths} paths on ${D.M} fine steps (${Date.now() - t0} ms)`);
for (let j = 0; j < s.ns.length; j++) {
  const ratioEx = j + 1 < s.ns.length ? (s.exact[j] / s.exact[j + 1]).toFixed(3) : "";
  const ratioMC = j + 1 < s.ns.length ? (s.mean[j] / s.mean[j + 1]).toFixed(3) : "";
  console.log(`  n = ${String(s.ns[j]).padStart(2)}  dt = ${s.dts[j].toFixed(5)}  exact ${s.exact[j].toFixed(5)}  first ${s.first[j].toFixed(5)}  ` +
    `MC ${s.mean[j].toFixed(5)} +- ${s.se[j].toFixed(5)}  sd ${s.sd[j].toFixed(5)}  over-billed paths ${(s.negShare[j] * 100).toFixed(1)}%  ` +
    `ratio to next exact ${ratioEx} MC ${ratioMC}  path0 ${s.loss0[j].toFixed(5)}`);
}
const j = D.log2n;
const exHalf = C.exactLoss(p.mu, p.R0, p.T, p.T / s.ns[j] / 2);
console.log(`\nAt the default n = ${s.ns[j]}: exact ${s.exact[j].toFixed(5)}, first order ${s.first[j].toFixed(5)}, MC ${s.mean[j].toFixed(5)} +- ${s.se[j].toFixed(5)}, ` +
  `halving ratio exact ${(s.exact[j] / exHalf).toFixed(3)}, MC ${(s.mean[j] / s.mean[j + 1]).toFixed(3)}, sd ${s.sd[j].toFixed(5)}`);
console.log(`first order overstates the exact loss by ${((s.first[j] / s.exact[j] - 1) * 100).toFixed(1)}% at n = ${s.ns[j]} and by ${((s.first[0] / s.exact[0] - 1) * 100).toFixed(1)}% at n = 1`);

// 3. sigma = 0: the path is deterministic, so MC must equal the exact formula up to the trapezoid error
const s0 = C.simulate({ ...p, sigma: 0, paths: 50 }, D.seed);
console.log(`sigma = 0, n = 8: MC ${s0.mean[3].toFixed(7)} vs exact ${s0.exact[3].toFixed(7)}`);
