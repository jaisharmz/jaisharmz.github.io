// Prints every number the idl-linreg caption quotes, at the figure's defaults.
import { load } from "./_load.mjs";

const spec = load("idl-linreg");
const C = spec.compute, D = spec.defaults;
const d = C.data(D.seed, D.m);
const opt = C.optimum(d);
const lim = C.alphaLimit(d);
const f = (v, k = 3) => v.toFixed(k);

console.log(`data: m = ${d.m}, seed = ${D.seed}, default alpha = ${D.alpha}, start = (${D.start})`);
console.log(`least squares optimum: theta0 = ${f(opt.t0)}, theta1 = ${f(opt.t1)}, J = ${f(opt.J, 4)}`);
console.log(`curvatures: lmax = ${f(lim.lmax)}, lmin = ${f(lim.lmin)}; alpha limit 2/lmax = ${f(lim.limit, 2)}`);
console.log(`J at the start: ${f(C.cost(d, D.start[0], D.start[1]), 3)}`);

for (const a of [0.05, 0.3, 0.8, 0.85, 0.95]) {
  const r = C.run(d, D.start, a, D.maxIter);
  const k = C.itersToReach(d, D.start, a, 0.01, D.maxIter);
  console.log(`alpha = ${a}: ${r.status} after ${r.iter} steps, final J = ${r.J.toPrecision(4)}, ` +
    `steps until theta is within 0.01 of least squares: ${k}`);
}
// how fast the diverging rate blows up
let t = D.start.slice();
const Js = [];
for (let k = 1; k <= 40; k++) { t = C.step(d, t[0], t[1], 0.95); if (k % 10 === 0) Js.push(`step ${k}: J = ${C.cost(d, t[0], t[1]).toPrecision(3)}`); }
console.log(`alpha = 0.95: ${Js.join(", ")}`);
