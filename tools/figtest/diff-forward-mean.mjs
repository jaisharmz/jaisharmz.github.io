// Caption numbers for diff-forward-mean: the derived mean and variance at t = 1
// against 150 Euler-Maruyama paths, at the default beta_max = 20 and at 0.02.
import { load } from "./_load.mjs";
const spec = load("diff-forward-mean");
const C = spec.compute, P = C.defaults;
const z = C.noise(P.nPaths, P.nSteps, P.seed);
for (const bmax of [20, 0.02, 30, 1]) {
  const r = C.simulate({ ...P, bmax }, z);
  console.log(`beta_max=${bmax} beta_min=${r.bmin}  int beta=${r.intBeta1.toFixed(5)}  alpha(1)=${r.alpha1.toFixed(5)}`);
  console.log(`  m(1) formula=${r.m1.toFixed(4)}  mean at t=1 (150 paths)=${r.mean1.toFixed(4)}  m(1)/x0=${(r.m1 / P.x0).toFixed(4)}`);
  console.log(`  Var(1) formula=${r.var1.toFixed(4)}  Var(1) paths=${r.var1Sample.toFixed(4)}  sd formula=${Math.sqrt(r.var1).toFixed(4)}`);
}
