// Prints every number the idl-derivative caption quotes, at the figure's defaults.
import { load } from "./_load.mjs";

const spec = load("idl-derivative");
const C = spec.compute, D = spec.defaults;
const x = D.x;
console.log(`f(x) = x^2 + 2x + 3, point x = ${x}, f(x) = ${C.f(x)}, f'(x) = ${C.fprime(x)}`);
for (const r of C.table(x, [D.h, 1, 0.1, 0.01, 0.001, -D.h, -0.1, -0.001])) {
  console.log(`h = ${String(r.h).padStart(6)}: secant slope = ${r.slope.toFixed(6)}, minus f'(x) = ${r.gap.toFixed(6)}`);
}
