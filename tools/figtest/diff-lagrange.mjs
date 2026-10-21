// node tools/figtest/diff-lagrange.mjs
// Prints every number the diff-lagrange caption quotes, at the figure's defaults.
import { load } from "./_load.mjs";

const spec = load("diff-lagrange");
const C = spec.compute;
const d = spec.defaults;
const f3 = (v) => v.toFixed(3);

function show(name, a) {
  console.log(`${name}: theta ${a.theta.toFixed(3)} deg, point (${f3(a.x)}, ${f3(a.y)}), g = ${f3(a.g)}, f = ${f3(a.f)}, ` +
    `angle(grad f, grad g) = ${a.angle.toFixed(2)} deg, grad f . tangent = ${f3(a.along)}, ` +
    `lambda fit = ${f3(a.lambda)}, |grad f|/|grad g| = ${f3(Math.hypot(...a.gradF) / Math.hypot(...a.gradG))}, parallel ${a.parallel}`);
}

const o = C.optima();
show(`default (theta = ${d.thetaDeg})`, C.at(d.thetaDeg));
show("maximum", o.max);
show("minimum", o.min);
console.log(`analytic: optimum at (sqrt 2, 1/sqrt 2) = (${f3(Math.SQRT2)}, ${f3(Math.SQRT1_2)}), f = 2 sqrt 2 = ${f3(2 * Math.SQRT2)}, lambda = 1/(2 sqrt 2) = ${f3(1 / (2 * Math.SQRT2))}`);
for (const th of [44, 44.5, 45, 45.5, 46, 90, 0]) show(`  theta ${th}`, C.at(th));
