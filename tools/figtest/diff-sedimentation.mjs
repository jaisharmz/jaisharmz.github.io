// node tools/figtest/diff-sedimentation.mjs
// Prints every number the diff-sedimentation caption quotes, at the figure's defaults.
// The animation advances a fixed number of steps per frame, so these runs replay
// exactly what the page shows for the default seed.
import { load } from "./_load.mjs";

const spec = load("diff-sedimentation");
const C = spec.compute;
const d0 = spec.defaults;
const P = (over) => Object.assign({}, d0, over);
const T = 40;

function run(over, label) {
  const p = P(over), dv = C.derived(p);
  const t0 = Date.now();
  const st = C.simulate(p, d0.seed, T);
  const ms = Date.now() - t0;
  console.log(`${label}: kT ${p.kT}, D = ${dv.D.toFixed(3)} (${p.mult}x kT/gamma), dt ${dv.dt}, steps/frame ${C.stepsPerFrame(p)}, ` +
    `frames ${st.frames}, t = ${st.t.toFixed(2)}`);
  console.log(`   mean height now ${st.mean.toFixed(3)}, averaged over t in [${T / 2}, ${T}] ${st.avgMean.toFixed(3)}, ` +
    `Boltzmann k_BT/(mg) ${dv.lB.toFixed(3)}, dynamics predict gamma D/(mg) ${dv.lD.toFixed(3)}, above z=${p.zTop}: ${(100 * st.above).toFixed(1)}%  [${ms} ms]`);
}

run({}, "default");
run({ mult: 4 }, "D = 4x");
run({ mult: 0.25 }, "D = 0.25x");
run({ kT: 2 }, "kT = 2");
run({ kT: 0.5 }, "kT = 0.5");

// settling from the well-mixed start at the defaults
const p = P({}), s = C.init(p, d0.seed), spf = C.stepsPerFrame(p);
const marks = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20];
let line = "settling at defaults, mean height at t =";
for (const m of marks) {
  while (s.t < m - 1e-9) C.step(s, p, spf);
  line += ` ${m}: ${C.stats(s, p).mean.toFixed(3)};`;
}
console.log(line);
console.log(`frames per time unit ${(1 / d0.simPerFrame).toFixed(1)}; at 60 frames per second, 1 time unit = ${(1 / d0.simPerFrame / 60).toFixed(2)} s`);
