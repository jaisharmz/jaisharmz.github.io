// Load a figure script in node and return its spec, so `compute` can be run
// without a browser. Usage in tools/figtest/<id>.mjs:
//
//   import { load } from "./_load.mjs";
//   const spec = load("diff-example");
//   console.log(spec.compute.simulate({ ... }, 1));
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const repo = fileURLToPath(new URL("../../", import.meta.url));

// The same seeded generator as figlib's ctx.rng, for tests that need it.
export function rng(seed) {
  let s = (seed >>> 0) || 1;
  let spare = null;
  function uniform() {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
  function normal() {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0;
    while (u === 0) u = uniform();
    const w = uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * w);
    return r * Math.cos(2 * Math.PI * w);
  }
  return { uniform, normal };
}

export function load(id) {
  const specs = {};
  const sandbox = {
    FIG: { register: (name, spec) => { specs[name] = spec; }, ctx: { rng } },
    Math, console, Float64Array, Float32Array, Int32Array, Uint8Array, Array, Object, JSON
  };
  sandbox.window = sandbox;
  const code = readFileSync(`${repo}assets/js/figures/${id}.js`, "utf8");
  vm.runInNewContext(code, sandbox, { filename: `${id}.js` });
  if (!specs[id]) throw new Error(`${id}.js did not register "${id}"`);
  return specs[id];
}
