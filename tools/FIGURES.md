# Interactive figures

Every interactive figure on a writing page is two files:

```
src/figures/<id>.html         the title, the stage and the caption (inlined by the build)
assets/js/figures/<id>.js     FIG.register("<id>", { compute, mount })
```

The build replaces a `\webfig{<id>}` line in a LaTeX source with
`<figure class="fig" id="fig-<id>">` + the fragment, numbers it ("Figure 3."), renders any
`\( ... \)` math in the fragment with KaTeX, and loads `assets/js/figlib.js` and the figure's
script at the end of the page.

## The fragment

```html
<p class="fig__title">One sentence stating the one idea this figure shows.</p>
<div class="fig__stage" data-fig="<id>"></div>
<figcaption>What to try and what happens, with the numbers the figure itself computes:
"Slide \(\beta_{\max}\) down to 0.02. The mean at \(t=1\) stays at 0.995 of where it started."</figcaption>
```

- Add `class="fig fig--wide"` intent by starting the fragment with `<!-- wide -->`: the build then
  lets the figure extend 4rem past the text column on wide screens.
- Math in the title and caption uses `\( \)` only. No `$`.

## The script

```js
FIG.register("diff-example", {
  // Pure math, no DOM. Must run in node: tools/figtest/<id>.mjs calls it.
  compute: {
    simulate: function (params, seed) { /* ... */ return { /* numbers */ }; }
  },
  mount: function (stage, ctx) {
    var C = this.compute;                  // `this` is the spec
    var row = ctx.controls(stage);
    var s = ctx.slider(row, { label: "β max", min: 0.02, max: 30, step: 0.01, value: 20,
                              fmt: v => v.toFixed(2), oninput: draw });
    var cv = ctx.canvas(stage, { aspect: 0.55, minHeight: 240, maxHeight: 420, label: "..." });
    var ro = ctx.readout(stage);
    cv.onresize = draw;
    function draw() {
      var col = ctx.colors();              // read every draw: the theme can change
      cv.clear();
      var ax = ctx.axes(cv, { xmin: 0, xmax: 1, ymin: -3, ymax: 3, xlabel: "t", ylabel: "x" });
      ax.draw();
      // ... cv.g is the 2D context, ax.X / ax.Y map data to pixels
      ro.set({ "m(1)": ctx.fmt(1.23, 3) });
    }
    draw();
    return { draw: draw };                 // redrawn on theme change
  }
});
```

### `ctx` (from `assets/js/figlib.js`)

| call | does |
| --- | --- |
| `ctx.colors()` | `{paper, sunk, ink, soft, muted, rule, ruleSoft, accent, c: [blue, orange, green, red, purple]}`, theme aware |
| `ctx.alpha(color, a)` | the same color with alpha |
| `ctx.rng(seed)` | `{uniform(), normal()}`, seeded, so captions are reproducible |
| `ctx.controls(parent)` | a flex row for controls |
| `ctx.slider(parent, {label, min, max, step, value, fmt, oninput})` | `{value, set(v), input}` |
| `ctx.toggle(parent, {label, options: [{value, label}], value, onchange})` | segmented control |
| `ctx.button(parent, label, onclick)` | a button; add class `fig__btn--primary` for play |
| `ctx.readout(parent)` | `{set({label: text})}` live numbers under the drawing |
| `ctx.canvas(parent, {aspect, minHeight, maxHeight, label})` | HiDPI canvas `{g, w, h, clear(), onresize}`; width follows the column |
| `ctx.axes(canvas, {xmin, xmax, ymin, ymax, xlabel, ylabel, margin, box, xticks, yticks, xfmt, yfmt, grid})` | `{X, Y, invX, invY, box, draw()}` |
| `ctx.loop(stage, fn)` | rAF loop paused off screen; `fn(dt)` returns `false` to stop; `{start, stop, running}` |
| `ctx.fmt(v, digits)` | fixed decimals with a real minus sign |
| `ctx.el(tag, cls, parent, text)` | create an element |
| `ctx.reducedMotion` | true when the reader asked for less motion: show the end state, offer a play button, never autoplay |

Layout classes: `fig__row` (two columns, stacks under 620px), `fig__row--3-2`, `fig__legend`
(with `<span><i style="background:..."></i>label</span>`).

## Rules

These come from the operator's `visual` guideline and are not optional.

- **Compute the real mechanism on a toy.** A figure that only looks like the result has no
  place on the page. If the text derives a formula, the figure simulates the process and
  overlays the formula, so the reader sees them agree (or sees exactly where they do not).
- **One idea per figure**, stated in the title sentence.
- **A control that moves the idea**: the knob that matters, a play button, or a toggle between
  the method and its baseline. The caption says what to try and what happens, in numbers.
- **Every number in a caption comes from running `compute` at the figure's own default
  settings and seed.** Write `tools/figtest/<id>.mjs` that loads the script in node with a stub
  `FIG.register`, runs `compute` at the defaults, and prints the numbers the caption quotes.
  Run it and copy the numbers from its output.
- **Live readouts** carry the numbers next to the drawing.
- **Honest labels.** Say "on this toy" where it matters. When a model in the text is
  underspecified, state the assumption the figure makes in the caption.
- **Both themes and phone width.** Read colors from `ctx.colors()` inside `draw()`, never
  hard-code hex. The figure must work at 340px wide: labels must not collide, controls wrap.
  Canvas text: `11px` or `12px` in `getComputedStyle(document.documentElement).getPropertyValue("--sans")`.
- **Fast.** Simulations of a few thousand particles at most per frame; precompute where you
  can. Animations pause off screen (`ctx.loop` does this) and never autoplay under reduced motion.
- **Notation matches the text** around the figure: same symbols, same names.
- **Words**: no em dashes, Oxford comma, plain sentences, no "simply", no exclamation marks.
  Controls are labeled in words ("noise term", "step size ε"), not just symbols.
- Plain ES2017 inside an IIFE-free file (the file only calls `FIG.register`). No imports, no
  libraries, no network.

## Checking a figure

- `node tools/figtest/<id>.mjs` prints the caption numbers.
- With the preview server running (`python3 -m http.server 7792 --directory <repo>`), open
  `http://127.0.0.1:7792/tools/figpreview.html?fig=<id>`, `&theme=dark`, `&w=375`.
