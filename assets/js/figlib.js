/* Interactive figure runtime for the writing pages.

   A figure is a pair of files:
     src/figures/<id>.html          title, stage and caption, inlined by the build
     assets/js/figures/<id>.js      FIG.register("<id>", { compute, mount })

   `compute` holds the pure math (no DOM), so it can be run and checked in node.
   `mount(stage, ctx)` builds controls and canvases inside the stage element and may
   return { draw, destroy }. `draw` is called again on resize and on theme change.
   Figures mount lazily, when their stage comes within 300px of the viewport. */
(function () {
  "use strict";

  var specs = {};
  var root = document.documentElement;

  /* ---------- colors ---------------------------------------------------- */

  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  function colors() {
    return {
      paper: cssVar("--paper"),
      sunk: cssVar("--paper-sunk"),
      ink: cssVar("--ink"),
      soft: cssVar("--ink-soft"),
      muted: cssVar("--muted"),
      rule: cssVar("--rule"),
      ruleSoft: cssVar("--rule-soft"),
      accent: cssVar("--accent"),
      // categorical series, theme aware: blue, orange, green, red, purple
      c: [cssVar("--fig-1"), cssVar("--fig-2"), cssVar("--fig-3"), cssVar("--fig-4"), cssVar("--fig-5")]
    };
  }

  // "#1a4f8c" or "rgb(...)" plus alpha -> rgba string
  function alpha(color, a) {
    color = color.trim();
    if (color[0] === "#") {
      var h = color.slice(1);
      if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
      var n = parseInt(h, 16);
      return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + a + ")";
    }
    var m = color.match(/[\d.]+/g);
    if (!m) return color;
    return "rgba(" + m[0] + "," + m[1] + "," + m[2] + "," + a + ")";
  }

  /* ---------- random numbers ------------------------------------------- */

  // mulberry32: small, fast, seedable. normal() uses Box-Muller.
  function rng(seed) {
    var s = (seed >>> 0) || 1;
    var spare = null;
    function uniform() {
      s |= 0; s = s + 0x6D2B79F5 | 0;
      var t = Math.imul(s ^ s >>> 15, 1 | s);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
    function normal() {
      if (spare !== null) { var v = spare; spare = null; return v; }
      var u = 0, w = 0;
      while (u === 0) u = uniform();
      w = uniform();
      var r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * w);
      return r * Math.cos(2 * Math.PI * w);
    }
    return { uniform: uniform, normal: normal };
  }

  /* ---------- DOM helpers ---------------------------------------------- */

  function el(tag, cls, parent, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  // A row of controls. Figures put sliders, toggles and buttons in one of these.
  function controls(parent) {
    return el("div", "fig__controls", parent);
  }

  function slider(parent, o) {
    var wrap = el("label", "fig__slider", parent);
    var top = el("span", "fig__slider-top", wrap);
    el("span", "fig__slider-label", top, o.label);
    var val = el("span", "fig__slider-value", top);
    var input = el("input", null, wrap);
    input.type = "range";
    input.min = o.min; input.max = o.max; input.step = o.step || "any";
    input.value = o.value;
    var fmt = o.fmt || function (v) { return String(v); };
    function show() { val.textContent = fmt(+input.value); }
    show();
    input.addEventListener("input", function () {
      show();
      if (o.oninput) o.oninput(+input.value);
    });
    return {
      el: wrap,
      input: input,
      get value() { return +input.value; },
      set: function (v) { input.value = v; show(); }
    };
  }

  // Segmented control. options: [{value, label}] or strings.
  function toggle(parent, o) {
    var wrap = el("div", "fig__toggle", parent);
    wrap.setAttribute("role", "radiogroup");
    if (o.label) {
      el("span", "fig__toggle-label", wrap, o.label);
      wrap.setAttribute("aria-label", o.label);
    }
    var current = o.value;
    var buttons = [];
    (o.options || []).forEach(function (opt) {
      if (typeof opt !== "object") opt = { value: opt, label: String(opt) };
      var b = el("button", "fig__toggle-btn", wrap, opt.label);
      b.type = "button";
      b.setAttribute("role", "radio");
      b.dataset.value = opt.value;
      b.addEventListener("click", function () { set(opt.value, true); });
      buttons.push({ b: b, v: opt.value });
    });
    function set(v, fire) {
      current = v;
      buttons.forEach(function (x) { x.b.setAttribute("aria-checked", String(x.v === v)); });
      if (fire && o.onchange) o.onchange(v);
    }
    set(current, false);
    return { el: wrap, get value() { return current; }, set: set };
  }

  function button(parent, label, onclick) {
    var b = el("button", "fig__btn", parent, label);
    b.type = "button";
    b.addEventListener("click", onclick);
    return b;
  }

  // Live numbers beside a drawing. set({label: value, ...}) keeps insertion order.
  function readout(parent) {
    var wrap = el("dl", "fig__readout", parent);
    var rows = {};
    return {
      el: wrap,
      set: function (obj) {
        Object.keys(obj).forEach(function (k) {
          if (!rows[k]) {
            var row = el("div", "fig__readout-row", wrap);
            el("dt", null, row, k);
            rows[k] = el("dd", null, row);
          }
          rows[k].textContent = obj[k];
        });
      }
    };
  }

  /* ---------- canvas ---------------------------------------------------- */

  // HiDPI canvas whose width follows its container. Height is width * aspect,
  // clamped to [minHeight, maxHeight]. c.w and c.h are CSS pixels.
  function canvas(parent, o) {
    o = o || {};
    var wrap = el("div", "fig__canvas", parent);
    var cv = el("canvas", null, wrap);
    if (o.label) cv.setAttribute("aria-label", o.label);
    cv.setAttribute("role", "img");
    var c = { el: cv, wrap: wrap, g: cv.getContext("2d"), w: 0, h: 0, dpr: 1, onresize: null };
    function size() {
      var w = Math.max(200, Math.floor(wrap.clientWidth || parent.clientWidth || 600));
      var h = Math.round(w * (o.aspect || 0.6));
      if (o.minHeight) h = Math.max(h, o.minHeight);
      if (o.maxHeight) h = Math.min(h, o.maxHeight);
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (w === c.w && h === c.h && dpr === c.dpr) return false;
      c.w = w; c.h = h; c.dpr = dpr;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      cv.style.width = w + "px"; cv.style.height = h + "px";
      c.g.setTransform(dpr, 0, 0, dpr, 0, 0);
      return true;
    }
    size();
    c.clear = function () {
      c.g.setTransform(c.dpr, 0, 0, c.dpr, 0, 0);
      c.g.clearRect(0, 0, c.w, c.h);
    };
    c.resize = function () { if (size() && c.onresize) c.onresize(); };
    if (window.ResizeObserver) new ResizeObserver(function () { c.resize(); }).observe(wrap);
    else window.addEventListener("resize", c.resize);
    return c;
  }

  // Plot box with linear axes. box = {x, y, w, h} in CSS px, or a canvas (uses
  // default margins). Returns mapping functions and draws axes when draw() runs.
  function axes(c, o) {
    var m = o.margin || { l: 44, r: 12, t: 12, b: 34 };
    var box = o.box || { x: m.l, y: m.t, w: c.w - m.l - m.r, h: c.h - m.t - m.b };
    var ax = {
      box: box,
      xmin: o.xmin, xmax: o.xmax, ymin: o.ymin, ymax: o.ymax,
      X: function (x) { return box.x + (x - ax.xmin) / (ax.xmax - ax.xmin) * box.w; },
      Y: function (y) { return box.y + box.h - (y - ax.ymin) / (ax.ymax - ax.ymin) * box.h; },
      invX: function (px) { return ax.xmin + (px - box.x) / box.w * (ax.xmax - ax.xmin); },
      invY: function (py) { return ax.ymin + (box.y + box.h - py) / box.h * (ax.ymax - ax.ymin); },
      draw: function () {
        var g = c.g, col = colors();
        g.save();
        g.font = "11px " + cssVar("--sans");
        g.lineWidth = 1;
        var xt = o.xticks || niceTicks(ax.xmin, ax.xmax, Math.max(3, Math.floor(box.w / 80)));
        var yt = o.yticks || niceTicks(ax.ymin, ax.ymax, Math.max(3, Math.floor(box.h / 50)));
        if (o.grid !== false) {
          g.strokeStyle = col.ruleSoft;
          g.beginPath();
          xt.forEach(function (t) { var px = Math.round(ax.X(t)) + .5; g.moveTo(px, box.y); g.lineTo(px, box.y + box.h); });
          yt.forEach(function (t) { var py = Math.round(ax.Y(t)) + .5; g.moveTo(box.x, py); g.lineTo(box.x + box.w, py); });
          g.stroke();
        }
        g.strokeStyle = col.rule;
        g.strokeRect(box.x + .5, box.y + .5, box.w, box.h);
        g.fillStyle = col.muted;
        var fx = o.xfmt || fmtTick, fy = o.yfmt || fmtTick;
        g.textAlign = "center"; g.textBaseline = "top";
        xt.forEach(function (t) { g.fillText(fx(t), ax.X(t), box.y + box.h + 5); });
        g.textAlign = "right"; g.textBaseline = "middle";
        yt.forEach(function (t) { g.fillText(fy(t), box.x - 6, ax.Y(t)); });
        if (o.xlabel) {
          g.textAlign = "center"; g.textBaseline = "bottom";
          g.fillText(o.xlabel, box.x + box.w / 2, c.h - 2);
        }
        if (o.ylabel) {
          g.save();
          g.translate(11, box.y + box.h / 2);
          g.rotate(-Math.PI / 2);
          g.textAlign = "center"; g.textBaseline = "middle";
          g.fillText(o.ylabel, 0, 0);
          g.restore();
        }
        g.restore();
      }
    };
    return ax;
  }

  function niceTicks(a, b, n) {
    var span = b - a;
    if (!(span > 0)) return [a];
    var step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var err = n / span * step;
    if (err <= .15) step *= 10; else if (err <= .35) step *= 5; else if (err <= .75) step *= 2;
    var out = [];
    for (var t = Math.ceil(a / step) * step; t <= b + step * 1e-9; t += step) out.push(+t.toFixed(10));
    return out;
  }

  function fmtTick(t) {
    var a = Math.abs(t);
    if (a === 0) return "0";
    if (a >= 1e4 || a < 1e-3) return t.toExponential(0);
    return String(+t.toPrecision(3));
  }

  // Format a number for a readout: fixed decimals, minus sign as U+2212.
  function fmt(v, d) {
    if (!isFinite(v)) return "—";
    var s = v.toFixed(d == null ? 2 : d);
    return s.replace("-", "−");
  }

  /* ---------- animation ------------------------------------------------- */

  var reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // requestAnimationFrame loop that pauses while the figure is off screen.
  // fn(dt) gets seconds since the last frame (capped at 0.05).
  function loop(stage, fn) {
    var running = false, visible = true, raf = 0, last = 0;
    function frame(t) {
      raf = 0;
      if (!running || !visible) return;
      var dt = last ? Math.min((t - last) / 1000, 0.05) : 0.016;
      last = t;
      if (fn(dt) === false) { running = false; return; }
      raf = requestAnimationFrame(frame);
    }
    function kick() { if (running && visible && !raf) { last = 0; raf = requestAnimationFrame(frame); } }
    if (window.IntersectionObserver) {
      new IntersectionObserver(function (es) {
        visible = es[0].isIntersecting;
        kick();
      }).observe(stage);
    }
    return {
      start: function () { running = true; kick(); },
      stop: function () { running = false; },
      get running() { return running; }
    };
  }

  /* ---------- theme ----------------------------------------------------- */

  var themeListeners = [];
  function onTheme(fn) { themeListeners.push(fn); }
  function fireTheme() {
    // let the new custom properties land before figures read them
    requestAnimationFrame(function () { themeListeners.forEach(function (f) { f(); }); });
  }
  new MutationObserver(fireTheme).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", fireTheme);
  }

  /* ---------- registry and mounting -------------------------------------- */

  var ctx = {
    colors: colors, alpha: alpha, rng: rng, el: el, controls: controls, slider: slider,
    toggle: toggle, button: button, readout: readout, canvas: canvas, axes: axes,
    niceTicks: niceTicks, fmt: fmt, loop: loop, onTheme: onTheme, reducedMotion: reducedMotion
  };

  function mount(stage) {
    if (stage.dataset.mounted) return;
    var id = stage.getAttribute("data-fig");
    var spec = specs[id];
    if (!spec) return;
    stage.dataset.mounted = "1";
    try {
      var inst = spec.mount(stage, ctx) || {};
      if (inst.draw) onTheme(inst.draw);
    } catch (err) {
      stage.textContent = "This figure failed to load.";
      if (window.console) console.error("figure " + id, err);
    }
  }

  function scan() {
    var stages = document.querySelectorAll("[data-fig]");
    if (!window.IntersectionObserver) { stages.forEach(mount); return; }
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); mount(e.target); } });
    }, { rootMargin: "300px 0px" });
    stages.forEach(function (s) { if (specs[s.getAttribute("data-fig")]) io.observe(s); });
  }

  var scanned = false;
  window.FIG = {
    ctx: ctx,
    specs: specs,
    register: function (id, spec) {
      specs[id] = spec;
      if (scanned) document.querySelectorAll('[data-fig="' + id + '"]').forEach(mount);
    }
  };

  function ready() { scanned = true; scan(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready);
  else ready();
})();
