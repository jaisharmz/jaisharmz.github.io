FIG.register("diff-equipartition", {
  // Units: m = 1 and k_B T = 1. kappa is the noise strength C in units of 2 gamma k_B T.
  // n particles start at rest; the stationary window is t in [T/2, T].
  defaults: { seed: 5, n: 2000, kappa: 1, gamma: 1, m: 1, kT: 1, dt: 0.02, T: 16, snapEvery: 5, maxLag: 4 },

  // Pure math, no DOM. The model is the text's m dv/dt = -gamma v + F_random(t), with
  // white noise <F_random(t) F_random(s)> = C delta(t - s), so m dv = -gamma v dt + sqrt(C) dW.
  compute: {
    // Exact one-step solution of the Ornstein-Uhlenbeck equation over dt:
    // v(t+dt) = v e^{-gamma dt/m} + sqrt(C/(2 gamma m) (1 - e^{-2 gamma dt/m})) z.
    // It is exact for any dt, so the plateau carries no discretization error.
    simulate: function (p, seed) {
      var r = FIG.ctx.rng(seed), n = p.n;
      var C = p.kappa * 2 * p.gamma * p.kT;
      var a = Math.exp(-p.gamma * p.dt / p.m);
      var sd = Math.sqrt(C / (2 * p.gamma * p.m) * (1 - a * a));
      var steps = Math.round(p.T / p.dt), every = p.snapEvery;
      var nsnap = Math.floor(steps / every) + 1;
      var v = new Float64Array(n), snap = new Float64Array(nsnap * n);
      var t = new Float64Array(steps + 1), v2 = new Float64Array(steps + 1);
      var i, s, k;
      for (s = 1; s <= steps; s++) {
        var acc = 0;
        for (i = 0; i < n; i++) {
          v[i] = a * v[i] + sd * r.normal();
          acc += v[i] * v[i];
        }
        t[s] = s * p.dt;
        v2[s] = acc / n;
        if (s % every === 0) snap.set(v, (s / every) * n);
      }
      // stationary <v^2>: average over particles and over t in [T/2, T]
      var s0 = Math.round(steps / 2), stat = 0;
      for (s = s0; s <= steps; s++) stat += v2[s];
      stat /= (steps - s0 + 1);
      // velocity autocorrelation <v(t0) v(t0 + tau)> / <v^2>, t0 in the stationary window
      var sdt = every * p.dt, nlag = Math.round(p.maxLag / sdt), i0 = Math.round(nsnap / 2);
      var tau = [], acf = [];
      for (k = 0; k <= nlag; k++) {
        var sum = 0, cnt = 0;
        for (var j = i0; j + k < nsnap; j++) {
          var b0 = j * n, b1 = (j + k) * n;
          for (i = 0; i < n; i++) sum += snap[b0 + i] * snap[b1 + i];
          cnt += n;
        }
        tau.push(k * sdt);
        acf.push(sum / cnt);
      }
      var c0 = acf[0];
      for (k = 0; k <= nlag; k++) acf[k] /= c0;
      return {
        t: t, v2: v2, stationary: stat, tau: tau, acf: acf, C: C,
        predicted: C / (2 * p.gamma * p.m),      // stationary <v^2> this noise gives
        target: p.kT / p.m,                      // equipartition: <v^2> = k_B T / m
        window: [s0 * p.dt, steps * p.dt]
      };
    },

    // <v^2>(t) from rest: the text's v(t) = (1/m) int e^{-gamma (t-s)/m} F_random(s) ds
    // gives (C / (2 gamma m)) (1 - e^{-2 gamma t / m}).
    theoryV2: function (p, t) {
      var C = p.kappa * 2 * p.gamma * p.kT;
      return C / (2 * p.gamma * p.m) * (1 - Math.exp(-2 * p.gamma * t / p.m));
    },

    // Interpolated autocorrelation at lag x (for the readout at tau = m / gamma).
    acfAt: function (res, x) {
      var d = res.tau[1] - res.tau[0], last = res.acf.length - 1;
      if (x > res.tau[last] + 1e-9) return NaN;
      var k = Math.min(Math.floor(x / d + 1e-9), last - 1), f = Math.min(1, x / d - k);
      return res.acf[k] * (1 - f) + res.acf[k + 1] * f;
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var p = {}; Object.keys(D).forEach(function (k) { p[k] = D[k]; });
    var res = null, pending = 0;

    var row = ctx.controls(stage);
    var sC = ctx.slider(row, {
      label: "noise strength C, in units of 2γk_BT", min: -2, max: 2, step: 0.05, value: 0,
      fmt: function (u) { return Math.pow(2, u).toFixed(2) + "×"; },
      oninput: function (u) { p.kappa = Math.pow(2, u); schedule(); }
    });
    var sG = ctx.slider(row, {
      label: "drag γ", min: 0.25, max: 4, step: 0.05, value: D.gamma,
      fmt: function (v) { return v.toFixed(2); },
      oninput: function (v) { p.gamma = v; schedule(); }
    });
    ctx.button(row, "Reset", function () {
      p.kappa = D.kappa; p.gamma = D.gamma; sC.set(0); sG.set(D.gamma); recompute();
    });

    var grid = ctx.el("div", "fig__row", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);

    // stack the two panels whenever the stage itself is narrow, not only the viewport
    function stack() { grid.style.gridTemplateColumns = stage.clientWidth < 520 ? "1fr" : ""; }
    stack();
    if (window.ResizeObserver) new ResizeObserver(stack).observe(stage);
    var cvL = ctx.canvas(left, { aspect: 0.72, minHeight: 220, maxHeight: 320,
      label: "Mean squared velocity of 2000 particles over time against k_B T over m" });
    var cvR = ctx.canvas(right, { aspect: 0.72, minHeight: 220, maxHeight: 320,
      label: "Velocity autocorrelation against exp(-gamma tau / m)" });
    var legend = ctx.el("div", "fig__legend", stage);
    var ro = ctx.readout(stage);
    ro.el.style.gridTemplateColumns = "repeat(auto-fill, minmax(15rem, 1fr))";
    // k_B as a real subscript wherever a label carries it
    function sub(el) { el.innerHTML = el.textContent.replace(/k_B/g, "k<sub>B</sub>"); }
    sub(sC.el.querySelector(".fig__slider-label"));

    function schedule() {
      if (pending) return;
      pending = requestAnimationFrame(function () { pending = 0; recompute(); });
    }
    function recompute() { res = C.simulate(p, D.seed); draw(); }

    function line(g, ax, xs, ys, n) {
      g.beginPath();
      for (var i = 0; i < n; i++) {
        var px = ax.X(xs(i)), py = ax.Y(ys(i));
        if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
      }
      g.stroke();
    }

    function draw() {
      if (!res) return;
      var col = ctx.colors();
      // ---- left: <v^2>(t) ----
      var g = cvL.g;
      cvL.clear();
      var top = Math.max(1.5, 1.2 * Math.max(res.predicted, res.target));
      var axL = ctx.axes(cvL, { xmin: 0, xmax: p.T, ymin: 0, ymax: top, xlabel: "time t", ylabel: "⟨v²⟩",
        margin: { l: 44, r: 10, t: 12, b: 34 } });
      g.save();
      g.fillStyle = ctx.alpha(col.ink, 0.05);
      g.fillRect(axL.X(res.window[0]), axL.box.y, axL.X(res.window[1]) - axL.X(res.window[0]), axL.box.h);
      g.restore();
      axL.draw();
      g.save();
      g.beginPath(); g.rect(axL.box.x, axL.box.y, axL.box.w, axL.box.h); g.clip();
      // equipartition target
      g.strokeStyle = col.c[1]; g.lineWidth = 2;
      g.beginPath(); g.moveTo(axL.box.x, axL.Y(res.target)); g.lineTo(axL.box.x + axL.box.w, axL.Y(res.target)); g.stroke();
      // measured
      g.strokeStyle = col.c[0]; g.lineWidth = 1.5;
      var stride = 2;
      line(g, axL, function (i) { return res.t[i * stride]; }, function (i) { return res.v2[i * stride]; },
        Math.floor((res.t.length - 1) / stride) + 1);
      // theory from rest
      g.strokeStyle = col.ink; g.lineWidth = 1.25; g.setLineDash([5, 4]);
      line(g, axL, function (i) { return i / 200 * p.T; }, function (i) { return C.theoryV2(p, i / 200 * p.T); }, 201);
      g.setLineDash([]);
      g.restore();
      g.save();
      g.font = "11px " + sans; g.fillStyle = col.muted; g.textAlign = "right"; g.textBaseline = "top";
      g.fillText("averaged here", axL.X(res.window[1]) - 4, axL.box.y + 4);
      g.restore();

      // ---- right: velocity autocorrelation ----
      g = cvR.g;
      cvR.clear();
      var axR = ctx.axes(cvR, { xmin: 0, xmax: D.maxLag, ymin: -0.1, ymax: 1.05, xlabel: "lag τ", ylabel: "correlation",
        margin: { l: 44, r: 10, t: 12, b: 34 }, yticks: [0, 0.5, 1] });
      axR.draw();
      g.save();
      g.beginPath(); g.rect(axR.box.x, axR.box.y, axR.box.w, axR.box.h); g.clip();
      g.strokeStyle = col.ink; g.lineWidth = 1.25; g.setLineDash([5, 4]);
      line(g, axR, function (i) { return i / 200 * D.maxLag; }, function (i) { return Math.exp(-p.gamma * (i / 200 * D.maxLag) / p.m); }, 201);
      g.setLineDash([]);
      g.fillStyle = col.c[0];
      for (var k = 0; k < res.tau.length; k++) {
        g.beginPath(); g.arc(axR.X(res.tau[k]), axR.Y(res.acf[k]), 2.4, 0, 2 * Math.PI); g.fill();
      }
      // mark tau = m / gamma
      var tm = p.m / p.gamma;
      if (tm <= D.maxLag) {
        g.strokeStyle = col.muted; g.lineWidth = 1; g.setLineDash([2, 3]);
        g.beginPath(); g.moveTo(axR.X(tm), axR.box.y); g.lineTo(axR.X(tm), axR.box.y + axR.box.h); g.stroke();
        g.setLineDash([]);
        g.font = "11px " + sans; g.fillStyle = col.muted; g.textAlign = "left"; g.textBaseline = "top";
        g.fillText("τ = m/γ", axR.X(tm) + 4, axR.box.y + 4);
      }
      g.restore();

      var atTm = C.acfAt(res, tm);
      ro.set({
        "⟨v²⟩ measured, shaded": ctx.fmt(res.stationary, 3),
        "C/(2γm), this noise": ctx.fmt(res.predicted, 3),
        "k_BT/m, equipartition": ctx.fmt(res.target, 3),
        "correlation at τ = m/γ": isFinite(atTm) ? ctx.fmt(atTm, 3) + " (e⁻¹ = 0.368)" : "—"
      });
      if (!ro.el.dataset.sub) { ro.el.dataset.sub = "1"; ro.el.querySelectorAll("dt").forEach(sub); }
    }

    legend.innerHTML = "";
    function key(style, color, text) {
      var s = ctx.el("span", null, legend), i = ctx.el("i", null, s);
      if (style === "dot") { i.className = "dot"; i.style.background = color; }
      else if (style === "dash") { i.style.background = "none"; i.style.height = "0"; i.style.width = "1.2rem"; i.style.borderTop = "2px dashed " + color; }
      else i.style.background = color;
      var tx = ctx.el("span", null, s); tx.style.display = "inline"; tx.innerHTML = text;
    }
    key("line", "var(--fig-1)", "simulated, 2000 particles");
    key("dash", "var(--ink)", "formula from the text");
    key("line", "var(--fig-2)", "equipartition k<sub>B</sub>T/m");

    cvL.onresize = draw;
    cvR.onresize = draw;
    recompute();
    return { draw: draw };
  }
});
