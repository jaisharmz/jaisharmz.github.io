FIG.register("diff-predictor-corrector", {
  // N predictor steps from t = 1 to t = tEnd, K corrector steps after each one, each
  // corrector step size c times the variance of one mixture component of p_t.
  defaults: { seed: 11, n: 2000, N: 10, K: 1, c: 0.5, tEnd: 1e-3, xmin: -2.2, xmax: 2.4, bins: 69,
    Ns: [3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200] },

  // Pure math, no DOM. Forward process: the VP SDE dx = f(x,t) dt + g(t) dw with
  // f = -beta(t) x / 2, g = sqrt(beta(t)), beta(t) = 0.1 + t (20 - 0.1).
  // Data: P(x) = 0.7 N(-1, 0.15^2) + 0.3 N(1.2, 0.15^2). Its marginal at time t is
  // sum_k w_k N(alpha mu_k, alpha^2 s_k^2 + 1 - alpha^2), so the exact score
  // grad log P(x, t) is available and stands in for s_theta(x, t).
  compute: {
    sde: { bmin: 0.1, bmax: 20 },
    data: { w: [0.7, 0.3], mu: [-1, 1.2], s: [0.15, 0.15] },

    beta: function (t) { return this.sde.bmin + t * (this.sde.bmax - this.sde.bmin); },
    // alpha(t) = exp(-1/2 int_0^t beta)
    alpha: function (t) { return Math.exp(-0.5 * (this.sde.bmin * t + 0.5 * (this.sde.bmax - this.sde.bmin) * t * t)); },

    // Means, variances and log weights of the two components of p_t.
    level: function (t) {
      var a = this.alpha(t), a2 = a * a, d = this.data;
      var v0 = a2 * d.s[0] * d.s[0] + 1 - a2, v1 = a2 * d.s[1] * d.s[1] + 1 - a2;
      return { m0: a * d.mu[0], m1: a * d.mu[1], v0: v0, v1: v1,
        lw0: Math.log(d.w[0]) - 0.5 * Math.log(v0), lw1: Math.log(d.w[1]) - 0.5 * Math.log(v1),
        vmin: Math.min(v0, v1) };
    },

    // grad_x log p_t(x) for the level L, written with one exp and no overflow
    score: function (x, L) {
      var d0 = x - L.m0, d1 = x - L.m1;
      var r0 = 1 / (1 + Math.exp((L.lw1 - 0.5 * d1 * d1 / L.v1) - (L.lw0 - 0.5 * d0 * d0 / L.v0)));
      return -(r0 * d0 / L.v0 + (1 - r0) * d1 / L.v1);
    },

    pdf: function (x, t) {
      var L = this.level(t), d = this.data;
      return d.w[0] * Math.exp(-0.5 * (x - L.m0) * (x - L.m0) / L.v0) / Math.sqrt(2 * Math.PI * L.v0) +
             d.w[1] * Math.exp(-0.5 * (x - L.m1) * (x - L.m1) / L.v1) / Math.sqrt(2 * Math.PI * L.v1);
    },

    erf: function (x) {
      var s = x < 0 ? -1 : 1; x = Math.abs(x);
      var t = 1 / (1 + 0.3275911 * x);
      return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
    },

    cdf: function (x, L) {
      var d = this.data;
      return d.w[0] * 0.5 * (1 + this.erf((x - L.m0) / Math.sqrt(2 * L.v0))) +
             d.w[1] * 0.5 * (1 + this.erf((x - L.m1) / Math.sqrt(2 * L.v1)));
    },

    // Predictor-corrector sampler. Start from x ~ N(0, 1) at t = 1. Predictor (the text's
    // reverse SDE, Euler-Maruyama): Delta x = [f - g^2 s] dt + g sqrt(dt) z, x <- x - Delta x.
    // Corrector at the new t, K times: x <- x + eps s(x, t) + sqrt(2 eps) z, with
    // eps = c * (alpha^2 s_k^2 + 1 - alpha^2), a fixed fraction of one component's variance.
    sample: function (p, seed) {
      var r = FIG.ctx.rng(seed), n = p.n, x = new Float64Array(n), i, j, k;
      for (j = 0; j < n; j++) x[j] = r.normal();
      var dt = (1 - p.tEnd) / p.N;
      for (i = 0; i < p.N; i++) {
        var t = 1 - i * dt, b = this.beta(t), L = this.level(t), sq = Math.sqrt(b * dt);
        for (j = 0; j < n; j++) {
          var dx = (-0.5 * b * x[j] - b * this.score(x[j], L)) * dt + sq * r.normal();
          x[j] = x[j] - dx;
        }
        if (p.K > 0) {
          var t2 = 1 - (i + 1) * dt, L2 = this.level(t2), eps = p.c * L2.vmin, se = Math.sqrt(2 * eps);
          for (k = 0; k < p.K; k++) {
            for (j = 0; j < n; j++) x[j] = x[j] + eps * this.score(x[j], L2) + se * r.normal();
          }
        }
      }
      return x;
    },

    // 1-Wasserstein distance between the samples and p at tEnd: int |F_n(x) - F(x)| dx.
    w1: function (xs, t) {
      var s = Float64Array.from(xs).sort(), n = s.length, L = this.level(t);
      var lo = Math.min(s[0], -4), hi = Math.max(s[n - 1], 4), M = 6000, h = (hi - lo) / M, j = 0, acc = 0;
      for (var i = 0; i < M; i++) {
        var x = lo + (i + 0.5) * h;
        while (j < n && s[j] <= x) j++;
        acc += Math.abs(j / n - this.cdf(x, L)) * h;
      }
      return acc;
    },

    // What n exact draws would score on average: E|F_n - F| ~ sqrt(2 F (1 - F) / (pi n)).
    w1Floor: function (n, t) {
      var L = this.level(t), M = 4000, lo = -4, hi = 4, h = (hi - lo) / M, acc = 0;
      for (var i = 0; i < M; i++) {
        var F = this.cdf(lo + (i + 0.5) * h, L);
        acc += Math.sqrt(Math.max(0, F * (1 - F))) * h;
      }
      return Math.sqrt(2 / (Math.PI * n)) * acc;
    },

    // Point between the modes where the two weighted components of p_tEnd are equal.
    boundary: function (t) {
      var L = this.level(t), lo = L.m0, hi = L.m1;
      for (var it = 0; it < 60; it++) {
        var mid = (lo + hi) / 2, d0 = mid - L.m0, d1 = mid - L.m1;
        var g = (L.lw0 - 0.5 * d0 * d0 / L.v0) - (L.lw1 - 0.5 * d1 * d1 / L.v1);
        if (g > 0) lo = mid; else hi = mid;
      }
      return (lo + hi) / 2;
    },

    run: function (p, seed) {
      var x = this.sample(p, seed), b = this.boundary(p.tEnd), left = 0, out = 0;
      for (var j = 0; j < x.length; j++) {
        if (x[j] < b) left++;
        if (!(x[j] >= p.xmin && x[j] <= p.xmax)) out++;
      }
      var L = this.level(p.tEnd);
      return { x: x, w1: this.w1(x, p.tEnd), wLeft: left / x.length, wLeftTrue: this.cdf(b, L),
        outside: out / x.length, evals: p.N * (1 + p.K) };
    },

    // W1 for every N in Ns at a fixed K, same seed (the curve on the right).
    curve: function (p, K, seed) {
      var self = this;
      return p.Ns.map(function (N) {
        var q = {}; Object.keys(p).forEach(function (k) { q[k] = p[k]; });
        q.N = N; q.K = K;
        return self.w1(self.sample(q, seed), p.tEnd);
      });
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var p = {}; Object.keys(D).forEach(function (k) { p[k] = D[k]; });
    var curves = {}, res = null, xmode = "N", pending = 0;
    var floor = C.w1Floor(D.n, D.tEnd);
    var lnR = Math.log(200 / 3);
    function NofU(u) { return Math.round(3 * Math.exp(u * lnR)); }

    var row = ctx.controls(stage);
    var sN = ctx.slider(row, {
      label: "predictor steps N", min: 0, max: 1, step: 0.005, value: Math.log(D.N / 3) / lnR,
      fmt: function (u) { return String(NofU(u)); },
      oninput: function (u) { var N = NofU(u); if (N !== p.N) { p.N = N; schedule(); } }
    });
    var sK = ctx.slider(row, {
      label: "corrector steps K", min: 0, max: 5, step: 1, value: D.K,
      fmt: function (v) { return String(v); },
      oninput: function (v) { p.K = v; schedule(); }
    });
    var tg = ctx.toggle(row, {
      label: "x-axis",
      options: [{ value: "N", label: "predictor steps" }, { value: "E", label: "score evaluations" }],
      value: "N",
      onchange: function (v) { xmode = v; draw(); }
    });

    var grid = ctx.el("div", "fig__row fig__row--3-2", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);

    // stack the two panels whenever the stage itself is narrow, not only the viewport
    function stack() { grid.style.gridTemplateColumns = stage.clientWidth < 560 ? "1fr" : ""; }
    stack();
    if (window.ResizeObserver) new ResizeObserver(stack).observe(stage);
    var cvH = ctx.canvas(left, { aspect: 0.6, minHeight: 230, maxHeight: 340,
      label: "Histogram of 2000 samples against the true density" });
    var cvW = ctx.canvas(right, { aspect: 0.86, minHeight: 230, maxHeight: 340,
      label: "Wasserstein distance to the true density against the number of steps" });
    var legend = ctx.el("div", "fig__legend", stage);
    var ro = ctx.readout(stage);
    ro.el.style.gridTemplateColumns = "repeat(auto-fill, minmax(14rem, 1fr))";

    function schedule() {
      if (pending) return;
      pending = requestAnimationFrame(function () { pending = 0; recompute(); });
    }
    // the curve for a new K takes up to a few hundred ms, so it fills in after the histogram
    var curveTimer = 0;
    function recompute() {
      res = C.run(p, D.seed);
      if (!curves[0]) curves[0] = C.curve(p, 0, D.seed);
      draw();
      if (!curves[p.K] && !curveTimer) {
        curveTimer = setTimeout(function () {
          curveTimer = 0;
          if (!curves[p.K]) { curves[p.K] = C.curve(p, p.K, D.seed); draw(); }
        }, 30);
      }
    }

    function draw() {
      if (!res) return;
      var col = ctx.colors();
      drawHist(col);
      drawCurve(col);
      ro.set({
        "W₁ to the true density": ctx.fmt(res.w1, 3),
        "W₁ of 2000 exact draws": ctx.fmt(floor, 3),
        "heavier mode's weight": ctx.fmt(res.wLeft, 3) + " (true " + ctx.fmt(res.wLeftTrue, 3) + ")",
        "score evaluations": String(res.evals),
        "samples off the plot": (100 * res.outside).toFixed(1) + "%"
      });
    }

    function drawHist(col) {
      var g = cvH.g;
      cvH.clear();
      var ax = ctx.axes(cvH, { xmin: D.xmin, xmax: D.xmax, ymin: 0, ymax: 2.4, xlabel: "x", ylabel: "density",
        margin: { l: 40, r: 10, t: 12, b: 34 } });
      ax.draw();
      var nb = D.bins, w = (D.xmax - D.xmin) / nb, cnt = new Float64Array(nb);
      for (var j = 0; j < res.x.length; j++) {
        var b = Math.floor((res.x[j] - D.xmin) / w);
        if (b >= 0 && b < nb) cnt[b]++;
      }
      g.save();
      g.beginPath(); g.rect(ax.box.x, ax.box.y, ax.box.w, ax.box.h); g.clip();
      g.fillStyle = ctx.alpha(col.c[0], 0.5);
      for (b = 0; b < nb; b++) {
        var hgt = cnt[b] / (res.x.length * w);
        var x0 = ax.X(D.xmin + b * w), x1 = ax.X(D.xmin + (b + 1) * w);
        g.fillRect(x0 + 0.5, ax.Y(hgt), Math.max(1, x1 - x0 - 1), ax.Y(0) - ax.Y(hgt));
      }
      g.strokeStyle = col.ink; g.lineWidth = 1.6;
      g.beginPath();
      for (var i = 0; i <= 400; i++) {
        var x = D.xmin + i / 400 * (D.xmax - D.xmin), y = C.pdf(x, D.tEnd);
        if (i === 0) g.moveTo(ax.X(x), ax.Y(y)); else g.lineTo(ax.X(x), ax.Y(y));
      }
      g.stroke();
      g.restore();
      g.save();
      g.font = "11px " + sans; g.fillStyle = col.soft; g.textAlign = "right"; g.textBaseline = "top";
      g.fillText("N = " + p.N + ", K = " + p.K, ax.box.x + ax.box.w - 6, ax.box.y + 6);
      g.restore();
    }

    // W1 against N (or against score evaluations), log x and linear y
    var YMAX = 0.7;
    function drawCurve(col) {
      var g = cvW.g;
      cvW.clear();
      var lx = function (v) { return Math.log10(v); };
      var xmax = xmode === "N" ? 200 : 1200;
      var ax = ctx.axes(cvW, { xmin: lx(2.6), xmax: lx(xmax * 1.15), ymin: 0, ymax: YMAX,
        xlabel: xmode === "N" ? "predictor steps N" : "score evaluations N(1 + K)", ylabel: "W₁ to the true density",
        margin: { l: 44, r: 10, t: 14, b: 34 },
        xticks: [3, 10, 30, 100, 300, 1000].filter(function (v) { return v <= xmax; }).map(lx),
        yticks: [0, 0.2, 0.4, 0.6],
        xfmt: function (v) { return String(Math.round(Math.pow(10, v))); } });
      ax.draw();
      g.save();
      g.beginPath(); g.rect(ax.box.x, ax.box.y - 1, ax.box.w, ax.box.h + 1); g.clip();
      // what 2000 exact draws score
      g.strokeStyle = col.muted; g.lineWidth = 1; g.setLineDash([4, 3]);
      g.beginPath(); g.moveTo(ax.box.x, ax.Y(floor)); g.lineTo(ax.box.x + ax.box.w, ax.Y(floor)); g.stroke();
      g.setLineDash([]);
      var offscale = [];
      function plot(K, color, width) {
        var cu = curves[K];
        if (!cu) return;
        var f = xmode === "N" ? 1 : 1 + K;
        g.strokeStyle = color; g.lineWidth = width;
        g.beginPath();
        D.Ns.forEach(function (N, i) {
          var px = ax.X(lx(N * f)), py = ax.Y(Math.min(cu[i], YMAX * 1.5));
          if (cu[i] > YMAX) offscale.push([px, cu[i], color]);
          if (i === 0) g.moveTo(px, py); else g.lineTo(px, py);
        });
        g.stroke();
      }
      plot(0, col.soft, 1.5);
      if (p.K > 0) plot(p.K, col.c[0], 2);
      // the current run
      var fx = xmode === "N" ? p.N : res.evals, my = Math.min(res.w1, YMAX);
      if (res.w1 > YMAX) offscale.push([ax.X(lx(fx)), res.w1, p.K > 0 ? col.c[0] : col.soft]);
      g.fillStyle = p.K > 0 ? col.c[0] : col.soft;
      g.strokeStyle = col.paper; g.lineWidth = 1.5;
      g.beginPath(); g.arc(ax.X(lx(fx)), ax.Y(my), 4.5, 0, 2 * Math.PI); g.fill(); g.stroke();
      g.restore();
      g.save();
      g.font = "11px " + sans; g.textBaseline = "bottom";
      g.fillStyle = col.muted; g.textAlign = "left";
      g.fillText("2000 exact draws", ax.box.x + 4, ax.Y(floor) - 3);
      // values above the top of the axis, written at the top edge
      g.textAlign = "left"; g.textBaseline = "top";
      offscale.forEach(function (o, i) {
        g.fillStyle = o[2];
        g.fillText("↑ " + o[1].toFixed(2), o[0] + 5, ax.box.y + 3 + 13 * i);
      });
      g.restore();
    }

    legend.innerHTML = "";
    function key(style, color, text) {
      var s = ctx.el("span", null, legend), i = ctx.el("i", null, s);
      if (style === "box") { i.style.height = ".55rem"; i.style.background = color; }
      else if (style === "dash") { i.style.background = "none"; i.style.height = "0"; i.style.width = "1.2rem"; i.style.borderTop = "2px dashed " + color; }
      else i.style.background = color;
      s.appendChild(document.createTextNode(text));
    }
    key("box", "var(--fig-1)", "2000 samples");
    key("line", "var(--ink)", "true density");
    key("line", "var(--ink-soft)", "W₁ with K = 0");
    key("line", "var(--fig-1)", "W₁ with the chosen K");
    key("dash", "var(--muted)", "exact draws");

    cvH.onresize = draw;
    cvW.onresize = draw;
    recompute();
    return { draw: draw };
  }
});
