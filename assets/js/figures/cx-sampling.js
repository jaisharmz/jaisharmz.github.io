FIG.register("cx-sampling", {
  defaults: { mu: 0.5, sigma: 0.3, R0: 1, T: 1, M: 256, paths: 2000, seed: 5, log2n: 3, maxLog2n: 6 },

  // Pure math, no DOM. Notation follows the text: R(t) is utilization (revenue rate),
  // dR = mu R dt + sigma R dW, the provider bills sum_{i=1}^n R(t_i) dt with
  // t_i = (i - 1) dt (left endpoints) and n dt = T, and Loss = int_0^T R dt - that sum.
  compute: {
    // E[Loss] = R0 (e^{mu T} - 1) [1/mu - dt / (e^{mu dt} - 1)], the text's closed form.
    exactLoss: function (mu, R0, T, dt) {
      if (Math.abs(mu) < 1e-9) return 0;
      return R0 * Math.expm1(mu * T) * (1 / mu - dt / Math.expm1(mu * dt));
    },
    // First-order Taylor expansion: 1/2 R0 (e^{mu T} - 1) dt.
    firstOrder: function (mu, R0, T, dt) { return 0.5 * R0 * Math.expm1(mu * T) * dt; },

    // The two expectations the text subtracts, computed independently as a check.
    parts: function (mu, R0, T, n) {
      var dt = T / n, direct = 0;
      for (var i = 1; i <= n; i++) direct += R0 * Math.exp(mu * (i - 1) * dt) * dt;
      return {
        continuous: R0 / mu * Math.expm1(mu * T),
        discreteDirect: direct,
        discreteClosed: R0 * dt * Math.expm1(mu * T) / Math.expm1(mu * dt)
      };
    },

    // Monte Carlo: exact GBM steps on a fine grid of M steps, the integral by the
    // trapezoid rule on that grid, and the left-endpoint sum for every n = 2^j,
    // j = 0..maxLog2n, from the same paths (common random numbers).
    simulate: function (p, seed) {
      var r = FIG.ctx.rng(seed), M = p.M, J = p.maxLog2n + 1;
      var dtf = p.T / M, drift = (p.mu - 0.5 * p.sigma * p.sigma) * dtf, vol = p.sigma * Math.sqrt(dtf);
      var masks = [], dts = [], ns = [];
      for (var j = 0; j < J; j++) { var n = 1 << j; ns.push(n); dts.push(p.T / n); masks.push((M / n) - 1); }
      var sum = new Float64Array(J), sq = new Float64Array(J), neg = new Float64Array(J);
      var S = new Float64Array(J), path0 = new Float64Array(M + 1), loss0 = new Float64Array(J);
      for (var k = 0; k < p.paths; k++) {
        var R = p.R0, I = 0;
        for (j = 0; j < J; j++) S[j] = 0;
        if (k === 0) path0[0] = R;
        for (var m = 0; m < M; m++) {
          for (j = 0; j < J; j++) if ((m & masks[j]) === 0) S[j] += R * dts[j];
          var Rn = R * Math.exp(drift + vol * r.normal());
          I += 0.5 * (R + Rn) * dtf;
          R = Rn;
          if (k === 0) path0[m + 1] = R;
        }
        for (j = 0; j < J; j++) {
          var L = I - S[j];
          sum[j] += L; sq[j] += L * L; if (L < 0) neg[j]++;
          if (k === 0) loss0[j] = L;
        }
      }
      var mean = [], se = [], sd = [], negShare = [], exact = [], first = [];
      for (j = 0; j < J; j++) {
        var mu = sum[j] / p.paths, v = Math.max(0, sq[j] / p.paths - mu * mu) * p.paths / (p.paths - 1);
        mean.push(mu); sd.push(Math.sqrt(v)); se.push(Math.sqrt(v / p.paths)); negShare.push(neg[j] / p.paths);
        exact.push(this.exactLoss(p.mu, p.R0, p.T, dts[j]));
        first.push(this.firstOrder(p.mu, p.R0, p.T, dts[j]));
      }
      return { ns: ns, dts: dts, mean: mean, se: se, sd: sd, negShare: negShare, exact: exact, first: first,
        path0: path0, loss0: loss0 };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var p = { mu: D.mu, sigma: D.sigma, R0: D.R0, T: D.T, M: D.M, paths: D.paths, maxLog2n: D.maxLog2n };
    var j = D.log2n, sim = null, simKey = "", pending = false;

    var row = ctx.controls(stage);
    ctx.slider(row, { label: "samples per period n", min: 0, max: D.maxLog2n, step: 1, value: D.log2n,
      fmt: function (v) { return String(1 << v); },
      oninput: function (v) { j = v; schedule(); } });
    ctx.slider(row, { label: "drift μ", min: 0.1, max: 2, step: 0.05, value: D.mu,
      fmt: function (v) { return v.toFixed(2); }, oninput: function (v) { p.mu = v; schedule(); } });
    ctx.slider(row, { label: "volatility σ", min: 0, max: 1, step: 0.05, value: D.sigma,
      fmt: function (v) { return v.toFixed(2); }, oninput: function (v) { p.sigma = v; schedule(); } });

    var grid = ctx.el("div", "fig__row fig__row--3-2", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);
    // grid items default to min-width: auto, which would pin each column to its canvas width
    left.style.minWidth = "0"; right.style.minWidth = "0";
    var cvL = ctx.canvas(left, { aspect: 0.62, minHeight: 240, maxHeight: 360,
      label: "One utilization path R(t) with the left-endpoint billing rectangles and the revenue they miss" });
    var cvR = ctx.canvas(right, { aspect: 0.92, minHeight: 240, maxHeight: 360,
      label: "Expected loss against the sampling interval on doubling axes: exact formula, first-order line and Monte Carlo" });
    var legend = ctx.el("div", "fig__legend", stage);
    var ro = ctx.readout(stage);
    cvL.onresize = draw; cvR.onresize = draw;

    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; draw(); });
    }

    function frac(n) { return n === 1 ? "1" : "1/" + n; }

    function setLegend(col) {
      legend.innerHTML = "";
      [[col.ink, "utilization R(t)", ""], [col.c[0], "billed: R(tᵢ)Δt", ""], [col.c[3], "revenue missed", "dot"],
       [col.c[2], "over-billed", "dot"], [col.c[1], "exact E[Loss]", ""], [col.muted, "first order ½R₀(e^(μT) − 1)Δt", ""],
       [col.c[4], "Monte Carlo mean", "dot"]].forEach(function (it) {
        var s = ctx.el("span", null, legend);
        var i = ctx.el("i", it[2] || null, s);
        i.style.background = it[0];
        s.appendChild(document.createTextNode(it[1]));
      });
    }

    function drawPath(col) {
      var g = cvL.g, path = sim.path0, M = p.M, n = 1 << j, step = M / n;
      cvL.clear();
      var ymax = 0;
      for (var m = 0; m <= M; m++) ymax = Math.max(ymax, path[m]);
      ymax *= 1.08;
      var ax = ctx.axes(cvL, { xmin: 0, xmax: p.T, ymin: 0, ymax: ymax, xlabel: "time t (one billing period)", ylabel: "utilization R(t)",
        margin: { l: 44, r: 10, t: 10, b: 34 } });
      ax.draw();
      var b = ax.box;
      function stepAt(mm) { return path[Math.floor(mm / step) * step]; }
      // polygon between the curve and the billed step function
      function between() {
        g.beginPath();
        g.moveTo(ax.X(0), ax.Y(path[0]));
        for (var m = 1; m <= M; m++) g.lineTo(ax.X(m / M * p.T), ax.Y(path[m]));
        for (m = M; m >= 0; m--) {
          var h = stepAt(Math.min(m, M - 1));
          g.lineTo(ax.X(m / M * p.T), ax.Y(h));
          if (m % step === 0 && m > 0) g.lineTo(ax.X(m / M * p.T), ax.Y(stepAt(m - 1)));
        }
        g.closePath();
      }
      function stepOutline(close, toY) {
        g.beginPath();
        g.moveTo(ax.X(0), toY);
        for (var i = 0; i < n; i++) {
          var h = path[i * step];
          g.lineTo(ax.X(i * step / M * p.T), ax.Y(h));
          g.lineTo(ax.X((i + 1) * step / M * p.T), ax.Y(h));
        }
        g.lineTo(ax.X(p.T), toY);
        if (close) g.closePath();
      }
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      // billed rectangles
      g.fillStyle = ctx.alpha(col.c[0], 0.12);
      stepOutline(true, ax.Y(0)); g.fill();
      g.strokeStyle = ctx.alpha(col.c[0], 0.35); g.lineWidth = 1;
      for (var i = 1; i < n; i++) {
        var X = Math.round(ax.X(i / n * p.T)) + .5;
        g.beginPath(); g.moveTo(X, ax.Y(0)); g.lineTo(X, ax.Y(path[i * step])); g.stroke();
      }
      // revenue missed: area where the curve is above the step
      g.save(); stepOutline(true, b.y); g.clip();
      g.fillStyle = ctx.alpha(col.c[3], 0.45); between(); g.fill();
      g.restore();
      g.save(); stepOutline(true, b.y + b.h); g.clip();
      g.fillStyle = ctx.alpha(col.c[2], 0.45); between(); g.fill();
      g.restore();
      g.strokeStyle = col.c[0]; g.lineWidth = 1.6;
      stepOutline(false, ax.Y(0)); g.stroke();
      g.strokeStyle = col.ink; g.lineWidth = 1.5;
      g.beginPath();
      for (m = 0; m <= M; m++) { var x = ax.X(m / M * p.T), y = ax.Y(path[m]); if (m) g.lineTo(x, y); else g.moveTo(x, y); }
      g.stroke();
      // sample points
      g.fillStyle = col.c[0];
      if (n <= 32) for (i = 0; i < n; i++) { g.beginPath(); g.arc(ax.X(i / n * p.T), ax.Y(path[i * step]), 2.6, 0, 2 * Math.PI); g.fill(); }
      g.restore();
    }

    function drawLoss(col) {
      var g = cvR.g;
      cvR.clear();
      var J = p.maxLog2n + 1, lo = Infinity, hi = 0, q;
      for (q = 0; q < J; q++) {
        [sim.exact[q], sim.first[q], sim.mean[q]].forEach(function (v) { if (v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
      }
      var ymin = Math.log2(lo) - 0.6, ymax = Math.log2(hi) + 0.6;
      var yt = [];
      for (var d = Math.floor(Math.log10(lo)) - 1; d <= Math.ceil(Math.log10(hi)) + 1; d++) {
        [1, 2, 5].forEach(function (mm) { var v = mm * Math.pow(10, d), lv = Math.log2(v); if (lv >= ymin && lv <= ymax) yt.push(lv); });
      }
      if (yt.length > 6) yt = yt.filter(function (lv) { var v = Math.pow(2, lv); return Math.abs(v / Math.pow(10, Math.round(Math.log10(v))) - 1) < 1e-6; });
      var xt = [];
      for (q = -p.maxLog2n; q <= 0; q++) xt.push(q);
      var sparse = (cvR.w - 56) / (p.maxLog2n + 0.8) < 32;
      var ax = ctx.axes(cvR, { xmin: -p.maxLog2n - 0.4, xmax: 0.4, ymin: ymin, ymax: ymax, xticks: xt, yticks: yt,
        xfmt: function (t) { return (sparse && (-t) % 2 === 1) ? "" : frac(Math.round(Math.pow(2, -t))); },
        yfmt: function (t) { var v = Math.pow(2, t); return String(+v.toPrecision(2)); },
        xlabel: "sampling interval Δt", ylabel: "expected loss", margin: { l: 46, r: 10, t: 10, b: 34 } });
      ax.draw();
      var b = ax.box;
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      // first-order line and exact curve, sampled continuously in dt
      g.strokeStyle = col.muted; g.lineWidth = 1.3; g.setLineDash([5, 4]);
      g.beginPath();
      for (var s = 0; s <= 100; s++) {
        var lx = -p.maxLog2n - 0.4 + (p.maxLog2n + 0.8) * s / 100, dt = Math.pow(2, lx);
        var v = C.firstOrder(p.mu, p.R0, p.T, dt);
        if (s) g.lineTo(ax.X(lx), ax.Y(Math.log2(v))); else g.moveTo(ax.X(lx), ax.Y(Math.log2(v)));
      }
      g.stroke(); g.setLineDash([]);
      g.strokeStyle = col.c[1]; g.lineWidth = 2;
      g.beginPath();
      for (s = 0; s <= 100; s++) {
        lx = -p.maxLog2n - 0.4 + (p.maxLog2n + 0.8) * s / 100; dt = Math.pow(2, lx);
        v = C.exactLoss(p.mu, p.R0, p.T, dt);
        if (s) g.lineTo(ax.X(lx), ax.Y(Math.log2(v))); else g.moveTo(ax.X(lx), ax.Y(Math.log2(v)));
      }
      g.stroke();
      // Monte Carlo means with 2 standard errors
      for (q = 0; q < J; q++) {
        var x = ax.X(-q), m = sim.mean[q], e = 2 * sim.se[q];
        if (!(m > 0)) continue;
        g.strokeStyle = col.c[4]; g.lineWidth = 1;
        var top = m + e, bot = m - e;
        g.beginPath(); g.moveTo(x, ax.Y(Math.log2(top))); g.lineTo(x, bot > 0 ? ax.Y(Math.log2(bot)) : b.y + b.h); g.stroke();
        g.fillStyle = col.c[4];
        g.beginPath(); g.arc(x, ax.Y(Math.log2(m)), 3.2, 0, 2 * Math.PI); g.fill();
      }
      // current setting
      if (sim.mean[j] > 0 || sim.exact[j] > 0) {
        var cy = ax.Y(Math.log2(sim.exact[j]));
        g.strokeStyle = col.ink; g.lineWidth = 1.5;
        g.beginPath(); g.arc(ax.X(-j), cy, 7, 0, 2 * Math.PI); g.stroke();
      }
      g.restore();
    }

    function draw() {
      var key = p.mu + "|" + p.sigma;
      if (key !== simKey) { sim = C.simulate(p, D.seed); simKey = key; }
      var col = ctx.colors();
      setLegend(col);
      drawPath(col);
      drawLoss(col);
      var n = 1 << j, half = j < p.maxLog2n ? j + 1 : null;
      var exHalf = C.exactLoss(p.mu, p.R0, p.T, p.T / n / 2);
      var rows = {
        "Δt": frac(n),
        "exact E[Loss]": ctx.fmt(sim.exact[j], 5),
        "first order": ctx.fmt(sim.first[j], 5),
        "Monte Carlo mean": ctx.fmt(sim.mean[j], 5),
        "standard error": ctx.fmt(sim.se[j], 5),
        "halving Δt divides E[Loss] by": ctx.fmt(sim.exact[j] / exHalf, 3),
        "Monte Carlo ratio": half !== null ? ctx.fmt(sim.mean[j] / sim.mean[half], 3) : "n/a at n = 64",
        "sd of the loss across paths": ctx.fmt(sim.sd[j], 5),
        "paths over-billed": (sim.negShare[j] * 100).toFixed(1) + "%",
        "this path's loss": ctx.fmt(sim.loss0[j], 5)
      };
      ro.set(rows);
    }

    draw();
    return { draw: draw };
  }
});
