FIG.register("cx-total-correlation", {
  defaults: { n: 5, r: 0.5, s: 0.05, days: 2000, seed: 13, scatter: 400 },

  // Pure math, no DOM. Notation follows the text: tokens X_1..X_n, H is (differential)
  // entropy in nats, and the metric is sum_i H(X_i) - H(X_1, ..., X_n), known as total
  // correlation. Returns are jointly Gaussian with standard deviation s and common pairwise
  // correlation r, so the correlation matrix is R = (1 - r) I + r 1 1^T and the metric is
  // -1/2 log det R = -1/2 [(n - 1) log(1 - r) + log(1 + (n - 1) r)].
  compute: {
    rmin: function (n) { return -1 / (n - 1); },

    tc: function (n, r) { return -0.5 * ((n - 1) * Math.log(1 - r) + Math.log(1 + (n - 1) * r)); },

    entropies: function (n, r, s) {
      var h = 0.5 * Math.log(2 * Math.PI * Math.E * s * s);
      var tc = this.tc(n, r);
      return { marginal: h, sumMarginal: n * h, joint: n * h - tc, tc: tc };
    },

    // log det of a symmetric positive definite matrix (array of rows) by Cholesky.
    logdet: function (A) {
      var n = A.length, L = [], i, j, k, s = 0;
      for (i = 0; i < n; i++) L.push(new Float64Array(n));
      for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
          var v = A[i][j];
          for (k = 0; k < j; k++) v -= L[i][k] * L[j][k];
          if (i === j) { if (v <= 0) return -Infinity; L[i][i] = Math.sqrt(v); s += Math.log(L[i][i]); }
          else L[i][j] = v / L[j][j];
        }
      }
      return 2 * s;
    },

    // Simulated daily returns: X = s R^{1/2} Z with
    // R^{1/2} = sqrt(1 - r) (I - 11^T/n) + sqrt(1 + (n - 1) r) 11^T/n, valid for r in (-1/(n-1), 1).
    // Returns the first two tokens for the scatter and the Gaussian plug-in estimate
    // -1/2 log det of the sample correlation matrix.
    simulate: function (n, r, s, days, seed, keep) {
      var g = FIG.ctx.rng(seed), a = Math.sqrt(1 - r), b = Math.sqrt(1 + (n - 1) * r);
      var sum = new Float64Array(n), cross = [], i, j, d;
      for (i = 0; i < n; i++) cross.push(new Float64Array(n));
      var x = new Float64Array(n), z = new Float64Array(n), pts = [];
      for (d = 0; d < days; d++) {
        var zbar = 0;
        for (i = 0; i < n; i++) { z[i] = g.normal(); zbar += z[i]; }
        zbar /= n;
        for (i = 0; i < n; i++) { x[i] = s * (a * (z[i] - zbar) + b * zbar); sum[i] += x[i]; }
        for (i = 0; i < n; i++) for (j = 0; j <= i; j++) cross[i][j] += x[i] * x[j];
        if (d < keep) pts.push([x[0], x[1]]);
      }
      var cov = [];
      for (i = 0; i < n; i++) {
        cov.push(new Float64Array(n));
        for (j = 0; j <= i; j++) {
          var c = (cross[i][j] - sum[i] * sum[j] / days) / (days - 1);
          cov[i][j] = c;
        }
      }
      var corr = [];
      for (i = 0; i < n; i++) {
        corr.push(new Float64Array(n));
        for (j = 0; j < n; j++) {
          var cij = i >= j ? cov[i][j] : cov[j][i];
          corr[i][j] = cij / Math.sqrt(cov[i][i] * cov[j][j]);
        }
      }
      return { pts: pts, tcHat: -0.5 * this.logdet(corr), r12: corr[1][0] };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var n = D.n, r = D.r, sim = null, key = "", pending = false;

    function lowR(nn) { return Math.max(-0.99, Math.ceil((C.rmin(nn) + 0.002) * 1000) / 1000); }

    var row = ctx.controls(stage);
    var sR = ctx.slider(row, { label: "pairwise correlation r", min: lowR(D.n), max: 0.99, step: 0.001, value: D.r,
      fmt: function (v) { return ctx.fmt(v, 3); }, oninput: function (v) { r = v; schedule(); } });
    ctx.slider(row, { label: "number of tokens n", min: 2, max: 20, step: 1, value: D.n,
      fmt: function (v) { return String(v); },
      oninput: function (v) {
        n = v;
        var lo = lowR(n);
        sR.input.min = lo;
        if (r < lo) { r = lo; }
        sR.set(r);
        schedule();
      } });

    var grid = ctx.el("div", "fig__row fig__row--3-2", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);
    // grid items default to min-width: auto, which would pin each column to its canvas width
    left.style.minWidth = "0"; right.style.minWidth = "0";
    var cvM = ctx.canvas(left, { aspect: 0.7, minHeight: 240, maxHeight: 340,
      label: "Total correlation against the pairwise correlation r for n tokens" });
    var cvS = ctx.canvas(right, { aspect: 1, minHeight: 220, maxHeight: 300,
      label: "Scatter of simulated daily returns of two tokens at correlation r" });
    var ro = ctx.readout(stage);
    cvM.onresize = draw; cvS.onresize = draw;

    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; draw(); });
    }

    function drawMetric(col) {
      var g = cvM.g;
      cvM.clear();
      var lo = lowR(n), rm = C.rmin(n);
      var ymax = Math.max(C.tc(n, 0.9), C.tc(n, Math.max(lo, -0.9)), 1.2 * C.tc(n, r), 0.5);
      var ax = ctx.axes(cvM, { xmin: -1, xmax: 1, ymin: 0, ymax: ymax, xlabel: "pairwise correlation r", ylabel: "nats",
        xticks: [-1, -0.5, 0, 0.5, 1], margin: { l: 44, r: 12, t: 10, b: 34 } });
      ax.draw();
      var b = ax.box;
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      // invalid region: no n x n correlation matrix has common correlation below -1/(n-1)
      if (rm > -1) {
        g.fillStyle = ctx.alpha(col.muted, 0.13);
        g.fillRect(b.x, b.y, ax.X(rm) - b.x, b.h);
      }
      g.strokeStyle = col.c[0]; g.lineWidth = 2.2;
      g.beginPath();
      var N = 300, started = false;
      for (var i = 0; i <= N; i++) {
        var x = lo + (0.99 - lo) * i / N, y = C.tc(n, x);
        if (!started) { g.moveTo(ax.X(x), ax.Y(y)); started = true; } else g.lineTo(ax.X(x), ax.Y(y));
      }
      g.stroke();
      // independent tokens
      g.fillStyle = col.muted; g.beginPath(); g.arc(ax.X(0), ax.Y(0), 3, 0, 2 * Math.PI); g.fill();
      // current point
      var px = ax.X(r), py = ax.Y(C.tc(n, r));
      g.fillStyle = col.c[1];
      g.beginPath(); g.arc(px, py, 5, 0, 2 * Math.PI); g.fill();
      g.strokeStyle = col.paper; g.lineWidth = 1.5; g.stroke();
      g.font = "11px " + sans; g.fillStyle = col.ink; g.textBaseline = "bottom";
      var lab = ctx.fmt(C.tc(n, r), 3) + " nats", tw = g.measureText(lab).width;
      // put the label on the side away from the rising branch
      var onLeft = r < 0 || px + 8 + tw > b.x + b.w - 2;
      g.textAlign = onLeft ? "right" : "left";
      var lx = onLeft ? px - 8 : px + 8, ly = py - 4;
      if (onLeft && lx - tw < b.x + 2) { g.textAlign = "left"; lx = px + 8; ly = py - 14; }
      if (ly - 12 < b.y) ly = py + 16;
      g.fillText(lab, lx, ly);
      if (rm > -1 && ax.X(rm) - b.x > 70) {
        g.fillStyle = col.muted; g.textAlign = "left"; g.textBaseline = "top";
        g.fillText("r < −1/(n−1)", b.x + 5, b.y + 5);
        g.fillText("not possible", b.x + 5, b.y + 19);
      }
      g.restore();
    }

    function drawScatter(col) {
      var g = cvS.g, lim = 4 * D.s;
      cvS.clear();
      var side = Math.min(cvS.w, cvS.h);
      var ax = ctx.axes(cvS, { xmin: -lim, xmax: lim, ymin: -lim, ymax: lim, xlabel: "token 1 daily return", ylabel: "token 2 daily return",
        xticks: [-0.1, 0, 0.1], yticks: [-0.1, 0, 0.1],
        box: { x: 44 + Math.max(0, (cvS.w - side) / 2), y: 10, w: side - 56, h: side - 46 } });
      ax.draw();
      var b = ax.box;
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      g.fillStyle = ctx.alpha(col.c[0], 0.5);
      sim.pts.forEach(function (q) { g.beginPath(); g.arc(ax.X(q[0]), ax.Y(q[1]), 1.9, 0, 2 * Math.PI); g.fill(); });
      g.restore();
    }

    function draw() {
      var k = n + "|" + r;
      if (k !== key) { sim = C.simulate(n, r, D.s, D.days, D.seed, D.scatter); key = k; }
      var col = ctx.colors();
      drawMetric(col);
      drawScatter(col);
      var e = C.entropies(n, r, D.s);
      ro.set({
        "metric, nats": ctx.fmt(e.tc, 4),
        "Σ H(Xᵢ), nats": ctx.fmt(e.sumMarginal, 4),
        "H(X₁, …, Xₙ), nats": ctx.fmt(e.joint, 4),
        "estimate, 2000 days": ctx.fmt(sim.tcHat, 4),
        "sample r₁₂": ctx.fmt(sim.r12, 3)
      });
    }

    draw();
    return { draw: draw };
  }
});
