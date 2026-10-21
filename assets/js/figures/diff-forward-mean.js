FIG.register("diff-forward-mean", {
  compute: {
    defaults: { x0: 2.5, bmax: 20, nPaths: 150, nSteps: 1000, seed: 1 },

    // beta_min follows beta_max at the 1:200 ratio both schedules share:
    // 0.1 to 20 in continuous time (Song et al. 2021), 0.0001 to 0.02 per step.
    schedule: function (bmax) { return { bmin: bmax / 200, bmax: bmax }; },
    beta: function (t, s) { return s.bmin + t * (s.bmax - s.bmin); },
    intBeta: function (t, s) { return s.bmin * t + 0.5 * (s.bmax - s.bmin) * t * t; },
    alpha: function (t, s) { return Math.exp(-0.5 * this.intBeta(t, s)); },

    // one fixed table of N(0,1) draws, reused at every beta_max so paths morph smoothly
    noise: function (nPaths, nSteps, seed) {
      var r = FIG.ctx.rng(seed), z = new Float64Array(nPaths * nSteps);
      for (var i = 0; i < z.length; i++) z[i] = r.normal();
      return z;
    },

    // Euler-Maruyama on dx = -1/2 beta(t) x dt + sqrt(beta(t)) dw from a fixed x0
    simulate: function (p, z) {
      var n = p.nPaths, N = p.nSteps, dt = 1 / N, s = this.schedule(p.bmax);
      var paths = new Float32Array(n * (N + 1));
      var sum = 0, sum2 = 0;
      for (var i = 0; i < n; i++) {
        var x = p.x0, base = i * (N + 1);
        paths[base] = x;
        for (var k = 0; k < N; k++) {
          var b = this.beta(k * dt, s);
          x += -0.5 * b * x * dt + Math.sqrt(b * dt) * z[i * N + k];
          paths[base + k + 1] = x;
        }
        sum += x; sum2 += x * x;
      }
      var mean = sum / n, varS = (sum2 - n * mean * mean) / (n - 1);
      var a1 = this.alpha(1, s);
      return {
        paths: paths, bmin: s.bmin, bmax: s.bmax, intBeta1: this.intBeta(1, s), alpha1: a1,
        m1: p.x0 * a1, var1: 1 - a1 * a1, mean1: mean, var1Sample: varS
      };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, P = C.defaults;
    var z = C.noise(P.nPaths, P.nSteps, P.seed);
    var LO = Math.log10(0.02), HI = Math.log10(30);
    var res = null;

    var row = ctx.controls(stage);
    var sl = ctx.slider(row, {
      label: "largest noise rate β_max", min: LO, max: HI, step: 0.001, value: Math.log10(P.bmax),
      fmt: function (v) { var b = Math.pow(10, v); return b < 1 ? b.toFixed(3) : b.toFixed(1); },
      oninput: function () { run(); draw(); }
    });
    function preset(v) { sl.set(Math.log10(v)); run(); draw(); }
    ctx.button(row, "β_max = 0.02", function () { preset(0.02); });
    ctx.button(row, "β_max = 20", function () { preset(20); });

    var cv = ctx.canvas(stage, { aspect: 0.52, minHeight: 250, maxHeight: 400,
      label: "150 noising paths of x against t with the derived mean and a two standard deviation band" });
    var lg = ctx.el("div", "fig__legend", stage);
    function key(label) { var s = ctx.el("span", null, lg); var i = ctx.el("i", null, s); ctx.el("span", null, s, label); return i; }
    var kPath = key("150 sample paths"), kMean = key("m(t), derived"), kBand = key("m(t) ± 2√(1 − α(t)²)");
    var ro = ctx.readout(stage);

    function run() {
      var b = Math.pow(10, sl.value);
      if (Math.abs(sl.value - LO) < 1e-6) b = 0.02;
      res = C.simulate({ x0: P.x0, bmax: b, nPaths: P.nPaths, nSteps: P.nSteps }, z);
    }

    function draw() {
      var col = ctx.colors(), g = cv.g;
      cv.clear();
      var ax = ctx.axes(cv, { xmin: 0, xmax: 1, ymin: -3.5, ymax: 3.5, xlabel: "time t", ylabel: "x",
        margin: { l: 40, r: 12, t: 10, b: 34 } });
      ax.draw();
      var s = C.schedule(res.bmax), N = P.nSteps, n = P.nPaths, stride = 5;
      g.save();
      g.beginPath();
      g.rect(ax.box.x, ax.box.y, ax.box.w, ax.box.h);
      g.clip();
      var k, t;
      // sample paths
      g.beginPath();
      for (var i = 0; i < n; i++) {
        var base = i * (N + 1);
        g.moveTo(ax.X(0), ax.Y(res.paths[base]));
        for (k = stride; k <= N; k += stride) g.lineTo(ax.X(k / N), ax.Y(res.paths[base + k]));
      }
      g.strokeStyle = ctx.alpha(col.c[0], 0.2);
      g.lineWidth = 0.8;
      g.stroke();
      // band edges
      g.setLineDash([5, 4]);
      g.strokeStyle = col.c[1];
      g.lineWidth = 1.3;
      [2, -2].forEach(function (sgn) {
        g.beginPath();
        for (var k2 = 0; k2 <= 200; k2++) { var t2 = k2 / 200, a2 = C.alpha(t2, s), y = P.x0 * a2 + sgn * Math.sqrt(Math.max(0, 1 - a2 * a2)); if (k2) g.lineTo(ax.X(t2), ax.Y(y)); else g.moveTo(ax.X(t2), ax.Y(y)); }
        g.stroke();
      });
      g.setLineDash([]);
      // derived mean
      g.beginPath();
      for (k = 0; k <= 200; k++) { t = k / 200; if (k) g.lineTo(ax.X(t), ax.Y(P.x0 * C.alpha(t, s))); else g.moveTo(ax.X(t), ax.Y(P.x0)); }
      g.strokeStyle = col.c[1];
      g.lineWidth = 2.4;
      g.stroke();
      g.restore();

      kPath.style.background = col.c[0];
      kMean.style.background = col.c[1];
      kBand.style.background = ctx.alpha(col.c[1], 0.55);

      ro.set({
        "β_min": ctx.fmt(res.bmin, res.bmin < 0.01 ? 4 : 3),
        "∫₀¹ β(t) dt": ctx.fmt(res.intBeta1, res.intBeta1 < 1 ? 4 : 2),
        "m(1) formula": ctx.fmt(res.m1, 3),
        "m(1) paths": ctx.fmt(res.mean1, 3),
        "Var(1) formula": ctx.fmt(res.var1, 3),
        "Var(1) paths": ctx.fmt(res.var1Sample, 3)
      });
    }

    cv.onresize = draw;
    run();
    draw();
    return { draw: draw };
  }
});
