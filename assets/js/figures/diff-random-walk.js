FIG.register("diff-random-walk", {
  compute: {
    // D is held fixed, so each step size dx comes with its own dt = dx^2 / (2D)
    defaults: { D: 0.5, dx: 0.2, t: 1, N: 4000, seed: 5, tMax: 2 },
    steps: [0.4, 0.2, 0.1, 0.05],

    dtFor: function (dx, D) { return dx * dx / (2 * D); },

    // walkers live on the lattice x = k dx and all start at k = 0
    init: function (N, seed) { return { k: new Int32Array(N), n: 0, rng: FIG.ctx.rng(seed) }; },

    // one fair coin per walker per step; 32 coins come from each 32-bit draw.
    // Step-major order, so the state after n steps does not depend on how they were batched.
    step: function (st, m) {
      var k = st.k, N = k.length, r = st.rng, bits = 0;
      for (var s = 0; s < m; s++) {
        for (var i = 0; i < N; i++) {
          if ((i & 31) === 0) bits = (r.uniform() * 4294967296) >>> 0;
          k[i] += ((bits >>> (i & 31)) & 1) ? 1 : -1;
        }
        st.n++;
      }
    },

    stats: function (st, dx) {
      var k = st.k, N = k.length, s1 = 0, s2 = 0;
      for (var i = 0; i < N; i++) { s1 += k[i]; s2 += k[i] * k[i]; }
      var mean = s1 / N;
      return { mean: mean * dx, variance: (s2 - N * mean * mean) / (N - 1) * dx * dx };
    },

    // Bins line up with the lattice: after n steps only sites k with the parity of n
    // are reachable, so each bin holds m of them and is 2 m dx wide.
    histogram: function (st, dx) {
      var m = Math.max(1, Math.round(0.2 / (2 * dx))), w = 2 * m * dx, par = st.n & 1;
      var k = st.k, N = k.length, counts = {}, lo = Infinity, hi = -Infinity;
      for (var i = 0; i < N; i++) {
        var j = Math.floor((k[i] - par) / (2 * m));
        counts[j] = (counts[j] || 0) + 1;
        if (j < lo) lo = j;
        if (j > hi) hi = j;
      }
      var bins = [];
      for (var b = lo; b <= hi; b++) {
        var left = (par + 2 * m * b - 1) * dx;
        bins.push({ x0: left, x1: left + w, density: (counts[b] || 0) / (N * w) });
      }
      return { w: w, bins: bins };
    },

    // walk to the step nearest time t and report what the caption quotes
    run: function (p) {
      var dt = this.dtFor(p.dx, p.D), n = Math.round(p.t / dt);
      var st = this.init(p.N, p.seed);
      this.step(st, n);
      var s = this.stats(st, p.dx);
      return { dx: p.dx, dt: dt, n: n, t: n * dt, mean: s.mean, variance: s.variance, theory: 2 * p.D * n * dt };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, P = C.defaults;
    var dx = P.dx, st = null, clock = 0;

    var row = ctx.controls(stage);
    var playBtn = ctx.button(row, "Play", function () {
      if (anim.running) { anim.stop(); playBtn.textContent = "Play"; return; }
      if (clock >= P.tMax - 1e-9) { clock = 0; reset(0); }
      anim.start();
      playBtn.textContent = "Pause";
    });
    playBtn.classList.add("fig__btn--primary");
    ctx.button(row, "One step", function () {
      anim.stop(); playBtn.textContent = "Play";
      C.step(st, 1); clock = st.n * C.dtFor(dx, P.D); draw();
    });
    ctx.button(row, "Reset", function () {
      anim.stop(); playBtn.textContent = "Play";
      clock = 0; reset(0); draw();
    });
    ctx.toggle(row, {
      label: "step size dx", value: dx,
      options: C.steps.map(function (v) { return { value: v, label: String(v) }; }),
      // the clock keeps the time asked for, so passing through a coarse dx does not shift it
      onchange: function (v) {
        dx = v;
        reset(clock);
        draw();
      }
    });

    var cv = ctx.canvas(stage, { aspect: 0.5, minHeight: 240, maxHeight: 380,
      label: "Histogram of 4000 random walkers against the Gaussian solution of the heat equation" });
    var lg = ctx.el("div", "fig__legend", stage);
    function key(label) { var s = ctx.el("span", null, lg); var i = ctx.el("i", null, s); ctx.el("span", null, s, label); return i; }
    var kHist = key("histogram of 4000 walkers"), kHeat = key("heat equation: Gaussian with variance 2Dt");
    var ro = ctx.readout(stage);

    // re-walk from the origin with the current dx up to the step nearest time t
    function reset(t) {
      st = C.init(P.N, P.seed);
      C.step(st, Math.round(t / C.dtFor(dx, P.D)));
    }

    var anim = ctx.loop(stage, function (dtSec) {
      var dt = C.dtFor(dx, P.D);
      clock = Math.min(P.tMax, clock + 0.45 * dtSec);
      var target = Math.floor(clock / dt + 1e-9);
      if (target > st.n) C.step(st, target - st.n);
      draw();
      if (clock >= P.tMax) { playBtn.textContent = "Play"; return false; }
    });

    function draw() {
      var col = ctx.colors(), g = cv.g, D = P.D, dt = C.dtFor(dx, D);
      var t = st.n * dt, varT = 2 * D * t;
      var h = C.histogram(st, dx), s = C.stats(st, dx);
      var peak = t > 0 ? 1 / Math.sqrt(2 * Math.PI * varT) : 1 / h.w;
      var ymax = Math.max(0.45, 1.25 * Math.min(peak, 1 / h.w));
      cv.clear();
      var ax = ctx.axes(cv, { xmin: -4.5, xmax: 4.5, ymin: 0, ymax: ymax, xticks: [-4, -2, 0, 2, 4], xlabel: "position x", ylabel: "P(x, t)",
        margin: { l: 48, r: 12, t: 10, b: 34 } });
      ax.draw();
      g.save();
      g.beginPath(); g.rect(ax.box.x, ax.box.y, ax.box.w, ax.box.h); g.clip();
      g.fillStyle = ctx.alpha(col.c[0], 0.35);
      g.strokeStyle = ctx.alpha(col.c[0], 0.8);
      g.lineWidth = 1;
      h.bins.forEach(function (b) {
        if (!b.density) return;
        var x0 = ax.X(b.x0), x1 = ax.X(b.x1), y = ax.Y(Math.min(b.density, ymax * 1.02));
        g.fillRect(x0, y, x1 - x0, ax.Y(0) - y);
        g.beginPath(); g.moveTo(x0, ax.Y(0)); g.lineTo(x0, y); g.lineTo(x1, y); g.lineTo(x1, ax.Y(0)); g.stroke();
      });
      if (t > 0) {
        g.beginPath();
        for (var i = 0; i <= 360; i++) {
          var x = -4.5 + 9 * i / 360, y = Math.exp(-x * x / (2 * varT)) / Math.sqrt(2 * Math.PI * varT);
          if (i) g.lineTo(ax.X(x), ax.Y(y)); else g.moveTo(ax.X(x), ax.Y(y));
        }
        g.strokeStyle = col.c[1];
        g.lineWidth = 2.2;
        g.stroke();
      }
      g.restore();
      if (t === 0) {
        g.font = "12px " + getComputedStyle(document.documentElement).getPropertyValue("--sans");
        g.fillStyle = col.muted;
        g.textAlign = "left"; g.textBaseline = "top";
        g.fillText("t = 0: every walker at x = 0", ax.box.x + 8, ax.box.y + 8);
      }
      kHist.style.background = ctx.alpha(col.c[0], 0.6);
      kHeat.style.background = col.c[1];
      ro.set({
        "time t": ctx.fmt(t, 2),
        "steps taken": String(st.n),
        "dt = dx²/(2D)": ctx.fmt(dt, 4),
        "walker variance": ctx.fmt(s.variance, 3),
        "2Dt": ctx.fmt(varT, 3)
      });
    }

    cv.onresize = draw;
    clock = ctx.reducedMotion ? P.tMax : P.t;
    reset(clock);
    draw();
    return { draw: draw };
  }
});
