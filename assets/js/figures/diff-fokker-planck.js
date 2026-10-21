FIG.register("diff-fokker-planck", {
  compute: {
    // Drift f(x) = -x and D = 1: the forward SDE's drift -1/2 beta x and g^2 = 2D with beta
    // held at 2, so with both terms on the density settles to N(0, 1).
    // Grid: 560 cells on [-7, 7] with zero-flux walls; the particles reflect off the same walls.
    P: { lo: -7, hi: 7, M: 560, D: 1, frameDt: 0.0125, pSub: 5, tMax: 3, binCells: 8,
         init: [{ w: 0.4, mu: -2.5, s: 0.35 }, { w: 0.6, mu: 2, s: 0.35 }] },
    defaults: { N: 2000, seed: 9, drift: true, diff: true },
    f: function (x) { return -x; },

    initDensity: function (x) {
      var p = 0;
      this.P.init.forEach(function (c) { var u = (x - c.mu) / c.s; p += c.w * Math.exp(-0.5 * u * u) / (c.s * Math.sqrt(2 * Math.PI)); });
      return p;
    },

    // PDE state: cell densities plus scratch arrays for the scheme
    pdeInit: function () {
      var Q = this.P, M = Q.M, h = (Q.hi - Q.lo) / M, d = new Float64Array(M), mass = 0, i;
      for (i = 0; i < M; i++) { d[i] = this.initDensity(Q.lo + (i + 0.5) * h); mass += d[i] * h; }
      for (i = 0; i < M; i++) d[i] /= mass;
      return { d: d, h: h, t: 0, k1: new Float64Array(M), tmp: new Float64Array(M), slope: new Float64Array(M), F: new Float64Array(M + 1) };
    },

    // dP/dt = -(F_{i+1/2} - F_{i-1/2}) / h with flux F = f P - D dP/dx.
    // Advective part: MUSCL upwind with a van Leer limiter; diffusive part: central difference.
    // The two wall fluxes are zero, so the total mass sum_i P_i h only moves between cells.
    rhs: function (d, out, st, drift, D) {
      var M = d.length, h = st.h, lo = this.P.lo, s = st.slope, F = st.F, i;
      s[0] = 0; s[M - 1] = 0;
      for (i = 1; i < M - 1; i++) {
        var dl = d[i] - d[i - 1], dr = d[i + 1] - d[i];
        s[i] = dl * dr > 0 ? 2 * dl * dr / (dl + dr) : 0;
      }
      F[0] = 0; F[M] = 0;
      for (i = 0; i < M - 1; i++) {
        var a = drift ? this.f(lo + (i + 1) * h) : 0, adv = 0;
        if (a > 0) adv = a * (d[i] + 0.5 * s[i]);
        else if (a < 0) adv = a * (d[i + 1] - 0.5 * s[i + 1]);
        F[i + 1] = adv - D * (d[i + 1] - d[i]) / h;
      }
      for (i = 0; i < M; i++) out[i] = -(F[i + 1] - F[i]) / h;
    },

    // advance by T with SSP-RK2 steps small enough for both the advective and diffusive limits
    pdeAdvance: function (st, T, drift, diff) {
      var Q = this.P, D = diff ? Q.D : 0, h = st.h;
      if (!drift && !diff) { st.t += T; return; }
      var amax = drift ? Math.max(Math.abs(this.f(Q.lo)), Math.abs(this.f(Q.hi))) : 0;
      var dtMax = 0.9 / (2 * amax / h + 2 * D / (h * h));
      var n = Math.max(1, Math.ceil(T / dtMax)), dt = T / n, d = st.d, k1 = st.k1, tmp = st.tmp, M = d.length, i;
      for (var s = 0; s < n; s++) {
        this.rhs(d, k1, st, drift, D);
        for (i = 0; i < M; i++) tmp[i] = d[i] + dt * k1[i];
        this.rhs(tmp, k1, st, drift, D);
        for (i = 0; i < M; i++) d[i] = 0.5 * d[i] + 0.5 * (tmp[i] + dt * k1[i]);
      }
      st.t += T;
    },

    // particles: dx = f dt + sqrt(2D) dw, Euler-Maruyama, reflecting walls
    partInit: function (N, seed) {
      var Q = this.P, r = FIG.ctx.rng(seed), x = new Float64Array(N), jit = new Float32Array(N);
      for (var i = 0; i < N; i++) {
        var c = r.uniform() < Q.init[0].w ? Q.init[0] : Q.init[1];
        x[i] = this.reflect(c.mu + c.s * r.normal());
        jit[i] = r.uniform();
      }
      return { x: x, jit: jit, rng: FIG.ctx.rng(seed + 1), t: 0 };
    },
    reflect: function (x) {
      var lo = this.P.lo, hi = this.P.hi;
      if (x < lo) x = 2 * lo - x;
      if (x > hi) x = 2 * hi - x;
      return x;
    },
    partAdvance: function (st, T, drift, diff) {
      var Q = this.P, n = Q.pSub, dt = T / n, sd = diff ? Math.sqrt(2 * Q.D * dt) : 0, x = st.x, r = st.rng, N = x.length;
      if (drift || diff) {
        for (var s = 0; s < n; s++) {
          for (var i = 0; i < N; i++) {
            var v = x[i];
            if (drift) v += this.f(v) * dt;
            if (diff) v += sd * r.normal();
            x[i] = this.reflect(v);
          }
        }
      }
      st.t += T;
    },

    // L1 gap between the PDE and the particle histogram on bins of binCells cells
    compare: function (pde, part) {
      var Q = this.P, M = Q.M, h = pde.h, bc = Q.binCells, B = M / bc, w = bc * h, x = part.x, N = x.length, i, b;
      var counts = new Float64Array(B), pdeBins = new Float64Array(B), mass = 0, meanP = 0, meanX = 0;
      for (i = 0; i < M; i++) { pdeBins[Math.floor(i / bc)] += pde.d[i] / bc; mass += pde.d[i] * h; meanP += (Q.lo + (i + 0.5) * h) * pde.d[i] * h; }
      for (i = 0; i < N; i++) { b = Math.min(B - 1, Math.max(0, Math.floor((x[i] - Q.lo) / w))); counts[b]++; meanX += x[i]; }
      var L1 = 0, hist = new Float64Array(B);
      for (b = 0; b < B; b++) { hist[b] = counts[b] / (N * w); L1 += Math.abs(hist[b] - pdeBins[b]) * w; }
      return { L1: L1, mass: mass, meanPDE: meanP, meanParticles: meanX / N, hist: hist, binW: w };
    },

    // run from t = 0 with fixed switches and report at the requested times (for the figtest)
    run: function (opts, times) {
      var pde = this.pdeInit(), part = this.partInit(opts.N, opts.seed), out = [], frame = 0;
      var nFrames = Math.round(Math.max.apply(null, times) / this.P.frameDt);
      var want = times.map(function (t) { return Math.round(t / this.P.frameDt); }, this);
      if (want.indexOf(0) >= 0) out.push(Object.assign({ t: 0 }, this.compare(pde, part)));
      for (frame = 1; frame <= nFrames; frame++) {
        this.pdeAdvance(pde, this.P.frameDt, opts.drift, opts.diff);
        this.partAdvance(part, this.P.frameDt, opts.drift, opts.diff);
        if (want.indexOf(frame) >= 0) {
          var c = this.compare(pde, part), mx = 0;
          for (var i = 0; i < pde.d.length; i++) if (pde.d[i] > mx) mx = pde.d[i];
          out.push(Object.assign({ t: frame * this.P.frameDt, peak: mx, minDensity: Math.min.apply(null, pde.d) }, c));
        }
      }
      return out;
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, Q = C.P, Dflt = C.defaults;
    var drift = Dflt.drift, diff = Dflt.diff, N = Dflt.N, pde = null, part = null;
    var XMIN = -5, XMAX = 5, YMAX = 0.9;

    var row = ctx.controls(stage);
    var playBtn = ctx.button(row, "Play", function () {
      if (anim.running) { anim.stop(); playBtn.textContent = "Play"; return; }
      if (pde.t >= Q.tMax - 1e-9) reset();
      acc = 0;
      anim.start(); playBtn.textContent = "Pause";
    });
    playBtn.classList.add("fig__btn--primary");
    ctx.button(row, "Reset", function () { anim.stop(); playBtn.textContent = "Play"; reset(); draw(); });
    var onoff = [{ value: "on", label: "on" }, { value: "off", label: "off" }];
    ctx.toggle(row, { label: "drift f(x) = −x", value: "on", options: onoff, onchange: function (v) { drift = v === "on"; draw(); } });
    ctx.toggle(row, { label: "diffusion", value: "on", options: onoff, onchange: function (v) { diff = v === "on"; draw(); } });
    ctx.toggle(row, { label: "particles", value: N,
      options: [{ value: 500, label: "500" }, { value: 2000, label: "2000" }, { value: 8000, label: "8000" }],
      onchange: function (v) { N = v; anim.stop(); playBtn.textContent = "Play"; reset(); draw(); } });

    var cv = ctx.canvas(stage, { aspect: 0.48, minHeight: 280, maxHeight: 420,
      label: "Fokker-Planck density against a histogram of particles running the matching SDE" });
    var lg = ctx.el("div", "fig__legend", stage);
    function key(label, dot) { var s = ctx.el("span", null, lg); var i = ctx.el("i", dot ? "dot" : null, s); var t = ctx.el("span", null, s, label); return { i: i, t: t }; }
    var kD = key("particles, in the strip on top", true), kH = key("histogram of 2000 particles"), kP = key("Fokker-Planck PDE"), kS = key("steady state N(0, D)");
    var ro = ctx.readout(stage);

    function reset() { pde = C.pdeInit(); part = C.partInit(N, Dflt.seed); }

    // simulated time runs at 0.5 per second of real time, whatever the display's frame rate
    var acc = 0;
    var anim = ctx.loop(stage, function (dtSec) {
      acc += 0.5 * dtSec;
      while (acc >= Q.frameDt && pde.t < Q.tMax - 1e-9) {
        C.pdeAdvance(pde, Q.frameDt, drift, diff);
        C.partAdvance(part, Q.frameDt, drift, diff);
        acc -= Q.frameDt;
      }
      draw();
      if (pde.t >= Q.tMax - 1e-9) { playBtn.textContent = "Play"; return false; }
    });

    function draw() {
      var col = ctx.colors(), g = cv.g, W = cv.w, H = cv.h;
      var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
      var cmp = C.compare(pde, part), h = pde.h;
      cv.clear();
      var strip = { y: 4, h: 26 };
      var ax = ctx.axes(cv, { box: { x: 48, y: strip.y + strip.h + 8, w: W - 48 - 12, h: H - (strip.y + strip.h + 8) - 34 },
        xmin: XMIN, xmax: XMAX, ymin: 0, ymax: YMAX, yticks: [0, 0.2, 0.4, 0.6, 0.8], xlabel: "position x", ylabel: "P(x, t)" });
      ax.draw();

      // particle strip: up to 700 particles as specks with a fixed vertical jitter
      var stepI = Math.max(1, Math.floor(part.x.length / 700));
      g.fillStyle = ctx.alpha(col.c[0], 0.7);
      g.beginPath();
      for (var i = 0; i < part.x.length; i += stepI) {
        var px = ax.X(part.x[i]);
        if (px < ax.box.x || px > ax.box.x + ax.box.w) continue;
        var py = strip.y + 3 + part.jit[i] * (strip.h - 6);
        g.moveTo(px + 1.4, py); g.arc(px, py, 1.4, 0, 2 * Math.PI);
      }
      g.fill();
      g.strokeStyle = col.ruleSoft; g.lineWidth = 1;
      g.strokeRect(ax.box.x + 0.5, strip.y + 0.5, ax.box.w, strip.h);

      g.save();
      g.beginPath(); g.rect(ax.box.x, ax.box.y, ax.box.w, ax.box.h); g.clip();
      // histogram
      g.fillStyle = ctx.alpha(col.c[0], 0.32);
      for (var b = 0; b < cmp.hist.length; b++) {
        if (!cmp.hist[b]) continue;
        var x0 = ax.X(Q.lo + b * cmp.binW), x1 = ax.X(Q.lo + (b + 1) * cmp.binW), y = ax.Y(Math.min(cmp.hist[b], YMAX * 1.05));
        g.fillRect(x0, y, x1 - x0 - 0.5, ax.Y(0) - y);
      }
      // steady state N(0, D), when both terms are on
      if (drift && diff) {
        g.setLineDash([5, 4]); g.strokeStyle = col.muted; g.lineWidth = 1.3;
        g.beginPath();
        for (var k = 0; k <= 200; k++) { var xs = XMIN + (XMAX - XMIN) * k / 200, ys = Math.exp(-xs * xs / (2 * Q.D)) / Math.sqrt(2 * Math.PI * Q.D); if (k) g.lineTo(ax.X(xs), ax.Y(ys)); else g.moveTo(ax.X(xs), ax.Y(ys)); }
        g.stroke(); g.setLineDash([]);
      }
      // PDE density
      var peak = 0, peakX = 0;
      g.beginPath();
      for (i = 0; i < pde.d.length; i++) {
        var xc = Q.lo + (i + 0.5) * h;
        if (pde.d[i] > peak) { peak = pde.d[i]; peakX = xc; }
        var yy = ax.Y(Math.min(pde.d[i], YMAX * 1.05));
        if (i) g.lineTo(ax.X(xc), yy); else g.moveTo(ax.X(xc), yy);
      }
      g.strokeStyle = col.c[1]; g.lineWidth = 2.2; g.stroke();
      g.restore();

      // drift arrows along the top of the plot: f(x) = -x points at the center
      g.font = "11px " + sans;
      if (drift) {
        g.strokeStyle = col.soft; g.fillStyle = col.soft; g.lineWidth = 1.2;
        var ay = ax.box.y + 12;
        [-4, -3, -2, -1, 1, 2, 3, 4].forEach(function (xa) {
          var len = 0.11 * C.f(xa), xA = ax.X(xa), xB = ax.X(xa + len), dir = xB > xA ? 1 : -1;
          g.beginPath(); g.moveTo(xA, ay); g.lineTo(xB, ay); g.stroke();
          g.beginPath(); g.moveTo(xB, ay); g.lineTo(xB - dir * 4, ay - 3); g.lineTo(xB - dir * 4, ay + 3); g.closePath(); g.fill();
        });
      }
      g.fillStyle = col.muted; g.textAlign = "left"; g.textBaseline = "top";
      if (peak > YMAX) {
        g.textAlign = peakX > 0 ? "right" : "left";
        g.fillText("PDE peak " + ctx.fmt(peak, 1) + ", off the scale", ax.X(peakX) + (peakX > 0 ? -6 : 6), ax.box.y + 22);
      }

      kD.i.style.background = ctx.alpha(col.c[0], 0.8);
      kH.i.style.background = ctx.alpha(col.c[0], 0.55);
      kH.t.textContent = "histogram of " + part.x.length + " particles";
      kP.i.style.background = col.c[1];
      kS.i.style.background = col.muted;
      ro.set({
        "time t": ctx.fmt(pde.t, 2),
        "L1 gap": ctx.fmt(cmp.L1, 3),
        "PDE mass": cmp.mass.toFixed(6),
        "mean x, PDE": ctx.fmt(cmp.meanPDE, 3),
        "mean x, particles": ctx.fmt(cmp.meanParticles, 3)
      });
    }

    reset();
    if (ctx.reducedMotion) {
      while (pde.t < Q.tMax - 1e-9) { C.pdeAdvance(pde, Q.frameDt, drift, diff); C.partAdvance(part, Q.frameDt, drift, diff); }
    }
    cv.onresize = draw;
    draw();
    return { draw: draw };
  }
});
