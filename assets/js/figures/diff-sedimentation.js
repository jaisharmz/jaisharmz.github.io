FIG.register("diff-sedimentation", {
  defaults: { kT: 1, mult: 1, mg: 1, gamma: 1, N: 3000, zTop: 6, bins: 40, seed: 7, simPerFrame: 0.066 },

  // Pure math, no DOM. Overdamped Langevin dynamics for the speck of dust:
  //   dz = -(m g / gamma) dt + sqrt(2 D) dw,   reflecting floor at z = 0.
  // Drift alone gives the text's J_drift = -rho m g / gamma, diffusion alone
  // gives J_diffusion = -D d rho / dz, so the stationary profile is
  // rho(z) ~ exp(-m g z / (gamma D)). Boltzmann says rho(z) ~ exp(-m g z / (k_B T)).
  // They agree when D = k_B T / gamma. The figure sets D = mult * k_B T / gamma.
  compute: {
    derived: function (p) {
      var v = p.mg / p.gamma;                             // terminal velocity m g / gamma
      var D = p.mult * p.kT / p.gamma;
      // step size: drift per step is 1% of the decay length gamma D / (m g)
      var dt = Math.max(0.001, Math.min(0.05, 0.01 * D / (v * v)));
      return {
        v: v, D: D, dt: dt,
        lB: p.kT / p.mg,                                  // Boltzmann decay length k_B T / (m g)
        lD: p.gamma * D / p.mg                            // decay length the dynamics produce
      };
    },

    // Particles start well mixed over the column, as if the jar was just shaken.
    init: function (p, seed) {
      var r = FIG.ctx.rng(seed), z = new Float64Array(p.N), jx = new Float64Array(p.N);
      for (var i = 0; i < p.N; i++) { z[i] = r.uniform() * p.zTop; jx[i] = r.uniform(); }
      return { z: z, jx: jx, t: 0, rng: r };
    },

    // Euler-Maruyama steps; a particle that crosses the floor is reflected.
    step: function (s, p, nSteps) {
      var d = this.derived(p), z = s.z, n = z.length;
      var drift = d.v * d.dt, kick = Math.sqrt(2 * d.D * d.dt), r = s.rng;
      for (var k = 0; k < nSteps; k++) {
        for (var i = 0; i < n; i++) {
          var x = z[i] - drift + kick * r.normal();
          z[i] = x < 0 ? -x : x;
        }
      }
      s.t += nSteps * d.dt;
      return s;
    },

    stepsPerFrame: function (p) {
      // a guard only: the slider range needs at most 66 steps per frame (about 3 ms)
      return Math.max(1, Math.min(80, Math.round(p.simPerFrame / this.derived(p).dt)));
    },

    // Mean height is the maximum-likelihood decay length of an exponential profile.
    stats: function (s, p) {
      var z = s.z, n = z.length, w = p.zTop / p.bins, hist = new Float64Array(p.bins);
      var sum = 0, above = 0;
      for (var i = 0; i < n; i++) {
        sum += z[i];
        if (z[i] >= p.zTop) above++;
        else hist[Math.floor(z[i] / w)] += 1;
      }
      for (var b = 0; b < p.bins; b++) hist[b] /= n * w;  // density, integrates to the fraction shown
      return { mean: sum / n, above: above / n, hist: hist, binWidth: w, t: s.t };
    },

    // Run the same frames the animation runs, and average the mean height
    // over the second half of the run.
    simulate: function (p, seed, T) {
      var s = this.init(p, seed), spf = this.stepsPerFrame(p), acc = 0, cnt = 0;
      while (s.t < T - 1e-9) {
        this.step(s, p, spf);
        if (s.t > T / 2) { acc += this.stats(s, p).mean; cnt++; }
      }
      var st = this.stats(s, p);
      st.avgMean = acc / cnt;
      st.frames = Math.round(s.t / (spf * this.derived(p).dt));
      return st;
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D0 = this.defaults;
    var p = {};
    Object.keys(D0).forEach(function (k) { p[k] = D0[k]; });

    var row = ctx.controls(stage);
    var sT = ctx.slider(row, {
      label: "temperature k_B T", min: 0.4, max: 2, step: 0.05, value: p.kT,
      fmt: function (v) { return v.toFixed(2); },
      oninput: function (v) { p.kT = v; changed(); }
    });
    var sD = ctx.slider(row, {
      label: "diffusion D, × k_B T / γ", min: -2, max: 2, step: 0.05, value: 0,
      fmt: function (v) { return Math.pow(2, v).toFixed(2) + "×"; },
      oninput: function (v) { p.mult = Math.pow(2, v); changed(); }
    });
    var bPlay = ctx.button(row, "pause", function () {
      if (anim.running) { userPaused = true; anim.stop(); }
      else { userPaused = false; idle = 0; anim.start(); }
      label();
    });
    bPlay.classList.add("fig__btn--primary");
    ctx.button(row, "shake the jar", function () { reset(); if (!userPaused) anim.start(); label(); draw(); });
    // k_B with a real subscript in every label and readout (static strings only)
    function subB(root) {
      root.querySelectorAll(".fig__slider-label, .fig__legend .lg-t, .fig__readout dt").forEach(function (e) {
        if (e.textContent.indexOf("k_B") >= 0 && !e.querySelector("sub")) {
          var html = e.innerHTML.replace(/k_B ?/g, "k<sub>B</sub>");
          e.innerHTML = html;
        }
      });
    }

    var cv = ctx.canvas(stage, { aspect: 0.5, minHeight: 280, maxHeight: 420,
      label: "Particles settling under gravity beside a histogram of their heights and the Boltzmann profile" });
    var legend = ctx.el("div", "fig__legend", stage);
    function key(text, cls) {
      var s = ctx.el("span", null, legend), i = ctx.el("i", cls || null, s);
      ctx.el("span", "lg-t", s, text).style.display = "inline"; // one flex item, so a <sub> stays inline
      return i;
    }
    var kP = key("particles and their histogram"), kB = key("Boltzmann profile, decay length k_B T / (m g)"),
        kS = key("stationary profile for this D, decay length γD / (m g)");
    kP.style.height = ".55rem";
    var ro = ctx.readout(stage);

    var state, userPaused = false, idle = 0;
    function reset() { state = C.init(p, D0.seed); idle = 0; }
    reset();

    // fixed frames of 1/60 s, replayed from real time, so the speed does not
    // depend on the display and the sequence matches tools/figtest exactly
    var acc = 0;
    var anim = ctx.loop(stage, function (dt) {
      acc += dt;
      var k = 0;
      while (acc >= 1 / 60 - 1e-6 && k < 3) { C.step(state, p, C.stepsPerFrame(p)); idle += D0.simPerFrame; acc -= 1 / 60; k++; }
      if (k === 3) acc = 0;
      if (!k) return;
      draw();
      if (idle > 80) { setTimeout(label, 0); return false; } // rest once it has sat at equilibrium a while
    });
    function label() { bPlay.textContent = anim.running ? "pause" : "play"; }
    function changed() {
      idle = 0;
      if (!userPaused && !anim.running && !ctx.reducedMotion) { anim.start(); label(); }
      draw();
    }

    cv.onresize = draw;
    var font = getComputedStyle(document.documentElement).getPropertyValue("--sans");

    function draw() {
      var col = ctx.colors(), g = cv.g, d = C.derived(p), st = C.stats(state, p);
      cv.clear();
      var narrow = cv.w < 520;
      var m = { l: 40, r: 12, t: 22, b: 36 };
      var stripW = Math.max(56, Math.min(130, cv.w * 0.2));
      var gap = narrow ? 12 : 22;
      var H = cv.h - m.t - m.b;
      var strip = { x: m.l, y: m.t, w: stripW, h: H };
      var plotBox = { x: m.l + stripW + gap, y: m.t, w: cv.w - m.l - stripW - gap - m.r, h: H };

      var xmax = Math.max(1 / d.lB, 1 / d.lD) * 1.12;
      var ax = ctx.axes(cv, {
        box: plotBox, xmin: 0, xmax: xmax, ymin: 0, ymax: p.zTop,
        xlabel: "", yticks: [], xfmt: function (t) { return String(+t.toPrecision(2)); }
      });
      ax.draw();
      var Y = ax.Y;

      g.save();
      g.font = "11px " + font; g.fillStyle = col.muted;
      // shared height axis, on the left of the strip
      g.textAlign = "right"; g.textBaseline = "middle";
      ctx.niceTicks(0, p.zTop, 6).forEach(function (t) {
        g.fillText(String(t), strip.x - 6, Y(t));
        g.strokeStyle = col.ruleSoft; g.lineWidth = 1;
        g.beginPath(); g.moveTo(plotBox.x, Math.round(Y(t)) + .5); g.lineTo(plotBox.x + plotBox.w, Math.round(Y(t)) + .5); g.stroke();
      });
      g.save(); g.translate(11, strip.y + strip.h / 2); g.rotate(-Math.PI / 2);
      g.textAlign = "center"; g.fillText("height z", 0, 0); g.restore();
      // panel titles
      g.textAlign = "left"; g.textBaseline = "bottom"; g.fillStyle = col.ink;
      g.fillText("particles", strip.x, m.t - 6);
      g.fillText("density ρ(z)", plotBox.x, m.t - 6);
      g.textAlign = "center"; g.textBaseline = "bottom"; g.fillStyle = col.muted;
      g.fillText("fraction of particles per unit height", plotBox.x + plotBox.w / 2, cv.h - 2);

      // the jar
      g.fillStyle = col.sunk; g.fillRect(strip.x, strip.y, strip.w, strip.h);
      g.fillStyle = ctx.alpha(col.c[0], 0.55);
      var z = state.z, jx = state.jx, n = z.length;
      for (var i = 0; i < n; i++) {
        if (z[i] > p.zTop) continue;
        g.fillRect(strip.x + 2 + jx[i] * (strip.w - 6), Y(z[i]) - 1, 2, 2);
      }
      g.strokeStyle = col.rule; g.lineWidth = 1;
      g.strokeRect(strip.x + .5, strip.y + .5, strip.w - 1, strip.h - 1);
      g.strokeStyle = col.ink; g.lineWidth = 2.5;
      g.beginPath(); g.moveTo(strip.x, Y(0)); g.lineTo(strip.x + strip.w, Y(0)); g.stroke();
      if (st.above > 0.005) {
        var txt = "\u2191 " + Math.round(100 * st.above) + "% above";
        g.textAlign = "center"; g.textBaseline = "top"; g.lineJoin = "round";
        g.lineWidth = 3; g.strokeStyle = col.sunk; g.strokeText(txt, strip.x + strip.w / 2, strip.y + 4);
        g.fillStyle = col.ink; g.fillText(txt, strip.x + strip.w / 2, strip.y + 4);
      }

      // histogram of heights
      g.save();
      g.beginPath(); g.rect(plotBox.x, plotBox.y, plotBox.w, plotBox.h); g.clip();
      g.fillStyle = ctx.alpha(col.c[0], 0.42);
      for (var b = 0; b < st.hist.length; b++) {
        var y0 = Y((b + 1) * st.binWidth), y1 = Y(b * st.binWidth);
        g.fillRect(plotBox.x, y0 + 0.5, ax.X(st.hist[b]) - plotBox.x, y1 - y0 - 1);
      }
      // stationary profile for this D (dashed) and the Boltzmann profile (solid)
      function curve(l, color, width, dash) {
        g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []);
        g.beginPath();
        for (var k = 0; k <= 200; k++) {
          var zz = p.zTop * k / 200, px = ax.X(Math.exp(-zz / l) / l), py = Y(zz);
          if (k === 0) g.moveTo(px, py); else g.lineTo(px, py);
        }
        g.stroke(); g.setLineDash([]);
      }
      curve(d.lD, col.ink, 1.6, [5, 4]);
      curve(d.lB, col.c[1], 2.4);
      g.restore();
      g.restore();

      kP.style.background = ctx.alpha(col.c[0], 0.55);
      kB.style.background = col.c[1];
      kS.style.background = "repeating-linear-gradient(90deg," + col.ink + " 0 4px, transparent 4px 7px)";

      ro.set({
        "mean height": st.mean.toFixed(3),
        "Boltzmann length": d.lB.toFixed(3),
        "D": d.D.toFixed(3),
        "k_B T / γ": (p.kT / p.gamma).toFixed(3),
        "time": st.t.toFixed(1)
      });
    }

    if (ctx.reducedMotion) {
      // the end state, already settled; play is offered, never automatic
      while (state.t < 20) C.step(state, p, C.stepsPerFrame(p));
      userPaused = true;
    } else {
      anim.start();
    }
    label();
    draw();
    subB(stage);
    return { draw: draw };
  }
});
