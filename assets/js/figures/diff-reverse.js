FIG.register("diff-reverse", {
  compute: {
    // data: 0.3 N(-2, 0.3^2) + 0.7 N(1.5, 0.3^2); VP schedule beta_min = 0.1, beta_max = 20
    data: { w: [0.3, 0.7], mu: [-2, 1.5], s: [0.3, 0.3] },
    sched: { bmin: 0.1, bmax: 20 },
    split: -0.25,
    defaults: { nPart: 300, nSteps: 1000, keep: 8, seed: 1, mode: "sde" },
    // sampler settings: weight on the score term and whether g(t) dw is kept
    modes: {
      sde: { c: 1, noise: true },
      drift: { c: 1, noise: false },
      ode: { c: 0.5, noise: false }
    },

    beta: function (t) { var s = this.sched; return s.bmin + t * (s.bmax - s.bmin); },
    alpha: function (t) { var s = this.sched; return Math.exp(-0.5 * (s.bmin * t + 0.5 * (s.bmax - s.bmin) * t * t)); },

    // exact marginal at time t: sum_k w_k N(alpha mu_k, alpha^2 s_k^2 + 1 - alpha^2)
    marginal: function (t) {
      var a = this.alpha(t), d = this.data, m = [], v = [], lw = [];
      for (var j = 0; j < d.w.length; j++) {
        m.push(a * d.mu[j]);
        v.push(a * a * d.s[j] * d.s[j] + 1 - a * a);
        lw.push(Math.log(d.w[j]) - 0.5 * Math.log(2 * Math.PI * v[j]));
      }
      return { m: m, v: v, lw: lw };
    },
    density: function (x, mg) {
      var p = 0;
      for (var j = 0; j < mg.m.length; j++) { var u = x - mg.m[j]; p += Math.exp(mg.lw[j] - u * u / (2 * mg.v[j])); }
      return p;
    },
    // grad_x log P(x, t), computed in log space so far-off points do not underflow
    score: function (x, mg) {
      var J = mg.m.length, best = -Infinity, l = [], j;
      for (j = 0; j < J; j++) { var u = x - mg.m[j]; l[j] = mg.lw[j] - u * u / (2 * mg.v[j]); if (l[j] > best) best = l[j]; }
      var num = 0, den = 0;
      for (j = 0; j < J; j++) { var r = Math.exp(l[j] - best); den += r; num += r * (mg.m[j] - x) / mg.v[j]; }
      return num / den;
    },

    // Reverse-time Euler-Maruyama from x ~ N(0,1) at t = 1 down to t = 0:
    //   x_{t - dt} = x_t - [f - c g^2 s] dt (+ g sqrt(dt) z), f = -1/2 beta x, g^2 = beta.
    // c = 1 is the text's reverse SDE; c = 1/2 without noise is the probability flow ODE.
    sample: function (p) {
      var mode = this.modes[p.mode], n = p.nPart, N = p.nSteps, dt = 1 / N, K = N / p.keep + 1;
      var r0 = FIG.ctx.rng(p.seed), rn = FIG.ctx.rng(p.seed + 7919);
      var x = new Float64Array(n), traj = new Float32Array(n * K), i;
      for (i = 0; i < n; i++) { x[i] = r0.normal(); traj[i * K] = x[i]; }
      for (var k = 0; k < N; k++) {
        var t = 1 - k * dt, b = this.beta(t), mg = this.marginal(t), sq = Math.sqrt(b * dt);
        // two-component score inlined (same formula as score()), no allocation per particle
        var m0 = mg.m[0], m1 = mg.m[1], v0 = mg.v[0], v1 = mg.v[1], lw0 = mg.lw[0], lw1 = mg.lw[1];
        for (i = 0; i < n; i++) {
          var xi = x[i], u0 = xi - m0, u1 = xi - m1;
          var l0 = lw0 - u0 * u0 / (2 * v0), l1 = lw1 - u1 * u1 / (2 * v1), top = l0 > l1 ? l0 : l1;
          var e0 = Math.exp(l0 - top), e1 = Math.exp(l1 - top);
          var sc = (e0 * (m0 - xi) / v0 + e1 * (m1 - xi) / v1) / (e0 + e1);
          x[i] = xi + (0.5 * b * xi + mode.c * b * sc) * dt;
          if (mode.noise) x[i] += sq * rn.normal();
        }
        if ((k + 1) % p.keep === 0) { var f = (k + 1) / p.keep; for (i = 0; i < n; i++) traj[i * K + f] = x[i]; }
      }
      return { traj: traj, K: K, final: x, stats: this.stats(x) };
    },

    stats: function (x) {
      var L = [], R = [];
      for (var i = 0; i < x.length; i++) (x[i] < this.split ? L : R).push(x[i]);
      function sd(a) {
        if (a.length < 2) return NaN;
        var m = 0; a.forEach(function (v) { m += v; }); m /= a.length;
        var s = 0; a.forEach(function (v) { s += (v - m) * (v - m); });
        return Math.sqrt(s / (a.length - 1));
      }
      return { nLeft: L.length, nRight: R.length, shareLeft: L.length / x.length, shareRight: R.length / x.length, sdLeft: sd(L), sdRight: sd(R) };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, P = C.defaults;
    var mode = P.mode, res = null, cur = 0, progress = 1, heat = null, heatKey = "";
    var YMIN = -3.5, YMAX = 3.5;

    var row = ctx.controls(stage);
    ctx.toggle(row, {
      label: "sampler", value: mode,
      options: [{ value: "sde", label: "reverse SDE" }, { value: "drift", label: "no noise term" },
                { value: "ode", label: "half score, no noise" }],
      onchange: function (v) { mode = v; res = C.sample(Object.assign({}, P, { mode: mode })); draw(); }
    });
    var playBtn = ctx.button(row, "Play", function () {
      if (anim.running) { anim.stop(); playBtn.textContent = "Play"; return; }
      if (progress >= 1) progress = 0;
      anim.start(); playBtn.textContent = "Pause";
    });
    playBtn.classList.add("fig__btn--primary");

    var cv = ctx.canvas(stage, { aspect: 0.5, minHeight: 290, maxHeight: 440,
      label: "Reverse-time trajectories of 300 samples from t = 1 to t = 0 and a histogram of where they land" });
    var lg = ctx.el("div", "fig__legend", stage);
    function key(label, dot) { var s = ctx.el("span", null, lg); var i = ctx.el("i", dot ? "dot" : null, s); ctx.el("span", null, s, label); return i; }
    var kL = key("path ends in the mode at −2"), kR = key("path ends in the mode at 1.5"),
        kH = key("histogram of the samples"), kP = key("exact P(x, t), also the shading");
    var ro = ctx.readout(stage);

    var anim = ctx.loop(stage, function (dtSec) {
      progress = Math.min(1, progress + dtSec / 3.5);
      draw();
      if (progress >= 1) { playBtn.textContent = "Play"; return false; }
    });

    // exact density on a (t, x) grid, each time column scaled to its own peak
    function heatImage(col, bw, bh) {
      var k = col.ink + bw + "x" + bh;
      if (heat && heatKey === k) return heat;
      var nt = Math.min(240, Math.max(60, Math.round(bw / 2))), nx = Math.min(200, Math.max(60, Math.round(bh / 2)));
      var off = document.createElement("canvas");
      off.width = nt; off.height = nx;
      var og = off.getContext("2d"), img = og.createImageData(nt, nx);
      var rgb = rgbOf(col.ink), colv = new Float64Array(nx);
      for (var a = 0; a < nt; a++) {
        var t = 1 - (a + 0.5) / nt, mg = C.marginal(t), mx = 0, b;
        for (b = 0; b < nx; b++) { var x = YMAX - (b + 0.5) / nx * (YMAX - YMIN); colv[b] = C.density(x, mg); if (colv[b] > mx) mx = colv[b]; }
        for (b = 0; b < nx; b++) {
          var o = 4 * (b * nt + a);
          img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2];
          img.data[o + 3] = Math.round(255 * 0.2 * Math.sqrt(colv[b] / mx));
        }
      }
      og.putImageData(img, 0, 0);
      heat = off; heatKey = k;
      return off;
    }
    function rgbOf(c) {
      c = c.trim();
      if (c[0] === "#") {
        var h = c.slice(1);
        if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
        var n = parseInt(h, 16);
        return [n >> 16 & 255, n >> 8 & 255, n & 255];
      }
      var m = c.match(/[\d.]+/g) || [0, 0, 0];
      return [+m[0], +m[1], +m[2]];
    }

    function draw() {
      var col = ctx.colors(), g = cv.g, W = cv.w, H = cv.h, K = res.K, n = P.nPart;
      var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
      cur = Math.round(progress * (K - 1));
      var tNow = 1 - cur * P.keep / P.nSteps;
      cv.clear();
      var mainR = Math.round(W * 0.72), sideL = mainR + Math.max(10, Math.round(W * 0.025));
      var top = 10, bot = 34;
      var ax = ctx.axes(cv, { box: { x: 40, y: top, w: mainR - 40, h: H - top - bot },
        xmin: 1, xmax: 0, ymin: YMIN, ymax: YMAX, xticks: [1, 0.8, 0.6, 0.4, 0.2, 0],
        xlabel: "time t, running from 1 down to 0", ylabel: "x" });
      g.save();
      g.imageSmoothingEnabled = true;
      g.drawImage(heatImage(col, ax.box.w, ax.box.h), ax.box.x, ax.box.y, ax.box.w, ax.box.h);
      g.restore();
      ax.draw();

      // trajectories up to the current frame, colored by the mode they end in
      var fin = res.final;
      g.save();
      g.beginPath(); g.rect(ax.box.x, ax.box.y, ax.box.w, ax.box.h); g.clip();
      g.lineWidth = 0.8;
      [0, 1].forEach(function (side) {
        g.beginPath();
        for (var i = 0; i < n; i += 2) {
          if ((fin[i] < C.split ? 0 : 1) !== side) continue;
          var base = i * K;
          g.moveTo(ax.X(1), ax.Y(res.traj[base]));
          for (var f = 1; f <= cur; f++) g.lineTo(ax.X(1 - f * P.keep / P.nSteps), ax.Y(res.traj[base + f]));
        }
        g.strokeStyle = ctx.alpha(side ? col.c[0] : col.c[1], mode === "sde" ? 0.3 : 0.4);
        g.stroke();
      });
      // current positions
      var xNow = ax.X(tNow);
      for (var i = 0; i < n; i++) {
        g.fillStyle = fin[i] < C.split ? col.c[1] : col.c[0];
        g.beginPath(); g.arc(xNow, ax.Y(res.traj[i * K + cur]), 1.7, 0, 2 * Math.PI); g.fill();
      }
      g.restore();

      // side panel: histogram at the current time against the exact P(x, t)
      var bw = 0.14, nb = Math.round((YMAX - YMIN) / bw), counts = new Float64Array(nb);
      for (i = 0; i < n; i++) {
        var b = Math.floor((res.traj[i * K + cur] - YMIN) / bw);
        if (b >= 0 && b < nb) counts[b]++;
      }
      var mg = C.marginal(tNow), hmax = 0, pmax = 0, j;
      for (j = 0; j < nb; j++) { counts[j] /= n * bw; if (counts[j] > hmax) hmax = counts[j]; }
      for (j = 0; j <= 200; j++) pmax = Math.max(pmax, C.density(YMIN + (YMAX - YMIN) * j / 200, mg));
      var dmax = Math.max(1, 1.1 * Math.max(hmax, pmax));
      var sx = ctx.axes(cv, { box: { x: sideL, y: top, w: W - 10 - sideL, h: H - top - bot },
        xmin: 0, xmax: dmax, ymin: YMIN, ymax: YMAX, yticks: [-2, 0, 2], yfmt: function () { return ""; },
        xticks: dmax < 2 ? [1] : [Math.max(1, Math.floor(dmax / 2) * 2)], xlabel: "density" });
      sx.draw();
      g.save();
      g.beginPath(); g.rect(sx.box.x, sx.box.y, sx.box.w, sx.box.h); g.clip();
      g.fillStyle = ctx.alpha(col.c[2], 0.4);
      for (j = 0; j < nb; j++) {
        if (!counts[j]) continue;
        var y0 = sx.Y(YMIN + (j + 1) * bw), y1 = sx.Y(YMIN + j * bw);
        g.fillRect(sx.X(0), y0, sx.X(counts[j]) - sx.X(0), y1 - y0);
      }
      g.beginPath();
      for (j = 0; j <= 240; j++) {
        var xv = YMIN + (YMAX - YMIN) * j / 240, px = sx.X(C.density(xv, mg)), py = sx.Y(xv);
        if (j) g.lineTo(px, py); else g.moveTo(px, py);
      }
      g.strokeStyle = col.ink; g.lineWidth = 1.6; g.stroke();
      g.restore();
      g.font = "11px " + sans;
      g.fillStyle = col.muted;
      g.textAlign = "right"; g.textBaseline = "top";
      g.fillText("t = " + ctx.fmt(Math.max(0, tNow), 2), sx.box.x + sx.box.w - 4, sx.box.y + 4);

      kL.style.background = col.c[1];
      kR.style.background = col.c[0];
      kH.style.background = ctx.alpha(col.c[2], 0.6);
      kP.style.background = col.ink;
      var s = res.stats;
      ro.set({
        "time t": ctx.fmt(Math.max(0, tNow), 2),
        "share at −2 (true 0.30)": ctx.fmt(s.shareLeft, 2),
        "share at 1.5 (true 0.70)": ctx.fmt(s.shareRight, 2),
        "spread at −2 (true 0.30)": isFinite(s.sdLeft) ? ctx.fmt(s.sdLeft, 2) : "no samples",
        "spread at 1.5 (true 0.30)": isFinite(s.sdRight) ? ctx.fmt(s.sdRight, 2) : "no samples"
      });
    }

    res = C.sample(Object.assign({}, P, { mode: mode }));
    cv.onresize = draw;
    draw();
    return { draw: draw };
  }
});
