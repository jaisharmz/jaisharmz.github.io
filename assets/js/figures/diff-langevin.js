FIG.register("diff-langevin", {
  // seed: the initial cloud and every z; n particles; eps the step size;
  // noise "2" = sqrt(2 eps) z, "1" = sqrt(eps) z, "0" = no noise; a run is `steps`
  // steps; readouts average a snapshot every `every` steps over the last `window`.
  defaults: { seed: 3, n: 500, eps: 0.05, noise: "2", steps: 2000, every: 10, window: 50, cloud: 2 },

  // Pure math, no DOM. Notation follows the text: P(x) is the target density,
  // the update is x <- x + eps grad log P(x) + c z with z ~ N(0, I).
  compute: {
    // P(x) = sum_k w_k N(x; mu_k, s^2 I) in two dimensions
    mix: { w: [0.5, 0.3, 0.2], mu: [[-1.3, -0.75], [1.3, -0.75], [0, 1.5]], s: 0.55 },

    noiseScale: function (noise) { return noise === "2" ? Math.SQRT2 : (noise === "1" ? 1 : 0); },

    // Writes grad log P(x, y) into out and returns the most responsible mode,
    // argmax_k w_k N(x; mu_k, s^2 I). That mode is how particles are assigned.
    score: function (x, y, out) {
      var M = this.mix, s2 = M.s * M.s, l0, l1, l2, d0x, d0y, d1x, d1y, d2x, d2y;
      d0x = x - M.mu[0][0]; d0y = y - M.mu[0][1];
      d1x = x - M.mu[1][0]; d1y = y - M.mu[1][1];
      d2x = x - M.mu[2][0]; d2y = y - M.mu[2][1];
      l0 = Math.log(M.w[0]) - 0.5 * (d0x * d0x + d0y * d0y) / s2;
      l1 = Math.log(M.w[1]) - 0.5 * (d1x * d1x + d1y * d1y) / s2;
      l2 = Math.log(M.w[2]) - 0.5 * (d2x * d2x + d2y * d2y) / s2;
      var m = Math.max(l0, l1, l2);
      var p0 = Math.exp(l0 - m), p1 = Math.exp(l1 - m), p2 = Math.exp(l2 - m), den = p0 + p1 + p2;
      out[0] = -(p0 * d0x + p1 * d1x + p2 * d2x) / (den * s2);
      out[1] = -(p0 * d0y + p1 * d1y + p2 * d2y) / (den * s2);
      return l0 >= l1 && l0 >= l2 ? 0 : (l1 >= l2 ? 1 : 2);
    },

    density: function (x, y) {
      var M = this.mix, s2 = M.s * M.s, p = 0;
      for (var k = 0; k < 3; k++) {
        var dx = x - M.mu[k][0], dy = y - M.mu[k][1];
        p += M.w[k] * Math.exp(-0.5 * (dx * dx + dy * dy) / s2) / (2 * Math.PI * s2);
      }
      return p;
    },

    // Particles start from a broad cloud N(0, cloud^2 I).
    init: function (n, seed, cloud) {
      var r = FIG.ctx.rng(seed), x = new Float64Array(n), y = new Float64Array(n);
      for (var i = 0; i < n; i++) { x[i] = cloud * r.normal(); y[i] = cloud * r.normal(); }
      return { x: x, y: y, n: n, r: r, step: 0, mode: new Int8Array(n) };
    },

    // k Langevin steps: x <- x + eps grad log P(x) + c sqrt(eps) z
    advance: function (st, eps, noise, k) {
      var c = this.noiseScale(noise) * Math.sqrt(eps), g = [0, 0], r = st.r;
      for (var s = 0; s < k; s++) {
        for (var i = 0; i < st.n; i++) {
          this.score(st.x[i], st.y[i], g);
          st.x[i] += eps * g[0];
          st.y[i] += eps * g[1];
          if (c > 0) { st.x[i] += c * r.normal(); st.y[i] += c * r.normal(); }
        }
        st.step++;
      }
    },

    // Fraction of particles per mode and the per-axis variance inside each mode.
    snapshot: function (st) {
      var cnt = [0, 0, 0], sx = [0, 0, 0], sy = [0, 0, 0], sq = [0, 0, 0], g = [0, 0];
      for (var i = 0; i < st.n; i++) {
        var k = this.score(st.x[i], st.y[i], g);
        st.mode[i] = k;
        cnt[k]++; sx[k] += st.x[i]; sy[k] += st.y[i];
        sq[k] += st.x[i] * st.x[i] + st.y[i] * st.y[i];
      }
      var w = [], v = [];
      for (var j = 0; j < 3; j++) {
        w.push(cnt[j] / st.n);
        if (cnt[j] < 3) { v.push(NaN); continue; }
        var mx = sx[j] / cnt[j], my = sy[j] / cnt[j];
        v.push(Math.max(0, (sq[j] / cnt[j] - mx * mx - my * my) / 2));
      }
      return { w: w, v: v };
    },

    average: function (snaps) {
      var w = [0, 0, 0], v = [0, 0, 0], nv = [0, 0, 0];
      snaps.forEach(function (s) {
        for (var k = 0; k < 3; k++) {
          w[k] += s.w[k];
          if (isFinite(s.v[k])) { v[k] += s.v[k]; nv[k]++; }
        }
      });
      for (var k = 0; k < 3; k++) { w[k] /= Math.max(1, snaps.length); v[k] = nv[k] ? v[k] / nv[k] : NaN; }
      return { w: w, v: v, count: snaps.length };
    },

    // The same two statistics computed exactly on P^power by integrating a fine
    // grid, with the same nearest-mode assignment. power 1 is the target P;
    // power 2 is the sharpened P^2, the distribution at half the temperature.
    targetStats: function (power) {
      this._ts = this._ts || {};
      if (this._ts[power]) return this._ts[power];
      var h = 0.025, L = 6, m = [0, 0, 0], sx = [0, 0, 0], sy = [0, 0, 0], sq = [0, 0, 0], g = [0, 0];
      for (var x = -L + h / 2; x < L; x += h) {
        for (var y = -L + h / 2; y < L; y += h) {
          var p = Math.pow(this.density(x, y), power);
          var k = this.score(x, y, g);
          m[k] += p; sx[k] += p * x; sy[k] += p * y; sq[k] += p * (x * x + y * y);
        }
      }
      var tot = m[0] + m[1] + m[2], w = [], v = [];
      for (var j = 0; j < 3; j++) {
        var mx = sx[j] / m[j], my = sy[j] / m[j];
        w.push(m[j] / tot);
        v.push((sq[j] / m[j] - mx * mx - my * my) / 2);
      }
      this._ts[power] = { w: w, v: v };
      return this._ts[power];
    },

    // One full run at fixed settings: what the figure shows when left alone.
    simulate: function (p, seed) {
      var st = this.init(p.n, seed, p.cloud), snaps = [];
      for (var s = 0; s < p.steps; s += p.every) {
        this.advance(st, p.eps, p.noise, p.every);
        snaps.push(this.snapshot(st));
        if (snaps.length > p.window) snaps.shift();
      }
      return { state: st, avg: this.average(snaps), last: snaps[snaps.length - 1] };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var tP = C.targetStats(1), tP2 = C.targetStats(2);
    var st, snaps = [], stopAt = D.steps;
    var cfg = { eps: D.eps, noise: D.noise };

    var row = ctx.controls(stage);
    var tg = ctx.toggle(row, {
      label: "noise term",
      options: [{ value: "2", label: "√(2ε) z" }, { value: "1", label: "√ε z" }, { value: "0", label: "none" }],
      value: D.noise,
      onchange: function (v) { cfg.noise = v; changed(); }
    });
    var sE = ctx.slider(row, {
      label: "step size ε", min: 0.005, max: 0.3, step: 0.005, value: D.eps,
      fmt: function (v) { return v.toFixed(3); },
      oninput: function (v) { cfg.eps = v; changed(); }
    });
    var bPlay = ctx.button(row, "Pause", function () {
      if (anim.running) { anim.stop(); syncButton(); }
      else play();
    });
    bPlay.classList.add("fig__btn--primary");
    ctx.button(row, "Reset", function () {
      cfg.eps = D.eps; cfg.noise = D.noise; sE.set(D.eps); tg.set(D.noise, false);
      reset();
    });

    var grid = ctx.el("div", "fig__row fig__row--3-2", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);

    // stack the two panels whenever the stage itself is narrow, not only the viewport
    function stack() { grid.style.gridTemplateColumns = stage.clientWidth < 560 ? "1fr" : ""; }
    stack();
    if (window.ResizeObserver) new ResizeObserver(stack).observe(stage);
    var cv = ctx.canvas(left, { aspect: 0.92, minHeight: 260, maxHeight: 440,
      label: "Langevin particles on a three-mode density P(x), colored by nearest mode" });
    var cb = ctx.canvas(right, { aspect: 0.95, minHeight: 200, maxHeight: 340,
      label: "Mode weights and within-mode variance of the particles against the target" });
    var legend = ctx.el("div", "fig__legend", right);
    var ro = ctx.readout(stage);
    ro.el.style.gridTemplateColumns = "repeat(auto-fill, minmax(17rem, 1fr))";

    function reset() {
      st = C.init(D.n, D.seed, D.cloud);
      snaps = [];
      stopAt = D.steps;
      if (ctx.reducedMotion) { runTo(stopAt); anim.stop(); draw(); syncButton(); }
      else { draw(); anim.start(); syncButton(); }
    }

    function changed() {
      snaps = [];                       // readouts describe the current setting only
      stopAt = st.step + D.steps;
      if (ctx.reducedMotion) { runTo(stopAt); draw(); }
      else if (!anim.running) { anim.start(); syncButton(); }
      else draw();
    }

    function play() {
      if (st.step >= stopAt) stopAt = st.step + D.steps;
      if (ctx.reducedMotion) { runTo(stopAt); draw(); return; }
      anim.start(); syncButton();
    }

    function syncButton() { bPlay.textContent = anim.running ? "Pause" : "Play"; }

    // advance in blocks of `every`, snapshot after each block (as compute.simulate does)
    function runTo(target) {
      while (st.step < target) {
        C.advance(st, cfg.eps, cfg.noise, D.every);
        snaps.push(C.snapshot(st));
        if (snaps.length > D.window) snaps.shift();
      }
    }

    var anim = ctx.loop(stage, function () {
      runTo(Math.min(stopAt, st.step + D.every));
      draw();
      if (st.step >= stopAt) { syncButton(); setTimeout(syncButton, 0); return false; }
    });

    // ---- drawing -------------------------------------------------------
    var heat = null, heatKey = "";
    function view() {
      var yr = 2.75, xr = yr * cv.w / cv.h;
      return { xmin: -xr, xmax: xr, ymin: -yr + 0.15, ymax: yr + 0.15 };
    }
    function rgb(color) {
      var m = ctx.alpha(color, 1).match(/[\d.]+/g);
      return m ? [+m[0], +m[1], +m[2]] : [128, 128, 128];
    }
    function heatmap(col, v) {
      var key = cv.w + "x" + cv.h + col.ink;
      if (heat && heatKey === key) return heat;
      var W = Math.max(60, Math.round(cv.w / 3)), H = Math.max(40, Math.round(cv.h / 3));
      var off = document.createElement("canvas");
      off.width = W; off.height = H;
      var g = off.getContext("2d"), img = g.createImageData(W, H), c = rgb(col.ink);
      var pmax = C.density(C.mix.mu[0][0], C.mix.mu[0][1]);
      for (var j = 0; j < H; j++) {
        for (var i = 0; i < W; i++) {
          var x = v.xmin + (i + 0.5) / W * (v.xmax - v.xmin);
          var y = v.ymax - (j + 0.5) / H * (v.ymax - v.ymin);
          var a = Math.pow(Math.min(1, C.density(x, y) / pmax), 0.75);
          var o = 4 * (j * W + i);
          img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2];
          img.data[o + 3] = Math.round(255 * 0.26 * a);
        }
      }
      g.putImageData(img, 0, 0);
      heat = off; heatKey = key;
      return heat;
    }

    function draw() {
      var col = ctx.colors(), g = cv.g, v = view();
      cv.clear();
      var ax = ctx.axes(cv, { xmin: v.xmin, xmax: v.xmax, ymin: v.ymin, ymax: v.ymax,
        margin: { l: 8, r: 8, t: 8, b: 8 }, grid: false, xticks: [], yticks: [] });
      g.save();
      g.beginPath(); g.rect(ax.box.x, ax.box.y, ax.box.w, ax.box.h); g.clip();
      g.imageSmoothingEnabled = true;
      g.drawImage(heatmap(col, v), ax.box.x, ax.box.y, ax.box.w, ax.box.h);
      // particles, colored by their most responsible mode
      var sc = [0, 0];
      for (var k = 0; k < 3; k++) {
        g.fillStyle = ctx.alpha(col.c[k], 0.8);
        g.beginPath();
        for (var i = 0; i < st.n; i++) {
          if (C.score(st.x[i], st.y[i], sc) !== k) continue;
          var px = ax.X(st.x[i]), py = ax.Y(st.y[i]);
          g.moveTo(px + 1.7, py); g.arc(px, py, 1.7, 0, 2 * Math.PI);
        }
        g.fill();
      }
      // a mode whose particles have collapsed onto one point gets a ring and a count
      var now = C.snapshot(st);
      g.font = "11px " + sans; g.textBaseline = "middle"; g.textAlign = "left";
      for (k = 0; k < 3; k++) {
        if (!(now.v[k] < 1e-3) || now.w[k] === 0) continue;
        var cxk = 0, cyk = 0, nk = 0;
        for (i = 0; i < st.n; i++) if (st.mode[i] === k) { cxk += st.x[i]; cyk += st.y[i]; nk++; }
        cxk = ax.X(cxk / nk); cyk = ax.Y(cyk / nk);
        g.strokeStyle = col.c[k]; g.lineWidth = 1.5;
        g.beginPath(); g.arc(cxk, cyk, 7, 0, 2 * Math.PI); g.stroke();
        g.fillStyle = col.ink;
        g.fillText(nk + " here", cxk + 11, cyk);
      }
      // mode labels, pushed outward from the middle of the three modes
      g.font = "600 12px " + sans;
      g.textBaseline = "middle";
      var cxm = 0, cym = 0;
      for (k = 0; k < 3; k++) { cxm += C.mix.mu[k][0] / 3; cym += C.mix.mu[k][1] / 3; }
      g.textAlign = "center";
      for (k = 0; k < 3; k++) {
        // below the modes under the middle, above the one over it
        var up = C.mix.mu[k][1] > cym;
        var lx = C.mix.mu[k][0], ly = C.mix.mu[k][1] + (up ? 1.18 : -1.42);
        var lab = "mode " + (k + 1) + ", w = " + C.mix.w[k].toFixed(1);
        g.strokeStyle = col.paper; g.lineWidth = 3; g.lineJoin = "round";
        g.strokeText(lab, ax.X(lx), ax.Y(ly));
        g.fillStyle = col.c[k];
        g.fillText(lab, ax.X(lx), ax.Y(ly));
      }
      g.restore();
      ax.draw();
      drawBars(col);
      readouts();
    }

    function drawBars(col) {
      var g = cb.g, a = C.average(snaps);
      cb.clear();
      var gap = 26, pw = (cb.w - 34 - gap - 6) / 2;
      var boxW = { x: 34, y: 22, w: pw, h: cb.h - 22 - 24 };
      var boxV = { x: 34 + pw + gap, y: 22, w: pw, h: boxW.h };
      var axW = ctx.axes(cb, { box: boxW, xmin: 0, xmax: 3, ymin: 0, ymax: 0.8, xticks: [], yticks: [0, 0.2, 0.4, 0.6, 0.8] });
      var axV = ctx.axes(cb, { box: boxV, xmin: 0, xmax: 3, ymin: 0, ymax: 1.6, xticks: [], yticks: [0, 0.5, 1, 1.5] });
      axW.draw(); axV.draw();
      g.save();
      g.font = "11px " + sans; g.fillStyle = col.soft;
      g.textAlign = "left"; g.textBaseline = "bottom";
      g.fillText("mode weight", boxW.x, boxW.y - 6);
      g.fillText("variance ÷ target", boxV.x, boxV.y - 6);
      var bw = Math.min(26, pw / 3 * 0.5);
      for (var k = 0; k < 3; k++) {
        var cxW = axW.X(k + 0.5), cxV = axV.X(k + 0.5);
        // weights: bar = particles, solid tick = P, dashed tick = P^2
        if (a.count) {
          g.fillStyle = ctx.alpha(col.c[k], 0.75);
          g.fillRect(cxW - bw / 2, axW.Y(a.w[k]), bw, axW.Y(0) - axW.Y(a.w[k]));
          var r = a.v[k] / tP.v[k];
          if (isFinite(r)) {
            var rr = Math.min(r, 1.6);
            g.fillRect(cxV - bw / 2, axV.Y(rr), bw, axV.Y(0) - axV.Y(rr));
          }
        }
        g.strokeStyle = col.ink; g.lineWidth = 2;
        g.beginPath(); g.moveTo(cxW - bw / 2 - 5, axW.Y(tP.w[k])); g.lineTo(cxW + bw / 2 + 5, axW.Y(tP.w[k])); g.stroke();
        g.lineWidth = 1.5; g.setLineDash([3, 2]);
        g.beginPath(); g.moveTo(cxW - bw / 2 - 5, axW.Y(tP2.w[k])); g.lineTo(cxW + bw / 2 + 5, axW.Y(tP2.w[k])); g.stroke();
        // variance: solid tick = P (1 by definition), dashed tick = P^2 relative to P
        g.lineWidth = 2;
        g.beginPath(); g.moveTo(cxV - bw / 2 - 5, axV.Y(1)); g.lineTo(cxV + bw / 2 + 5, axV.Y(1)); g.stroke();
        var r2 = tP2.v[k] / tP.v[k];
        g.lineWidth = 1.5; g.setLineDash([3, 2]);
        g.beginPath(); g.moveTo(cxV - bw / 2 - 5, axV.Y(r2)); g.lineTo(cxV + bw / 2 + 5, axV.Y(r2)); g.stroke();
        g.setLineDash([]);
        g.fillStyle = col.c[k]; g.font = "600 11px " + sans;
        g.textAlign = "center"; g.textBaseline = "top";
        g.fillText(String(k + 1), cxW, boxW.y + boxW.h + 5);
        g.fillText(String(k + 1), cxV, boxV.y + boxV.h + 5);
      }
      g.restore();
    }

    function three(arr, d) {
      return arr.map(function (x) { return ctx.fmt(x, d); }).join(" / ");
    }
    var meaning = { "2": "samples P", "1": "samples P² (half T)", "0": "climbs to the peaks" };
    function readouts() {
      var a = C.average(snaps);
      var ratio = a.v.map(function (x, k) { return x / tP.v[k]; });
      ro.set({
        "step": String(st.step),
        "bars average the last": a.count ? a.count * D.every + " steps" : "—",
        "this noise term": meaning[cfg.noise],
        "variance ÷ target": a.count ? three(ratio, 2) : "—",
        "weights, particles": a.count ? three(a.w, 2) : "—",
        "weights, P": three(tP.w, 2),
        "weights, P²": three(tP2.w, 2)
      });
    }

    legend.innerHTML = "";
    [["solid", "target P"], ["dashed", "P² (half the temperature)"]].forEach(function (L) {
      var s = ctx.el("span", null, legend);
      var i = ctx.el("i", null, s);
      i.style.background = "none";
      i.style.borderTop = "2px " + L[0] + " var(--ink)";
      i.style.height = "0"; i.style.width = "1.2rem";
      s.appendChild(document.createTextNode(L[1]));
    });

    cv.onresize = draw;
    cb.onresize = draw;
    reset();
    return { draw: draw };
  }
});
