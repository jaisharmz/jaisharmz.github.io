FIG.register("diff-noising", {
  compute: {
    // two tight modes away from the origin stand in for two kinds of real image
    modes: [
      { name: "faces", w: 0.5, mu: [2.3, 1.6], s: 0.25 },
      { name: "cars", w: 0.5, mu: [-2.4, -1.4], s: 0.25 }
    ],
    sched: { bmin: 0.1, bmax: 20 },
    defaults: { n: 400, steps: 500, seed: 21, trails: 2 },

    beta: function (t) { var s = this.sched; return s.bmin + t * (s.bmax - s.bmin); },
    alpha: function (t) { var s = this.sched; return Math.exp(-0.5 * (s.bmin * t + 0.5 * (s.bmax - s.bmin) * t * t)); },

    // exact marginal at time t: sum_k w_k N(alpha mu_k, (alpha^2 s_k^2 + 1 - alpha^2) I)
    marginal: function (t) {
      var a = this.alpha(t);
      return this.modes.map(function (md) {
        var v = a * a * md.s * md.s + 1 - a * a;
        return { mx: a * md.mu[0], my: a * md.mu[1], v: v, lw: Math.log(md.w) - Math.log(2 * Math.PI * v) };
      });
    },
    density: function (x, y, mg) {
      var p = 0;
      for (var j = 0; j < mg.length; j++) { var u = x - mg[j].mx, w = y - mg[j].my; p += Math.exp(mg[j].lw - (u * u + w * w) / (2 * mg[j].v)); }
      return p;
    },
    // grad log P(x, t) for the mixture, in log space; writes into out
    score: function (x, y, mg, out) {
      var a = mg[0], b = mg[1];
      var u0 = x - a.mx, w0 = y - a.my, u1 = x - b.mx, w1 = y - b.my;
      var l0 = a.lw - (u0 * u0 + w0 * w0) / (2 * a.v), l1 = b.lw - (u1 * u1 + w1 * w1) / (2 * b.v);
      var top = l0 > l1 ? l0 : l1, e0 = Math.exp(l0 - top), e1 = Math.exp(l1 - top), den = e0 + e1;
      out[0] = (e0 * (a.mx - x) / a.v + e1 * (b.mx - x) / b.v) / den;
      out[1] = (e0 * (a.my - y) / a.v + e1 * (b.my - y) / b.v) / den;
    },

    // pos[(k * n + i) * 2] holds particle i at time t_k = k / steps, for both directions
    run: function (dir, p) {
      var n = p.n, S = p.steps, dt = 1 / S, md = this.modes;
      var pos = new Float32Array((S + 1) * n * 2), label = new Uint8Array(n);
      var x = new Float64Array(n), y = new Float64Array(n), sc = [0, 0], i, k, b, sq;
      if (dir === "forward") {
        var r = FIG.ctx.rng(p.seed), rn = FIG.ctx.rng(p.seed + 101);
        for (i = 0; i < n; i++) {
          label[i] = r.uniform() < md[0].w ? 0 : 1;
          x[i] = md[label[i]].mu[0] + md[label[i]].s * r.normal();
          y[i] = md[label[i]].mu[1] + md[label[i]].s * r.normal();
          pos[2 * i] = x[i]; pos[2 * i + 1] = y[i];
        }
        // forward VP SDE: dx = -1/2 beta x dt + sqrt(beta) dw
        for (k = 0; k < S; k++) {
          b = this.beta(k * dt); sq = Math.sqrt(b * dt);
          for (i = 0; i < n; i++) {
            x[i] += -0.5 * b * x[i] * dt + sq * rn.normal();
            y[i] += -0.5 * b * y[i] * dt + sq * rn.normal();
            pos[((k + 1) * n + i) * 2] = x[i]; pos[((k + 1) * n + i) * 2 + 1] = y[i];
          }
        }
      } else {
        var r2 = FIG.ctx.rng(p.seed + 1), rn2 = FIG.ctx.rng(p.seed + 202);
        for (i = 0; i < n; i++) {
          x[i] = r2.normal(); y[i] = r2.normal();
          pos[(S * n + i) * 2] = x[i]; pos[(S * n + i) * 2 + 1] = y[i];
        }
        // reverse SDE with the exact score: x_{t-dt} = x_t + [1/2 beta x + beta s] dt + sqrt(beta dt) z
        for (k = S; k > 0; k--) {
          var t = k * dt, mg = this.marginal(t);
          b = this.beta(t); sq = Math.sqrt(b * dt);
          for (i = 0; i < n; i++) {
            this.score(x[i], y[i], mg, sc);
            x[i] += (0.5 * b * x[i] + b * sc[0]) * dt + sq * rn2.normal();
            y[i] += (0.5 * b * y[i] + b * sc[1]) * dt + sq * rn2.normal();
            pos[((k - 1) * n + i) * 2] = x[i]; pos[((k - 1) * n + i) * 2 + 1] = y[i];
          }
        }
        for (i = 0; i < n; i++) {
          var e0x = x[i] - md[0].mu[0], e0y = y[i] - md[0].mu[1], e1x = x[i] - md[1].mu[0], e1y = y[i] - md[1].mu[1];
          label[i] = e0x * e0x + e0y * e0y < e1x * e1x + e1y * e1y ? 0 : 1;
        }
      }
      // per-time statistics
      var meanDist = new Float64Array(S + 1), near = [new Float64Array(S + 1), new Float64Array(S + 1)], share = new Float64Array(S + 1);
      var r0 = 4 * md[0].s * md[0].s, r1 = 4 * md[1].s * md[1].s;
      for (k = 0; k <= S; k++) {
        var sd = 0, c0 = 0, c1 = 0;
        for (i = 0; i < n; i++) {
          var px = pos[(k * n + i) * 2], py = pos[(k * n + i) * 2 + 1];
          var ax0 = px - md[0].mu[0], ay0 = py - md[0].mu[1], ax1 = px - md[1].mu[0], ay1 = py - md[1].mu[1];
          sd += Math.sqrt(px * px + py * py);
          if (ax0 * ax0 + ay0 * ay0 < r0) c0++;
          else if (ax1 * ax1 + ay1 * ay1 < r1) c1++;
        }
        meanDist[k] = sd / n; near[0][k] = c0 / n; near[1][k] = c1 / n; share[k] = (c0 + c1) / n;
      }
      var trails = [], want = [p.trails, p.trails];
      for (i = 0; i < n && (want[0] || want[1]); i++) if (want[label[i]]) { trails.push(i); want[label[i]]--; }
      return { dir: dir, n: n, steps: S, pos: pos, label: label, meanDist: meanDist, near: near, share: share, trails: trails };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, P = C.defaults, md = C.modes;
    var runs = { forward: C.run("forward", P), reverse: C.run("reverse", P) };
    var dir = "forward", t = 0;
    var off = document.createElement("canvas"), G = 80;
    off.width = G; off.height = G;
    var og = off.getContext("2d"), img = og.createImageData(G, G);

    var row = ctx.controls(stage);
    ctx.toggle(row, {
      label: "direction", value: dir,
      options: [{ value: "forward", label: "forward: add noise" }, { value: "reverse", label: "reverse: remove noise" }],
      onchange: function (v) { dir = v; anim.stop(); playBtn.textContent = "Play"; draw(); }
    });
    var playBtn = ctx.button(row, "Play", function () {
      if (anim.running) { anim.stop(); playBtn.textContent = "Play"; return; }
      if (dir === "forward" && t >= 1) t = 0;
      if (dir === "reverse" && t <= 0) t = 1;
      anim.start(); playBtn.textContent = "Pause";
    });
    playBtn.classList.add("fig__btn--primary");
    var sl = ctx.slider(row, { label: "time t", min: 0, max: 1, step: 0.002, value: t,
      fmt: function (v) { return v.toFixed(2); },
      oninput: function (v) { t = v; anim.stop(); playBtn.textContent = "Play"; draw(); } });

    var grid = ctx.el("div", "fig__row fig__row--3-2", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);
    var cv = ctx.canvas(left, { aspect: 1, minHeight: 260, maxHeight: 560,
      label: "Particles in a 2D image space moving between two data modes and a Gaussian at the origin" });
    var ts = ctx.canvas(right, { aspect: 1, minHeight: 240, maxHeight: 330,
      label: "Mean distance from the origin and share of particles near a data mode against time, for both directions" });
    var lg = ctx.el("div", "fig__legend", stage);
    function key(label, dot) { var s = ctx.el("span", null, lg); var i = ctx.el("i", dot ? "dot" : null, s); ctx.el("span", null, s, label); return i; }
    var kF = key("faces", true), kC = key("cars", true), kN = key("2 standard deviations of N(0, I)"),
        kFw = key("forward run"), kRv = key("reverse run");
    var ro = ctx.readout(stage);

    var anim = ctx.loop(stage, function (dtSec) {
      t += (dir === "forward" ? 1 : -1) * 0.25 * dtSec;
      var done = dir === "forward" ? t >= 1 : t <= 0;
      t = Math.max(0, Math.min(1, t));
      sl.set(t);
      draw();
      if (done) { playBtn.textContent = "Play"; return false; }
    });

    function rgbOf(c) {
      c = c.trim();
      if (c[0] === "#") { var h = c.slice(1); if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2]; var n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
      var m = c.match(/[\d.]+/g) || [0, 0, 0]; return [+m[0], +m[1], +m[2]];
    }

    function draw() {
      var col = ctx.colors(), g = cv.g, R = runs[dir], n = R.n, S = R.steps;
      var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
      var k = Math.round(t * S), tk = k / S;
      cv.clear();
      // equal-aspect box: y spans [-4.2, 4.2], x spans whatever keeps circles round
      var m = { l: 30, r: 10, t: 10, b: 26 }, bw = cv.w - m.l - m.r, bh = cv.h - m.t - m.b;
      var half = 4.2, hx = half * bw / bh;
      var ax = ctx.axes(cv, { box: { x: m.l, y: m.t, w: bw, h: bh }, xmin: -hx, xmax: hx, ymin: -half, ymax: half,
        xticks: [-4, -2, 0, 2, 4].filter(function (v) { return Math.abs(v) <= hx; }), yticks: [-4, -2, 0, 2, 4] });

      // exact P(x, t) as shading
      var mg = C.marginal(tk), rgb = rgbOf(col.ink), vals = new Float64Array(G * G), mx = 0, a, b;
      for (b = 0; b < G; b++) for (a = 0; a < G; a++) {
        var v = C.density(-hx + (a + 0.5) / G * 2 * hx, half - (b + 0.5) / G * 2 * half, mg);
        vals[b * G + a] = v; if (v > mx) mx = v;
      }
      for (a = 0; a < G * G; a++) {
        img.data[4 * a] = rgb[0]; img.data[4 * a + 1] = rgb[1]; img.data[4 * a + 2] = rgb[2];
        img.data[4 * a + 3] = Math.round(255 * 0.2 * Math.sqrt(vals[a] / mx));
      }
      og.putImageData(img, 0, 0);
      g.save(); g.imageSmoothingEnabled = true; g.drawImage(off, ax.box.x, ax.box.y, bw, bh); g.restore();
      ax.draw();

      g.save();
      g.beginPath(); g.rect(ax.box.x, ax.box.y, bw, bh); g.clip();
      var scale = bw / (2 * hx);
      // the noise distribution: 2 standard deviations of N(0, I)
      g.setLineDash([5, 4]); g.strokeStyle = col.muted; g.lineWidth = 1.2;
      g.beginPath(); g.arc(ax.X(0), ax.Y(0), 2 * scale, 0, 2 * Math.PI); g.stroke();
      g.setLineDash([]);
      // the data modes: 2 standard deviations of each
      md.forEach(function (d, j) {
        g.strokeStyle = col.c[j]; g.lineWidth = 1.4;
        g.beginPath(); g.arc(ax.X(d.mu[0]), ax.Y(d.mu[1]), 2 * d.s * scale, 0, 2 * Math.PI); g.stroke();
      });
      // trails from where the run starts to the current time
      R.trails.forEach(function (i) {
        g.beginPath();
        var k0 = dir === "forward" ? 0 : S, step = dir === "forward" ? 1 : -1;
        g.moveTo(ax.X(R.pos[(k0 * n + i) * 2]), ax.Y(R.pos[(k0 * n + i) * 2 + 1]));
        for (var kk = k0 + step; dir === "forward" ? kk <= k : kk >= k; kk += step) g.lineTo(ax.X(R.pos[(kk * n + i) * 2]), ax.Y(R.pos[(kk * n + i) * 2 + 1]));
        g.strokeStyle = ctx.alpha(col.c[R.label[i]], 0.85); g.lineWidth = 1.1; g.stroke();
      });
      // particles
      [0, 1].forEach(function (j) {
        g.fillStyle = ctx.alpha(col.c[j], 0.75);
        g.beginPath();
        for (var i = 0; i < n; i++) {
          if (R.label[i] !== j) continue;
          var px = ax.X(R.pos[(k * n + i) * 2]), py = ax.Y(R.pos[(k * n + i) * 2 + 1]);
          g.moveTo(px + 1.9, py); g.arc(px, py, 1.9, 0, 2 * Math.PI);
        }
        g.fill();
      });
      g.restore();
      // labels
      // each mode's name sits diagonally outward from its circle
      g.font = "12px " + sans;
      md.forEach(function (d, j) {
        var sx = d.mu[0] > 0 ? 1 : -1, sy = d.mu[1] > 0 ? 1 : -1, rr = (3.3 * d.s * scale + 4) * 0.72;
        g.fillStyle = col.c[j];
        g.textAlign = sx > 0 ? "left" : "right";
        g.textBaseline = sy > 0 ? "bottom" : "top";
        g.fillText(d.name, ax.X(d.mu[0]) + sx * rr, ax.Y(d.mu[1]) - sy * rr);
      });
      g.fillStyle = col.muted; g.textAlign = "center"; g.textBaseline = "top";
      g.fillText("noise", ax.X(0), ax.Y(0) + 2 * scale + 4);

      drawSeries(col, sans, k);

      kF.style.background = col.c[0]; kC.style.background = col.c[1];
      kN.style.background = col.muted; kFw.style.background = col.c[2]; kRv.style.background = col.c[4];
      ro.set({
        "time t": ctx.fmt(tk, 2),
        "mean distance from origin": ctx.fmt(R.meanDist[k], 2),
        "within 2s of a mode": ctx.fmt(R.share[k], 2),
        "near faces": ctx.fmt(R.near[0][k], 2),
        "near cars": ctx.fmt(R.near[1][k], 2)
      });
    }

    function drawSeries(col, sans, k) {
      var g = ts.g, W = ts.w, H = ts.h, top = 18, gap = 40, bottom = 30, S = P.steps;
      ts.clear();
      var h1 = Math.floor((H - top - gap - bottom) / 2);
      var panels = [
        { key: "meanDist", title: "mean distance from origin", ymin: 0, ymax: 3.5, yticks: [0, 1, 2, 3], y: top },
        { key: "share", title: "share within 2s of a mode", ymin: 0, ymax: 1, yticks: [0, 0.5, 1], y: top + h1 + gap }
      ];
      panels.forEach(function (pn, pi) {
        var ax = ctx.axes(ts, { box: { x: 34, y: pn.y, w: W - 34 - 10, h: h1 }, xmin: 0, xmax: 1, ymin: pn.ymin, ymax: pn.ymax,
          yticks: pn.yticks, xticks: [0, 0.5, 1], xlabel: pi === 1 ? "time t" : null });
        ax.draw();
        g.save();
        g.beginPath(); g.rect(ax.box.x, ax.box.y, ax.box.w, ax.box.h); g.clip();
        ["forward", "reverse"].forEach(function (d) {
          var arr = runs[d][pn.key];
          g.beginPath();
          for (var kk = 0; kk <= S; kk += 2) { if (kk) g.lineTo(ax.X(kk / S), ax.Y(arr[kk])); else g.moveTo(ax.X(0), ax.Y(arr[0])); }
          g.strokeStyle = d === "forward" ? col.c[2] : col.c[4];
          g.lineWidth = d === dir ? 2.4 : 1.3;
          g.globalAlpha = d === dir ? 1 : 0.7;
          g.stroke();
          g.globalAlpha = 1;
        });
        var cx = ax.X(k / S);
        g.strokeStyle = col.soft; g.lineWidth = 1; g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(cx, ax.box.y); g.lineTo(cx, ax.box.y + ax.box.h); g.stroke(); g.setLineDash([]);
        g.fillStyle = dir === "forward" ? col.c[2] : col.c[4];
        g.beginPath(); g.arc(cx, ax.Y(runs[dir][pn.key][k]), 3.5, 0, 2 * Math.PI); g.fill();
        g.restore();
        g.font = "11px " + sans; g.fillStyle = col.soft; g.textAlign = "left"; g.textBaseline = "bottom";
        g.fillText(pn.title, ax.box.x, ax.box.y - 4);
      });
    }

    cv.onresize = draw;
    ts.onresize = draw;
    draw();
    return { draw: draw };
  }
});
