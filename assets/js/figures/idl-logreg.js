FIG.register("idl-logreg", {
  defaults: { seed: 3, perClass: 30, alpha: 1, kind: "separable", maxIter: 100000 },

  // Pure math, no DOM. Notation follows the book: h = sigma(theta0 + theta1 x1 + theta2 x2),
  // J = -(1/m) sum [ y log h + (1 - y) log(1 - h) ], parameters start at 0.
  compute: {
    // Two long clusters lying side by side, offset along their length, so the first
    // gradient points the wrong way and the boundary has to turn. "separable" drops
    // every point within 0.1 of the dividing line; "overlapping" is wider and keeps all.
    data: function (kind, seed, perClass) {
      var sep = kind === "separable";
      var r = FIG.ctx.rng(sep ? seed : seed + 50);
      var ang = 0.5, v = [Math.cos(ang), Math.sin(ang)], n = [-v[1], v[0]];
      var W = sep ? 0.35 : 0.6, m = 2 * perClass;
      var x1 = new Float64Array(m), x2 = new Float64Array(m), y = new Float64Array(m), k = 0;
      for (var cls = 0; cls < 2; cls++) {
        var s = cls === 1 ? 1 : -1, got = 0;
        while (got < perClass) {
          var a = r.normal(), b = W * r.normal();
          var p1 = s * (0.45 * n[0] + 0.7 * v[0]) + a * v[0] + b * n[0];
          var p2 = s * (0.45 * n[1] + 0.7 * v[1]) + a * v[1] + b * n[1];
          if (sep && s * (n[0] * p1 + n[1] * p2) < 0.1) continue;
          if (Math.abs(p1) > 2.6 || Math.abs(p2) > 2.6) continue;
          x1[k] = p1; x2[k] = p2; y[k] = cls; k++; got++;
        }
      }
      return { x1: x1, x2: x2, y: y, m: m };
    },

    sigmoid: function (z) { return 1 / (1 + Math.exp(-z)); },

    // log(1 + e^t) without overflow. The cross entropy of one example is
    // softplus(-z) when y = 1 and softplus(z) when y = 0, the same as the book's formula,
    // written so that h = 1 in floating point never takes log(0).
    softplus: function (t) { return Math.max(t, 0) + Math.log(1 + Math.exp(-Math.abs(t))); },

    cost: function (d, th) {
      var s = 0;
      for (var i = 0; i < d.m; i++) {
        var z = th[0] + th[1] * d.x1[i] + th[2] * d.x2[i];
        s += d.y[i] === 1 ? this.softplus(-z) : this.softplus(z);
      }
      return s / d.m;
    },

    accuracy: function (d, th) {
      var c = 0;
      for (var i = 0; i < d.m; i++) {
        var z = th[0] + th[1] * d.x1[i] + th[2] * d.x2[i];
        if ((z >= 0 ? 1 : 0) === d.y[i]) c++;     // h >= 0.5 exactly when z >= 0
      }
      return c / d.m;
    },

    norm: function (th) { return Math.sqrt(th[0] * th[0] + th[1] * th[1] + th[2] * th[2]); },

    // One gradient descent step in place; dJ/dtheta_j = (1/m) sum (h_i - y_i) x_ij with x_i0 = 1.
    // Returns true when every point was on the right side before the step.
    step: function (d, th, alpha) {
      var g0 = 0, g1 = 0, g2 = 0, correct = 0, m = d.m, x1 = d.x1, x2 = d.x2, y = d.y;
      var t0 = th[0], t1 = th[1], t2 = th[2];
      for (var i = 0; i < m; i++) {
        var z = t0 + t1 * x1[i] + t2 * x2[i];
        if ((z >= 0 ? 1 : 0) === y[i]) correct++;
        var e = 1 / (1 + Math.exp(-z)) - y[i];
        g0 += e; g1 += e * x1[i]; g2 += e * x2[i];
      }
      th[0] = t0 - alpha * g0 / m; th[1] = t1 - alpha * g1 / m; th[2] = t2 - alpha * g2 / m;
      return correct === m;
    },

    // Iterations to record: every one up to 100, then 3% apart, plus each power of 10.
    checkpoints: function (maxIter) {
      var out = [], k = 0;
      while (k <= Math.min(100, maxIter)) out.push(k++);
      var v = 100, p = 1000;
      while (true) {
        v = Math.max(v + 1, Math.round(v * 1.03));
        while (p < v && p < maxIter) { out.push(p); p *= 10; }
        if (v >= maxIter) break;
        out.push(v);
      }
      out.push(maxIter);
      return out;
    },

    // Run from theta = 0 and record the state at each checkpoint.
    run: function (d, alpha, maxIter) {
      var cps = this.checkpoints(maxIter), th = [0, 0, 0], rec = [], ci = 0, firstAll = -1;
      for (var it = 0; it <= maxIter; it++) {
        if (it === cps[ci]) {
          rec.push({ iter: it, th: th.slice(), J: this.cost(d, th), acc: this.accuracy(d, th), norm: this.norm(th) });
          ci++;
        }
        if (it === maxIter) break;
        var all = this.step(d, th, alpha);
        if (all && firstAll < 0) firstAll = it;
      }
      return { rec: rec, firstAll: firstAll };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var st = { kind: D.kind, alpha: D.alpha, data: null, run: null, idx: 0, acc: 0 };

    var row = ctx.controls(stage);
    var sA = ctx.slider(row, {
      label: "learning rate α", min: -2, max: 1, step: 0.01, value: Math.log10(D.alpha),
      fmt: function (u) { var a = Math.pow(10, u); return a < 0.1 ? a.toFixed(3) : a < 1 ? a.toFixed(2) : a.toFixed(1); },
      oninput: function (u) {
        // each rebuild runs 100,000 steps, so wait until the slider stops moving
        st.alpha = Math.pow(10, u);
        clearTimeout(st.timer);
        st.timer = setTimeout(function () { rebuild(true); }, 140);
      }
    });
    var tg = ctx.toggle(row, {
      label: "data", options: [{ value: "overlapping", label: "overlapping" }, { value: "separable", label: "separable" }],
      value: D.kind, onchange: function (v) { st.kind = v; rebuild(true); }
    });
    var bPlay = ctx.button(row, "Play", function () { togglePlay(); });
    bPlay.classList.add("fig__btn--primary");
    ctx.button(row, "Step", function () { anim.stop(); if (st.idx < st.run.rec.length - 1) st.idx++; syncButton(); draw(); });
    ctx.button(row, "Reset", function () {
      st.alpha = D.alpha; sA.set(Math.log10(D.alpha)); st.kind = D.kind; tg.set(D.kind); rebuild(false);
    });

    var grid = ctx.el("div", "fig__row fig__row--3-2", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);
    var cvP = ctx.canvas(left, { aspect: 0.9, minHeight: 260, maxHeight: 420, label: "Two classes of points, the decision boundary h = 0.5 and the probability shading" });
    // side by side the history matches the plane's height; stacked on a phone it is shorter
    var optH = { aspect: 1.35, minHeight: 260, maxHeight: 420, label: "Cost J and the size of theta against iteration, on a log scale" };
    var cvH = ctx.canvas(right, optH);
    var legend = ctx.el("div", "fig__legend", stage);
    var ro = ctx.readout(stage);

    var R = 2.8;   // plane is [-R, R]^2

    function rebuild(andPlay) {
      anim.stop();
      st.data = C.data(st.kind, D.seed, D.perClass);
      st.run = C.run(st.data, st.alpha, D.maxIter);
      st.idx = 0; st.acc = 0;
      st.normMax = 0;
      st.run.rec.forEach(function (r) { st.normMax = Math.max(st.normMax, r.norm); });
      draw();
      if (andPlay) play(); else syncButton();
    }

    var anim = ctx.loop(stage, function (dt) {
      st.acc += dt * 60;                     // one checkpoint per 60th of a second
      var n = Math.floor(st.acc); st.acc -= n;
      st.idx = Math.min(st.run.rec.length - 1, st.idx + n);
      draw();
      if (st.idx >= st.run.rec.length - 1) { syncButton(true); return false; }
    });
    function syncButton(stopped) { bPlay.textContent = (anim.running && !stopped) ? "Pause" : "Play"; }
    function play() {
      if (ctx.reducedMotion) { st.idx = st.run.rec.length - 1; draw(); syncButton(true); return; }
      if (st.idx >= st.run.rec.length - 1) st.idx = 0;
      st.acc = 0; anim.start(); syncButton();
    }
    function togglePlay() { if (anim.running) { anim.stop(); syncButton(); } else play(); }

    function rgbOf(c) { return (ctx.alpha(c, 1).match(/[\d.]+/g) || [0, 0, 0]).slice(0, 3).map(Number); }

    var shade = document.createElement("canvas"), N = 72;
    shade.width = N; shade.height = N;

    function drawPlane(rec, col) {
      var g = cvP.g;
      cvP.clear();
      var ax = ctx.axes(cvP, { xmin: -R, xmax: R, ymin: -R, ymax: R, xlabel: "x₁", ylabel: "x₂", margin: { l: 34, r: 10, t: 10, b: 32 } });
      // keep the plane square
      var side = Math.min(ax.box.w, ax.box.h);
      ax.box.x += (ax.box.w - side) / 2; ax.box.w = side; ax.box.h = side;
      var b = ax.box, th = rec.th;
      // probability shading h(x) on a coarse grid, scaled up smoothly
      var sg = shade.getContext("2d"), im = sg.createImageData(N, N);
      var c0 = rgbOf(col.c[0]), c1 = rgbOf(col.c[1]);
      for (var j = 0; j < N; j++) {
        for (var i = 0; i < N; i++) {
          var x1 = -R + 2 * R * (i + 0.5) / N, x2 = R - 2 * R * (j + 0.5) / N;
          var h = C.sigmoid(th[0] + th[1] * x1 + th[2] * x2);
          var cc = h >= 0.5 ? c1 : c0, a = Math.abs(2 * h - 1);
          var o = 4 * (j * N + i);
          im.data[o] = cc[0]; im.data[o + 1] = cc[1]; im.data[o + 2] = cc[2];
          im.data[o + 3] = Math.round(255 * 0.32 * a);
        }
      }
      sg.putImageData(im, 0, 0);
      g.imageSmoothingEnabled = true;
      g.drawImage(shade, b.x, b.y, b.w, b.h);
      ax.draw();
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      // boundary h = 0.5 and the 0.1 / 0.9 lines
      function levelLine(z, dash, width, color) {
        var w1 = th[1], w2 = th[2];
        if (Math.abs(w1) + Math.abs(w2) < 1e-9) return;
        g.setLineDash(dash); g.lineWidth = width; g.strokeStyle = color;
        g.beginPath();
        if (Math.abs(w2) > Math.abs(w1)) {
          g.moveTo(ax.X(-R), ax.Y((z - th[0] - w1 * -R) / w2)); g.lineTo(ax.X(R), ax.Y((z - th[0] - w1 * R) / w2));
        } else {
          g.moveTo(ax.X((z - th[0] - w2 * -R) / w1), ax.Y(-R)); g.lineTo(ax.X((z - th[0] - w2 * R) / w1), ax.Y(R));
        }
        g.stroke();
        g.setLineDash([]);
      }
      var L9 = Math.log(9);
      levelLine(L9, [3, 4], 1, ctx.alpha(col.ink, 0.45));
      levelLine(-L9, [3, 4], 1, ctx.alpha(col.ink, 0.45));
      levelLine(0, [], 2, col.ink);
      // points: class 1 circles, class 0 squares; wrong side gets a ring
      var d = st.data;
      for (var k = 0; k < d.m; k++) {
        var X = ax.X(d.x1[k]), Y = ax.Y(d.x2[k]);
        var hk = C.sigmoid(th[0] + th[1] * d.x1[k] + th[2] * d.x2[k]);
        var wrong = (hk >= 0.5 ? 1 : 0) !== d.y[k];
        g.fillStyle = d.y[k] === 1 ? col.c[1] : col.c[0];
        g.strokeStyle = col.paper; g.lineWidth = 1;
        g.beginPath();
        if (d.y[k] === 1) g.arc(X, Y, 4, 0, 7); else g.rect(X - 3.6, Y - 3.6, 7.2, 7.2);
        g.fill(); g.stroke();
        if (wrong && rec.iter > 0) {
          g.strokeStyle = col.ink; g.lineWidth = 1.4;
          g.beginPath(); g.arc(X, Y, 7.5, 0, 7); g.stroke();
        }
      }
      g.restore();
      if (rec.iter === 0) {
        g.font = "12px " + sans; g.fillStyle = col.soft; g.textAlign = "center"; g.textBaseline = "top";
        g.fillText("θ = 0: every point gets h = 0.5", b.x + b.w / 2, b.y + 8);
      }
    }

    function drawHistory(col) {
      var g = cvH.g, rec = st.run.rec, cur = st.idx;
      cvH.clear();
      var L = Math.log10(D.maxIter);
      var ml = 40, mr = 16, gap = 30, top = 18, bot = 32;
      var ph = (cvH.h - top - bot - gap) / 2;
      var xt = [], xl = {};
      for (var e = 0; e <= L; e++) { xt.push(e); xl[e] = e === 0 ? "1" : e < 3 ? String(Math.pow(10, e)) : Math.pow(10, e - 3) + "k"; }
      var narrow = cvH.w < 260;
      if (narrow) xt = xt.filter(function (e) { return e % 2 === 0 || e === L; });
      var fx = function (e) { return xl[e] || ""; };
      var axJ = ctx.axes(cvH, { xmin: 0, xmax: L, ymin: 0, ymax: 0.75, box: { x: ml, y: top, w: cvH.w - ml - mr, h: ph },
        xticks: xt, yticks: [0, 0.25, 0.5, 0.75], xfmt: function () { return ""; } });
      var nmax = Math.max(1, st.normMax * 1.1);
      var axN = ctx.axes(cvH, { xmin: 0, xmax: L, ymin: 0, ymax: nmax, box: { x: ml, y: top + ph + gap, w: cvH.w - ml - mr, h: ph },
        xticks: xt, xfmt: fx, xlabel: "iteration (log scale)" });
      axJ.draw(); axN.draw();
      g.font = "11px " + sans; g.fillStyle = col.soft; g.textAlign = "left"; g.textBaseline = "bottom";
      g.fillText("cost J", ml, top - 4);
      g.fillText("|θ|", ml, top + ph + gap - 4);
      // all points correct marker
      if (st.run.firstAll >= 0) {
        var fe = Math.log10(Math.max(1, st.run.firstAll));
        var reached = rec[cur].iter >= st.run.firstAll;
        g.strokeStyle = ctx.alpha(col.c[2], reached ? 0.9 : 0.35); g.setLineDash([3, 3]); g.lineWidth = 1.2;
        g.beginPath(); g.moveTo(axJ.X(fe), top); g.lineTo(axJ.X(fe), top + 2 * ph + gap); g.stroke(); g.setLineDash([]);
        if (reached) {
          g.fillStyle = col.c[2]; g.textAlign = fe > L * 0.6 ? "right" : "left"; g.textBaseline = "top";
          var tx = axJ.X(fe) + (fe > L * 0.6 ? -4 : 4);
          g.fillText("all points", tx, top + 4); g.fillText("correct", tx, top + 17);
        }
      }
      function series(ax, key, color) {
        g.strokeStyle = color; g.lineWidth = 1.8; g.beginPath();
        var started = false;
        for (var i = 1; i <= cur; i++) {
          var X = ax.X(Math.log10(rec[i].iter)), Y = ax.Y(rec[i][key]);
          if (!started) { g.moveTo(X, Y); started = true; } else g.lineTo(X, Y);
        }
        g.stroke();
        if (cur >= 1) { g.fillStyle = color; g.beginPath(); g.arc(ax.X(Math.log10(rec[cur].iter)), ax.Y(rec[cur][key]), 3.2, 0, 7); g.fill(); }
      }
      series(axJ, "J", col.c[3]);
      series(axN, "norm", col.c[4]);
    }

    function draw() {
      var aspect = right.offsetTop > left.offsetTop + 10 ? 0.8 : 1.35;
      if (optH.aspect !== aspect) { optH.aspect = aspect; cvH.resize(); }
      var col = ctx.colors(), rec = st.run.rec[st.idx];
      drawPlane(rec, col);
      drawHistory(col);
      var fa = st.run.firstAll;
      ro.set({
        "iteration": String(rec.iter),
        "cost J": ctx.fmt(rec.J, 4),
        "accuracy": (100 * rec.acc).toFixed(1) + "%",
        "|θ|": ctx.fmt(rec.norm, 2),
        "all correct": fa < 0 ? "never" : "from " + fa
      });
    }

    function paintLegend() {
      var col = ctx.colors();
      legend.innerHTML = "";
      [[col.c[0], "y = 0", "sq"], [col.c[1], "y = 1", "dot"], [col.ink, "boundary h = 0.5", ""], [ctx.alpha(col.ink, 0.45), "h = 0.1 and 0.9", ""]].forEach(function (s) {
        var sp = ctx.el("span", null, legend), ic = ctx.el("i", s[2] === "dot" ? "dot" : null, sp);
        ic.style.background = s[0];
        if (s[2] === "sq") { ic.style.width = ".55rem"; ic.style.height = ".55rem"; }
        sp.appendChild(document.createTextNode(s[1]));
      });
    }

    cvP.onresize = draw; cvH.onresize = draw;
    paintLegend();
    rebuild(!ctx.reducedMotion);
    if (ctx.reducedMotion) { st.idx = st.run.rec.length - 1; draw(); syncButton(true); }
    return { draw: function () { paintLegend(); draw(); } };
  }
});
