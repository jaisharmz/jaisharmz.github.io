FIG.register("idl-linreg", {
  defaults: { seed: 1, m: 20, alpha: 0.3, start: [0, 0], maxIter: 3000 },

  // Pure math, no DOM. Notation follows the book: h(x) = theta0 + theta1 x,
  // J = 1/(2m) sum (y_i - h(x_i))^2, theta_i := theta_i - alpha dJ/dtheta_i.
  compute: {
    data: function (seed, m) {
      var r = FIG.ctx.rng(seed), x = [], y = [];
      for (var i = 0; i < m; i++) {
        var xi = 2 * r.uniform();
        x.push(xi);
        y.push(1 + 1.5 * xi + 0.4 * r.normal());
      }
      return { x: x, y: y, m: m };
    },

    cost: function (d, t0, t1) {
      var s = 0;
      for (var i = 0; i < d.m; i++) { var e = d.y[i] - (t0 + t1 * d.x[i]); s += e * e; }
      return s / (2 * d.m);
    },

    grad: function (d, t0, t1) {
      var g0 = 0, g1 = 0;
      for (var i = 0; i < d.m; i++) {
        var e = (t0 + t1 * d.x[i]) - d.y[i];
        g0 += e; g1 += e * d.x[i];
      }
      return [g0 / d.m, g1 / d.m];
    },

    step: function (d, t0, t1, alpha) {
      var g = this.grad(d, t0, t1);
      return [t0 - alpha * g[0], t1 - alpha * g[1]];
    },

    // Closed-form least squares for one feature (the normal equations).
    optimum: function (d) {
      var mx = 0, my = 0, i;
      for (i = 0; i < d.m; i++) { mx += d.x[i]; my += d.y[i]; }
      mx /= d.m; my /= d.m;
      var sxy = 0, sxx = 0;
      for (i = 0; i < d.m; i++) { sxy += (d.x[i] - mx) * (d.y[i] - my); sxx += (d.x[i] - mx) * (d.x[i] - mx); }
      var t1 = sxy / sxx, t0 = my - t1 * mx;
      return { t0: t0, t1: t1, J: this.cost(d, t0, t1) };
    },

    // J is a quadratic bowl with curvature matrix (1/m) sum [1 x; x x^2].
    // Gradient descent shrinks every error once alpha < 2 / (largest curvature).
    alphaLimit: function (d) {
      var a = 1, b = 0, c = 0;
      for (var i = 0; i < d.m; i++) { b += d.x[i]; c += d.x[i] * d.x[i]; }
      b /= d.m; c /= d.m;
      var tr = a + c, det = a * c - b * b;
      var lmax = tr / 2 + Math.sqrt(tr * tr / 4 - det);
      var lmin = tr / 2 - Math.sqrt(tr * tr / 4 - det);
      return { limit: 2 / lmax, lmax: lmax, lmin: lmin };
    },

    // Run gradient descent until it settles, blows up, or hits maxIter.
    run: function (d, start, alpha, maxIter) {
      var t0 = start[0], t1 = start[1], path = [[t0, t1]], status = "running";
      for (var k = 0; k < maxIter; k++) {
        var n = this.step(d, t0, t1, alpha);
        var moved = Math.abs(n[0] - t0) + Math.abs(n[1] - t1);
        t0 = n[0]; t1 = n[1];
        path.push([t0, t1]);
        var J = this.cost(d, t0, t1);
        if (!isFinite(J) || J > 1e6) { status = "diverged"; break; }
        if (moved < 1e-9) { status = "settled"; break; }
      }
      return { path: path, status: status, t0: t0, t1: t1, J: this.cost(d, t0, t1), iter: path.length - 1 };
    },

    // First iteration at which (theta0, theta1) is within tol of the least-squares values.
    itersToReach: function (d, start, alpha, tol, maxIter) {
      var o = this.optimum(d), t0 = start[0], t1 = start[1];
      for (var k = 0; k <= maxIter; k++) {
        if (Math.sqrt((t0 - o.t0) * (t0 - o.t0) + (t1 - o.t1) * (t1 - o.t1)) < tol) return k;
        var n = this.step(d, t0, t1, alpha); t0 = n[0]; t1 = n[1];
        if (!isFinite(t0) || Math.abs(t0) > 1e8) return -1;
      }
      return -1;
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var data = C.data(D.seed, D.m);
    var opt = C.optimum(data);
    var lim = C.alphaLimit(data);
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");

    var state = { start: D.start.slice(), alpha: D.alpha, path: [D.start.slice()], status: "ready", acc: 0 };

    var row = ctx.controls(stage);
    var sA = ctx.slider(row, {
      label: "learning rate α", min: 0.01, max: 1.0, step: 0.01, value: D.alpha,
      fmt: function (v) { return v.toFixed(2); },
      oninput: function (v) { state.alpha = v; restart(true); }
    });
    var bPlay = ctx.button(row, "Play", function () { toggle(); });
    bPlay.classList.add("fig__btn--primary");
    ctx.button(row, "Step", function () { anim.stop(); advance(1); syncButton(); draw(); });
    ctx.button(row, "Reset", function () {
      state.start = D.start.slice(); state.alpha = D.alpha; sA.set(D.alpha); restart(false);
    });

    var grid = ctx.el("div", "fig__row", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);
    var cvL = ctx.canvas(left, { aspect: 0.82, minHeight: 230, maxHeight: 360, label: "Data points, the current line h(x) and its residuals" });
    var cvR = ctx.canvas(right, { aspect: 0.82, minHeight: 230, maxHeight: 360, label: "Contours of the cost J over theta0 and theta1 with the gradient descent path" });
    var legend = ctx.el("div", "fig__legend", stage);
    var ro = ctx.readout(stage);

    function cur() { return state.path[state.path.length - 1]; }

    // ranges for the parameter plane, centred on the least-squares optimum
    var P = { t0min: opt.t0 - 2.6, t0max: opt.t0 + 2.6, t1min: opt.t1 - 2.2, t1max: opt.t1 + 2.2 };
    var xmin = 0, xmax = 2, ymin = Infinity, ymax = -Infinity;
    data.y.forEach(function (v) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); });
    ymin = Math.floor(ymin - 0.6); ymax = Math.ceil(ymax + 0.6);

    function advance(n) {
      for (var k = 0; k < n; k++) {
        if (state.status === "diverged" || state.status === "settled") return;
        if (state.path.length - 1 >= D.maxIter) { state.status = "stopped"; return; }
        var p = cur(), q = C.step(data, p[0], p[1], state.alpha);
        state.path.push(q);
        var J = C.cost(data, q[0], q[1]);
        if (!isFinite(J) || J > 1e6) { state.status = "diverged"; return; }
        if (Math.abs(q[0] - p[0]) + Math.abs(q[1] - p[1]) < 1e-9) { state.status = "settled"; return; }
        state.status = "running";
      }
    }

    var anim = ctx.loop(stage, function (dt) {
      var it = state.path.length - 1;
      var rate = Math.min(160, 6 + it * 1.2);           // steps per second, speeding up
      state.acc += dt * rate;
      var n = Math.floor(state.acc);
      state.acc -= n;
      advance(n);
      draw();
      if (state.status === "diverged" || state.status === "settled" || state.status === "stopped") { syncButton(true); return false; }
    });

    function syncButton(stopped) {
      bPlay.textContent = (anim.running && !stopped) ? "Pause" : "Play";
    }
    function toggle() {
      if (anim.running) { anim.stop(); syncButton(); return; }
      if (state.status === "diverged" || state.status === "settled" || state.status === "stopped") restart(false);
      play();
    }
    function play() {
      if (ctx.reducedMotion) { advance(D.maxIter); draw(); syncButton(true); return; }
      state.acc = 0; anim.start(); syncButton();
    }
    function restart(andPlay) {
      anim.stop();
      state.path = [state.start.slice()];
      state.status = "ready";
      draw();
      if (andPlay) play(); else syncButton();
    }

    // click on the contour plot: start from there
    cvR.el.addEventListener("pointerdown", function (e) {
      var r = cvR.el.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      var b = axR && axR.box;
      if (!b || px < b.x || px > b.x + b.w || py < b.y || py > b.y + b.h) return;
      state.start = [axR.invX(px), axR.invY(py)];
      restart(true);
    });
    cvR.el.style.cursor = "crosshair";

    // contour image cache
    var cache = { key: "", img: null };
    function contourLayer(ax, col) {
      var key = [cvR.w, cvR.h, cvR.dpr, col.muted, col.paper].join("|");
      if (cache.key === key) return cache.img;
      var off = document.createElement("canvas");
      off.width = Math.round(cvR.w * cvR.dpr); off.height = Math.round(cvR.h * cvR.dpr);
      var g = off.getContext("2d");
      g.setTransform(cvR.dpr, 0, 0, cvR.dpr, 0, 0);
      var nx = 90, ny = 90, b = ax.box, vals = [];
      for (var j = 0; j <= ny; j++) {
        var rowv = [];
        for (var i = 0; i <= nx; i++) {
          var t0 = P.t0min + (P.t0max - P.t0min) * i / nx;
          var t1 = P.t1min + (P.t1max - P.t1min) * j / ny;
          rowv.push(C.cost(data, t0, t1));
        }
        vals.push(rowv);
      }
      // soft fill, darker toward the minimum: one small image scaled up smoothly
      var small = document.createElement("canvas");
      small.width = nx + 1; small.height = ny + 1;
      var sg = small.getContext("2d"), im = sg.createImageData(nx + 1, ny + 1);
      var rgb = (ctx.alpha(col.c[0], 1).match(/[\d.]+/g) || [0, 0, 0]).map(Number);
      for (var j2 = 0; j2 <= ny; j2++) {
        for (var i2 = 0; i2 <= nx; i2++) {
          var v = Math.max(0, vals[j2][i2] - opt.J);
          var s = Math.max(0, 1 - Math.log(1 + v / 0.02) / Math.log(1 + 40 / 0.02));
          var o = 4 * ((ny - j2) * (nx + 1) + i2);
          im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2];
          im.data[o + 3] = Math.round(255 * (0.03 + 0.2 * s * s));
        }
      }
      sg.putImageData(im, 0, 0);
      g.imageSmoothingEnabled = true;
      g.drawImage(small, b.x, b.y, b.w, b.h);
      g.strokeStyle = ctx.alpha(col.muted, 0.55);
      g.lineWidth = 1;
      for (var k = 0; k < 11; k++) {
        var level = opt.J + 0.02 * Math.pow(2, k);
        marching(g, vals, nx, ny, level, function (fi, fj) { return [b.x + b.w * fi / nx, b.y + b.h - b.h * fj / ny]; });
      }
      cache.key = key; cache.img = off;
      return off;
    }

    // marching squares for one level; map(i, j) turns fractional grid indices into px
    function marching(g, v, nx, ny, L, map) {
      g.beginPath();
      for (var j = 0; j < ny; j++) {
        for (var i = 0; i < nx; i++) {
          var a = v[j][i], b = v[j][i + 1], c = v[j + 1][i + 1], d = v[j + 1][i];
          var pts = [];
          if ((a < L) !== (b < L)) pts.push(map(i + (L - a) / (b - a), j));
          if ((b < L) !== (c < L)) pts.push(map(i + 1, j + (L - b) / (c - b)));
          if ((c < L) !== (d < L)) pts.push(map(i + 1 - (L - c) / (d - c), j + 1));
          if ((d < L) !== (a < L)) pts.push(map(i, j + 1 - (L - d) / (a - d)));
          for (var p = 0; p + 1 < pts.length; p += 2) { g.moveTo(pts[p][0], pts[p][1]); g.lineTo(pts[p + 1][0], pts[p + 1][1]); }
        }
      }
      g.stroke();
    }

    var axR = null;
    function draw() {
      var col = ctx.colors();
      var p = cur(), J = C.cost(data, p[0], p[1]);
      var narrow = cvL.w < 300;

      // ----- left: data, line, residuals
      cvL.clear();
      var g = cvL.g;
      var axL = ctx.axes(cvL, { xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax, xlabel: "x", ylabel: "y",
        margin: { l: 36, r: 10, t: 10, b: 32 } });
      axL.draw();
      var bL = axL.box;
      g.save();
      g.beginPath(); g.rect(bL.x, bL.y, bL.w, bL.h); g.clip();
      g.strokeStyle = ctx.alpha(col.c[3], 0.75); g.lineWidth = 1.2;
      g.beginPath();
      for (var i = 0; i < data.m; i++) {
        var hx = p[0] + p[1] * data.x[i];
        g.moveTo(axL.X(data.x[i]), axL.Y(data.y[i])); g.lineTo(axL.X(data.x[i]), axL.Y(hx));
      }
      g.stroke();
      // least-squares line, faint, for reference
      g.setLineDash([4, 4]); g.strokeStyle = ctx.alpha(col.ink, 0.35); g.lineWidth = 1;
      g.beginPath(); g.moveTo(axL.X(xmin), axL.Y(opt.t0 + opt.t1 * xmin)); g.lineTo(axL.X(xmax), axL.Y(opt.t0 + opt.t1 * xmax)); g.stroke();
      g.setLineDash([]);
      g.strokeStyle = col.c[1]; g.lineWidth = 2.2;
      g.beginPath(); g.moveTo(axL.X(xmin), axL.Y(p[0] + p[1] * xmin)); g.lineTo(axL.X(xmax), axL.Y(p[0] + p[1] * xmax)); g.stroke();
      g.fillStyle = col.c[0];
      for (i = 0; i < data.m; i++) { g.beginPath(); g.arc(axL.X(data.x[i]), axL.Y(data.y[i]), 3.6, 0, 7); g.fill(); }
      g.restore();
      g.font = "12px " + sans; g.fillStyle = col.soft; g.textAlign = "left"; g.textBaseline = "top";
      g.fillText("h(x) = θ₀ + θ₁x", bL.x + 8, bL.y + 6);

      // ----- right: contours and path
      cvR.clear();
      var gR = cvR.g;
      axR = ctx.axes(cvR, { xmin: P.t0min, xmax: P.t0max, ymin: P.t1min, ymax: P.t1max, xlabel: "θ₀", ylabel: "θ₁",
        margin: { l: 36, r: 10, t: 10, b: 32 }, grid: false });
      gR.save();
      gR.setTransform(1, 0, 0, 1, 0, 0);
      gR.drawImage(contourLayer(axR, col), 0, 0);
      gR.restore();
      axR.draw();
      var b = axR.box;
      gR.save();
      gR.beginPath(); gR.rect(b.x, b.y, b.w, b.h); gR.clip();
      // optimum
      var ox = axR.X(opt.t0), oy = axR.Y(opt.t1);
      gR.strokeStyle = col.ink; gR.lineWidth = 1.6;
      gR.beginPath(); gR.moveTo(ox - 5, oy - 5); gR.lineTo(ox + 5, oy + 5); gR.moveTo(ox + 5, oy - 5); gR.lineTo(ox - 5, oy + 5); gR.stroke();
      // path
      var path = state.path, n = path.length;
      gR.strokeStyle = ctx.alpha(col.c[1], 0.85); gR.lineWidth = 1.4;
      gR.beginPath();
      for (var k = 0; k < n; k++) {
        var X = clampPx(axR.X(path[k][0])), Y = clampPx(axR.Y(path[k][1]));
        if (k === 0) gR.moveTo(X, Y); else gR.lineTo(X, Y);
      }
      gR.stroke();
      gR.fillStyle = col.c[1];
      var every = n > 400 ? Math.ceil(n / 400) : 1;
      for (k = 1; k < n; k += every) { gR.beginPath(); gR.arc(axR.X(path[k][0]), axR.Y(path[k][1]), 2, 0, 7); gR.fill(); }
      // start point
      gR.strokeStyle = col.c[1]; gR.lineWidth = 1.8; gR.fillStyle = col.paper;
      gR.beginPath(); gR.arc(axR.X(path[0][0]), axR.Y(path[0][1]), 4.5, 0, 7); gR.fill(); gR.stroke();
      // current point
      gR.fillStyle = col.c[1];
      gR.beginPath(); gR.arc(axR.X(p[0]), axR.Y(p[1]), 4.5, 0, 7); gR.fill();
      gR.restore();
      gR.font = "11px " + sans; gR.fillStyle = col.muted; gR.textAlign = "right"; gR.textBaseline = "top";
      gR.fillText(narrow ? "click to start" : "click to start from there", b.x + b.w - 6, b.y + 5);

      var status = state.status === "diverged" ? "diverged" :
        state.status === "settled" ? "at the minimum" :
        state.status === "stopped" ? "stopped at " + D.maxIter + " steps" :
        state.status === "ready" ? "ready" :
        state.alpha > lim.limit ? "overshooting, growing" :
        state.alpha > lim.limit / 2 ? "zigzagging inward" : "descending";
      ro.set({
        "iteration": String(n - 1),
        "θ₀": fmtBig(p[0]),
        "θ₁": fmtBig(p[1]),
        "cost J": fmtBig(J),
        "least squares θ₀": ctx.fmt(opt.t0, 3),
        "least squares θ₁": ctx.fmt(opt.t1, 3),
        "least squares J": ctx.fmt(opt.J, 3),
        "diverges above α": ctx.fmt(lim.limit, 2),
        "status": status
      });
    }
    function clampPx(v) { return Math.max(-1e4, Math.min(1e4, v)); }
    function fmtBig(v) { return Math.abs(v) >= 1e4 ? v.toExponential(2).replace("-", "−") : ctx.fmt(v, Math.abs(v) >= 100 ? 1 : 3); }

    legend.innerHTML = "";
    [["dot", 0, "data"], ["", 1, "current line and descent path"], ["", 3, "residuals"]].forEach(function (s) {
      var sp = ctx.el("span", null, legend), ic = ctx.el("i", s[0] || null, sp);
      ic.dataset.c = s[1];
      sp.appendChild(document.createTextNode(s[2]));
    });
    var sp2 = ctx.el("span", null, legend); ctx.el("i", null, sp2).dataset.c = "x";
    sp2.appendChild(document.createTextNode("least squares line and optimum ×"));
    function paintLegend() {
      var col = ctx.colors();
      legend.querySelectorAll("i").forEach(function (ic) {
        var c = ic.dataset.c;
        ic.style.background = c === "x" ? ctx.alpha(col.ink, 0.45) : col.c[+c];
      });
    }

    cvL.onresize = draw; cvR.onresize = draw;
    paintLegend();
    draw();
    if (!ctx.reducedMotion) play(); else { advance(D.maxIter); draw(); syncButton(true); }
    return { draw: function () { paintLegend(); draw(); } };
  }
});
