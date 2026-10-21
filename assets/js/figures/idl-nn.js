FIG.register("idl-nn", {
  defaults: { seed: 5, m: 200, H: 4, act: "sigmoid", kind: "circles", alpha: 1, epochs: 3000 },

  // Pure math, no DOM. The network the book describes: inputs x1, x2, one hidden layer
  // a_j = g(theta0^[j] + theta1^[j] x1 + theta2^[j] x2), output h = sigma(phi0 + sum_j phi_j a_j),
  // cross entropy cost, full-batch gradient descent, weights uniform in [-1, 1] and biases 0.
  compute: {
    data: function (kind, seed, m) {
      var r = FIG.ctx.rng(seed + (kind === "xor" ? 1000 : 0));
      var x1 = new Float64Array(m), x2 = new Float64Array(m), y = new Float64Array(m);
      for (var i = 0; i < m; i++) {
        if (kind === "circles") {
          // inner disc is y = 1, outer ring is y = 0
          var inner = i < m / 2, ang = 2 * Math.PI * r.uniform();
          var rad = inner ? 0.9 * Math.sqrt(r.uniform()) : 1.35 + 0.65 * r.uniform();
          x1[i] = rad * Math.cos(ang) + 0.08 * r.normal();
          x2[i] = rad * Math.sin(ang) + 0.08 * r.normal();
          y[i] = inner ? 1 : 0;
        } else {
          // four blobs; y = 1 where x1 and x2 have the same sign
          var q = i % 4, sx = q === 0 || q === 3 ? 1 : -1, sy = q < 2 ? 1 : -1;
          x1[i] = sx * 1.0 + 0.38 * r.normal();
          x2[i] = sy * 1.0 + 0.38 * r.normal();
          y[i] = sx * sy > 0 ? 1 : 0;
        }
      }
      return { x1: x1, x2: x2, y: y, m: m };
    },

    init: function (H, seed) {
      var r = FIG.ctx.rng(seed);
      var P = { H: H, W1: new Float64Array(2 * H), b1: new Float64Array(H), W2: new Float64Array(H), b2: 0 };
      for (var j = 0; j < H; j++) { P.W1[2 * j] = 2 * r.uniform() - 1; P.W1[2 * j + 1] = 2 * r.uniform() - 1; }
      for (j = 0; j < H; j++) P.W2[j] = 2 * r.uniform() - 1;
      return P;
    },

    // activation g and its derivative written in terms of z
    g: function (act, z) {
      if (act === "sigmoid") return 1 / (1 + Math.exp(-z));
      if (act === "tanh") { var e = Math.exp(-2 * Math.abs(z)), t = (1 - e) / (1 + e); return z < 0 ? -t : t; }
      if (act === "relu") return z > 0 ? z : 0;
      return z;                                         // linear
    },
    dg: function (act, z, a) {
      if (act === "sigmoid") return a * (1 - a);
      if (act === "tanh") return 1 - a * a;
      if (act === "relu") return z > 0 ? 1 : 0;
      return 1;
    },

    // probability for one input
    predict: function (P, act, x1, x2) {
      var z2 = P.b2;
      for (var j = 0; j < P.H; j++) z2 += P.W2[j] * this.g(act, P.b1[j] + P.W1[2 * j] * x1 + P.W1[2 * j + 1] * x2);
      return 1 / (1 + Math.exp(-z2));
    },

    // One epoch of full-batch gradient descent with backpropagation, in place.
    // Returns the cost and accuracy of the parameters before the update.
    epoch: function (d, P, act, alpha, buf) {
      var H = P.H, m = d.m, J = 0, correct = 0;
      var gW1 = buf.gW1, gb1 = buf.gb1, gW2 = buf.gW2, z1 = buf.z1, a1 = buf.a1, gb2 = 0, j;
      for (j = 0; j < H; j++) { gW1[2 * j] = 0; gW1[2 * j + 1] = 0; gb1[j] = 0; gW2[j] = 0; }
      for (var i = 0; i < m; i++) {
        var x1 = d.x1[i], x2 = d.x2[i], y = d.y[i], z2 = P.b2;
        for (j = 0; j < H; j++) {
          z1[j] = P.b1[j] + P.W1[2 * j] * x1 + P.W1[2 * j + 1] * x2;
          a1[j] = this.g(act, z1[j]);
          z2 += P.W2[j] * a1[j];
        }
        // cross entropy of one example, written as softplus so h = 1 never takes log(0)
        var sp = Math.max(z2, 0) + Math.log(1 + Math.exp(-Math.abs(z2)));
        J += y === 1 ? sp - z2 : sp;
        if ((z2 >= 0 ? 1 : 0) === y) correct++;
        var dz2 = 1 / (1 + Math.exp(-z2)) - y;          // dJ/dz2 for sigmoid + cross entropy
        gb2 += dz2;
        for (j = 0; j < H; j++) {
          gW2[j] += dz2 * a1[j];
          var dz1 = dz2 * P.W2[j] * this.dg(act, z1[j], a1[j]);
          gW1[2 * j] += dz1 * x1; gW1[2 * j + 1] += dz1 * x2; gb1[j] += dz1;
        }
      }
      var s = alpha / m;
      for (j = 0; j < H; j++) {
        P.W1[2 * j] -= s * gW1[2 * j]; P.W1[2 * j + 1] -= s * gW1[2 * j + 1]; P.b1[j] -= s * gb1[j]; P.W2[j] -= s * gW2[j];
      }
      P.b2 -= s * gb2;
      return { J: J / m, acc: correct / m };
    },

    buffers: function (H) {
      return { gW1: new Float64Array(2 * H), gb1: new Float64Array(H), gW2: new Float64Array(H), z1: new Float64Array(H), a1: new Float64Array(H) };
    },

    evaluate: function (d, P, act) {
      var J = 0, c = 0;
      for (var i = 0; i < d.m; i++) {
        var h = this.predict(P, act, d.x1[i], d.x2[i]);
        h = Math.min(Math.max(h, 1e-15), 1 - 1e-15);
        J += d.y[i] === 1 ? -Math.log(h) : -Math.log(1 - h);
        if ((h >= 0.5 ? 1 : 0) === d.y[i]) c++;
      }
      return { J: J / d.m, acc: c / d.m };
    },

    // With a linear activation the network is h = sigma(w . x + b): logistic regression.
    collapse: function (P) {
      var w1 = 0, w2 = 0, b = P.b2;
      for (var j = 0; j < P.H; j++) { w1 += P.W2[j] * P.W1[2 * j]; w2 += P.W2[j] * P.W1[2 * j + 1]; b += P.W2[j] * P.b1[j]; }
      return { b: b, w1: w1, w2: w2 };
    },

    train: function (kind, H, act, epochs, alpha, seed, m) {
      var d = this.data(kind, seed, m), P = this.init(H, seed + H), buf = this.buffers(H);
      for (var e = 0; e < epochs; e++) this.epoch(d, P, act, alpha, buf);
      var ev = this.evaluate(d, P, act);
      return { J: ev.J, acc: ev.acc, P: P, d: d };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var st = { kind: D.kind, H: D.H, act: D.act, d: null, P: null, buf: null, ep: 0, hist: null, ghost: null, acc: 0, last: null };
    var NAMES = { sigmoid: "sigmoid", tanh: "tanh", relu: "ReLU", linear: "linear" };

    var row = ctx.controls(stage);
    var sH = ctx.slider(row, { label: "hidden units H", min: 1, max: 8, step: 1, value: D.H,
      fmt: function (v) { return String(v); }, oninput: function (v) { st.H = v; reset(true); } });
    var tA = ctx.toggle(row, { label: "activation", value: D.act,
      options: [{ value: "sigmoid", label: "sigmoid" }, { value: "tanh", label: "tanh" }, { value: "relu", label: "ReLU" }, { value: "linear", label: "linear" }],
      onchange: function (v) { st.act = v; reset(true); } });
    var tD = ctx.toggle(row, { label: "data", value: D.kind,
      options: [{ value: "circles", label: "circles" }, { value: "xor", label: "XOR" }],
      onchange: function (v) { st.kind = v; reset(true, true); } });
    var bPlay = ctx.button(row, "Play", function () { togglePlay(); });
    bPlay.classList.add("fig__btn--primary");
    ctx.button(row, "Reset", function () {
      st.H = D.H; sH.set(D.H); st.act = D.act; tA.set(D.act); st.kind = D.kind; tD.set(D.kind); reset(false, true);
    });

    var grid = ctx.el("div", "fig__row fig__row--3-2", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);
    var cvP = ctx.canvas(left, { aspect: 0.92, minHeight: 260, maxHeight: 430, label: "Training points and the network's decision region" });
    var cvN = ctx.canvas(right, { aspect: 0.78, minHeight: 190, maxHeight: 260, label: "Network diagram: edge thickness is the size of each weight" });
    var cvJ = ctx.canvas(right, { aspect: 0.55, minHeight: 140, maxHeight: 175, label: "Cost J against epoch" });
    var legend = ctx.el("div", "fig__legend", stage);
    var ro = ctx.readout(stage);
    var R = 2.5;

    function reset(andPlay, dropGhost) {
      anim.stop();
      // a finished run stays as a faint curve to compare against, until the data changes
      if (dropGhost) st.ghost = null;
      else if (st.hist && st.ep >= D.epochs) st.ghost = { hist: st.hist, label: NAMES[st.last.act] + ", H = " + st.last.H };
      st.d = C.data(st.kind, D.seed, D.m);
      st.P = C.init(st.H, D.seed + st.H);
      st.buf = C.buffers(st.H);
      st.hist = new Float64Array(D.epochs + 1);
      st.ep = 0; st.acc = 0;
      st.last = { act: st.act, H: st.H };
      var ev = C.evaluate(st.d, st.P, st.act);
      st.hist[0] = ev.J; st.cur = ev;
      draw();
      if (andPlay) play(); else syncButton();
    }

    function advance(n) {
      for (var k = 0; k < n && st.ep < D.epochs; k++) {
        st.hist[st.ep] = C.epoch(st.d, st.P, st.act, D.alpha, st.buf).J;   // cost before this update
        st.ep++;
      }
      st.cur = C.evaluate(st.d, st.P, st.act);
      st.hist[st.ep] = st.cur.J;
    }

    var anim = ctx.loop(stage, function (dt) {
      var per = Math.max(2, Math.min(30, Math.round(st.ep / 25)));
      advance(per);
      draw();
      if (st.ep >= D.epochs) { syncButton(true); return false; }
    });
    function syncButton(stopped) { bPlay.textContent = (anim.running && !stopped) ? "Pause" : "Play"; }
    function play() {
      if (st.ep >= D.epochs) reset(false);
      if (ctx.reducedMotion) { advance(D.epochs); draw(); syncButton(true); return; }
      anim.start(); syncButton();
    }
    function togglePlay() { if (anim.running) { anim.stop(); syncButton(); } else play(); }

    function rgbOf(c) { return (ctx.alpha(c, 1).match(/[\d.]+/g) || [0, 0, 0]).slice(0, 3).map(Number); }
    var N = 64, shade = document.createElement("canvas"), field = new Float64Array((N + 1) * (N + 1));
    shade.width = N + 1; shade.height = N + 1;

    function drawPlane(col) {
      var g = cvP.g, P = st.P;
      cvP.clear();
      var ax = ctx.axes(cvP, { xmin: -R, xmax: R, ymin: -R, ymax: R, xlabel: "x₁", ylabel: "x₂", margin: { l: 34, r: 10, t: 10, b: 32 } });
      var side = Math.min(ax.box.w, ax.box.h);
      ax.box.x += (ax.box.w - side) / 2; ax.box.w = side; ax.box.h = side;
      var b = ax.box;
      var sg = shade.getContext("2d"), im = sg.createImageData(N + 1, N + 1);
      var c0 = rgbOf(col.c[0]), c1 = rgbOf(col.c[1]);
      for (var j = 0; j <= N; j++) {
        for (var i = 0; i <= N; i++) {
          var x1 = -R + 2 * R * i / N, x2 = R - 2 * R * j / N;
          var h = C.predict(P, st.act, x1, x2);
          field[j * (N + 1) + i] = h;
          var cc = h >= 0.5 ? c1 : c0, o = 4 * (j * (N + 1) + i);
          im.data[o] = cc[0]; im.data[o + 1] = cc[1]; im.data[o + 2] = cc[2];
          im.data[o + 3] = Math.round(255 * (0.08 + 0.3 * Math.abs(2 * h - 1)));
        }
      }
      sg.putImageData(im, 0, 0);
      g.imageSmoothingEnabled = true;
      g.drawImage(shade, b.x, b.y, b.w, b.h);
      ax.draw();
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      // boundary h = 0.5 by marching squares on the same grid
      g.strokeStyle = col.ink; g.lineWidth = 2; g.beginPath();
      function pt(fi, fj) { return [b.x + b.w * fi / N, b.y + b.h * fj / N]; }
      for (j = 0; j < N; j++) {
        for (i = 0; i < N; i++) {
          var a = field[j * (N + 1) + i] - 0.5, bb = field[j * (N + 1) + i + 1] - 0.5;
          var c = field[(j + 1) * (N + 1) + i + 1] - 0.5, dd = field[(j + 1) * (N + 1) + i] - 0.5;
          var pts = [];
          if ((a < 0) !== (bb < 0)) pts.push(pt(i + a / (a - bb), j));
          if ((bb < 0) !== (c < 0)) pts.push(pt(i + 1, j + bb / (bb - c)));
          if ((c < 0) !== (dd < 0)) pts.push(pt(i + 1 - c / (c - dd), j + 1));
          if ((dd < 0) !== (a < 0)) pts.push(pt(i, j + 1 - dd / (dd - a)));
          for (var p = 0; p + 1 < pts.length; p += 2) { g.moveTo(pts[p][0], pts[p][1]); g.lineTo(pts[p + 1][0], pts[p + 1][1]); }
        }
      }
      g.stroke();
      // linear: the collapsed logistic regression line, drawn from W2 . W1 directly
      var lr = null;
      if (st.act === "linear") {
        lr = C.collapse(P);
        if (Math.abs(lr.w1) + Math.abs(lr.w2) > 1e-9) {
          g.strokeStyle = col.paper; g.lineWidth = 1.2; g.setLineDash([5, 4]); g.beginPath();
          if (Math.abs(lr.w2) > Math.abs(lr.w1)) {
            g.moveTo(ax.X(-R), ax.Y((-lr.b + lr.w1 * R) / lr.w2)); g.lineTo(ax.X(R), ax.Y((-lr.b - lr.w1 * R) / lr.w2));
          } else {
            g.moveTo(ax.X((-lr.b + lr.w2 * R) / lr.w1), ax.Y(-R)); g.lineTo(ax.X((-lr.b - lr.w2 * R) / lr.w1), ax.Y(R));
          }
          g.stroke(); g.setLineDash([]);
        }
      }
      var d = st.d;
      for (var k = 0; k < d.m; k++) {
        var X = ax.X(d.x1[k]), Y = ax.Y(d.x2[k]);
        g.fillStyle = d.y[k] === 1 ? col.c[1] : col.c[0];
        g.strokeStyle = col.paper; g.lineWidth = 0.8;
        g.beginPath();
        if (d.y[k] === 1) g.arc(X, Y, 3.2, 0, 7); else g.rect(X - 2.9, Y - 2.9, 5.8, 5.8);
        g.fill(); g.stroke();
      }
      g.restore();
      if (lr) {
        var th3 = ctx.fmt(lr.b, 3) + ", " + ctx.fmt(lr.w1, 3) + ", " + ctx.fmt(lr.w2, 3);
        var txt = "same as logistic regression with θ = (" + th3 + ")";
        g.font = "11px " + sans; g.textAlign = "left"; g.textBaseline = "top";
        var tw = g.measureText(txt).width;
        if (tw > b.w - 12) { txt = "logistic regression θ = (" + th3 + ")"; tw = g.measureText(txt).width; }
        if (tw > b.w - 12) { txt = "θ = (" + th3 + ")"; tw = g.measureText(txt).width; }
        g.fillStyle = ctx.alpha(col.paper, 0.85); g.fillRect(b.x + 4, b.y + 4, tw + 8, 17);
        g.fillStyle = col.ink; g.fillText(txt, b.x + 8, b.y + 7);
      }
    }

    // tiny plot of the activation inside each hidden node
    function glyph(g, cx, cy, r, col) {
      g.save();
      g.strokeStyle = col.soft; g.lineWidth = 1.2; g.beginPath();
      for (var t = 0; t <= 16; t++) {
        var u = -1 + 2 * t / 16, z = u * 4, v;
        if (st.act === "sigmoid") v = 2 * C.g("sigmoid", z) - 1;
        else if (st.act === "tanh") v = C.g("tanh", z * 0.6);
        else if (st.act === "relu") v = Math.max(0, u) * 1.6 - 0.8;
        else v = u * 0.8;
        var X = cx + u * r * 0.6, Y = cy - v * r * 0.55;
        if (t === 0) g.moveTo(X, Y); else g.lineTo(X, Y);
      }
      g.stroke(); g.restore();
    }

    function drawNet(col) {
      var g = cvN.g, P = st.P, H = P.H;
      cvN.clear();
      var W = cvN.w, Hh = cvN.h, top = 22, bot = 10;
      var xs = [W * 0.14, W * 0.5, W * 0.86];
      var rad = Math.max(7, Math.min(13, (Hh - top - bot) / (2.6 * Math.max(H, 2))));
      function ys(n) { var out = []; for (var i = 0; i < n; i++) out.push(top + (Hh - top - bot) * (i + 0.5) / n); return out; }
      var yi = ys(2), yh = ys(H), yo = ys(1);
      var wmax = 1e-9, j;
      for (j = 0; j < 2 * H; j++) wmax = Math.max(wmax, Math.abs(P.W1[j]));
      for (j = 0; j < H; j++) wmax = Math.max(wmax, Math.abs(P.W2[j]));
      function edge(x0, y0, x1, y1, w) {
        g.strokeStyle = ctx.alpha(w >= 0 ? col.c[2] : col.c[3], 0.25 + 0.65 * Math.min(1, Math.abs(w) / wmax));
        g.lineWidth = 0.6 + 4.4 * Math.abs(w) / wmax;
        g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      }
      for (j = 0; j < H; j++) {
        edge(xs[0], yi[0], xs[1], yh[j], P.W1[2 * j]);
        edge(xs[0], yi[1], xs[1], yh[j], P.W1[2 * j + 1]);
        edge(xs[1], yh[j], xs[2], yo[0], P.W2[j]);
      }
      g.font = "12px " + sans; g.textAlign = "center"; g.textBaseline = "middle";
      function node(x, y, label) {
        g.fillStyle = col.paper; g.strokeStyle = col.rule; g.lineWidth = 1.2;
        g.beginPath(); g.arc(x, y, rad + 2, 0, 7); g.fill(); g.stroke();
        if (label) { g.fillStyle = col.ink; g.fillText(label, x, y + 0.5); }
      }
      node(xs[0], yi[0], "x₁"); node(xs[0], yi[1], "x₂");
      for (j = 0; j < H; j++) { node(xs[1], yh[j], ""); glyph(g, xs[1], yh[j], rad + 2, col); }
      node(xs[2], yo[0], "h");
      g.font = "11px " + sans; g.fillStyle = col.muted; g.textBaseline = "top";
      g.fillText("inputs", xs[0], 3); g.fillText("hidden layer", xs[1], 3); g.fillText("output", xs[2], 3);
    }

    function drawCost(col) {
      var g = cvJ.g;
      cvJ.clear();
      var ax = ctx.axes(cvJ, { xmin: 0, xmax: D.epochs, ymin: 0, ymax: 0.8, xlabel: "epoch", ylabel: "cost J",
        margin: { l: 40, r: 18, t: 8, b: 32 }, yticks: [0, 0.4, 0.8], xticks: [0, 1000, 2000, 3000] });
      ax.draw();
      function curve(hist, upto, color, w, dash) {
        g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath();
        var started = false;
        for (var e = 0; e <= upto; e++) {
          var v = hist[e];
          if (!isFinite(v)) continue;
          var X = ax.X(e), Y = ax.Y(Math.min(v, 0.8));
          if (!started) { g.moveTo(X, Y); started = true; } else g.lineTo(X, Y);
        }
        g.stroke(); g.setLineDash([]);
      }
      if (st.ghost) {
        curve(st.ghost.hist, D.epochs, ctx.alpha(col.muted, 0.8), 1.3, [4, 3]);
        g.font = "11px " + sans; g.fillStyle = col.muted; g.textAlign = "right"; g.textBaseline = "bottom";
        var gl = st.ghost.hist[D.epochs];
        var gy = ax.Y(Math.min(gl, 0.8));
        g.fillText(st.ghost.label, ax.box.x + ax.box.w - 4, gy < ax.box.y + 16 ? gy + 14 : gy - 3);
      }
      curve(st.hist, st.ep, col.c[3], 1.8);
      g.fillStyle = col.c[3]; g.beginPath(); g.arc(ax.X(st.ep), ax.Y(Math.min(st.cur.J, 0.8)), 3, 0, 7); g.fill();
    }

    function draw() {
      var col = ctx.colors();
      drawPlane(col); drawNet(col); drawCost(col);
      ro.set({
        "epoch": String(st.ep),
        "cost J": ctx.fmt(st.cur.J, 4),
        "accuracy": (100 * st.cur.acc).toFixed(1) + "%"
      });
    }

    function paintLegend() {
      var col = ctx.colors();
      legend.innerHTML = "";
      [[col.c[0], "y = 0", "sq"], [col.c[1], "y = 1", "dot"], [col.ink, "boundary h = 0.5", ""],
       [col.c[2], "positive weight", ""], [col.c[3], "negative weight", ""]].forEach(function (s) {
        var sp = ctx.el("span", null, legend), ic = ctx.el("i", s[2] === "dot" ? "dot" : null, sp);
        ic.style.background = s[0];
        if (s[2] === "sq") { ic.style.width = ".55rem"; ic.style.height = ".55rem"; }
        sp.appendChild(document.createTextNode(s[1]));
      });
    }

    cvP.onresize = draw; cvN.onresize = draw; cvJ.onresize = draw;
    paintLegend();
    reset(false);
    if (ctx.reducedMotion) { advance(D.epochs); draw(); syncButton(true); } else play();
    return { draw: function () { paintLegend(); draw(); } };
  }
});
