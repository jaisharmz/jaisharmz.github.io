FIG.register("cx-bid", {
  defaults: { k: 4, lam: 2, C: 1, seed: 7, auctions: 2000 },

  // Pure math, no DOM. Notation follows the text: the provider asks B, the lowest
  // competing ask is B* ~ Gamma(k, lambda) with shape k and rate lambda, C is the
  // marginal cost, and E[Profit] = (B - C) P(B < B*) = (B - C)(1 - F(B)).
  compute: {
    // log Gamma(z), Lanczos approximation (g = 7, 9 terms), about 15 digits.
    lnGamma: function (z) {
      var L = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
      if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - this.lnGamma(1 - z);
      z -= 1;
      var x = L[0];
      for (var i = 1; i < 9; i++) x += L[i] / (z + i);
      var t = z + 7.5;
      return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
    },

    // Regularized lower incomplete gamma P(a, x) = gamma(a, x) / Gamma(a).
    // Series below x = a + 1, Lentz continued fraction for Q = 1 - P above it.
    gammaP: function (a, x) {
      if (!(x > 0)) return 0;
      var pre = Math.exp(-x + a * Math.log(x) - this.lnGamma(a));
      var n;
      if (x < a + 1) {
        var ap = a, del = 1 / a, sum = del;
        for (n = 1; n < 1000; n++) {
          ap += 1; del *= x / ap; sum += del;
          if (Math.abs(del) < Math.abs(sum) * 1e-16) break;
        }
        return Math.min(1, sum * pre);
      }
      var tiny = 1e-300, b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
      for (n = 1; n < 1000; n++) {
        var an = -n * (n - a);
        b += 2;
        d = an * d + b; if (Math.abs(d) < tiny) d = tiny;
        c = b + an / c; if (Math.abs(c) < tiny) c = tiny;
        d = 1 / d;
        var step = d * c;
        h *= step;
        if (Math.abs(step - 1) < 1e-16) break;
      }
      return Math.max(0, 1 - pre * h);
    },

    pdf: function (x, k, lam) {
      if (x <= 0) return k === 1 ? (x === 0 ? lam : 0) : 0;
      return Math.exp(k * Math.log(lam) + (k - 1) * Math.log(x) - lam * x - this.lnGamma(k));
    },

    cdf: function (x, k, lam) { return this.gammaP(k, lam * x); },

    quantile: function (q, k, lam) {
      var lo = 0, hi = k / lam + 20 * Math.sqrt(k) / lam;
      for (var i = 0; i < 80; i++) {
        var mid = 0.5 * (lo + hi);
        if (this.cdf(mid, k, lam) < q) lo = mid; else hi = mid;
      }
      return 0.5 * (lo + hi);
    },

    profit: function (B, p) { return (B - p.C) * (1 - this.cdf(B, p.k, p.lam)); },

    // Best ask: a grid over (C, far tail of B*), then golden-section refinement.
    optimum: function (p) {
      var self = this;
      var lo = p.C, hi = Math.max(p.C, 0) + p.k / p.lam + 12 * Math.sqrt(p.k) / p.lam;
      var N = 600, best = lo, bestV = -Infinity;
      for (var i = 0; i <= N; i++) {
        var B = lo + (hi - lo) * i / N, v = this.profit(B, p);
        if (v > bestV) { bestV = v; best = B; }
      }
      var h = (hi - lo) / N, a = Math.max(lo, best - h), b = best + h;
      var gr = (Math.sqrt(5) - 1) / 2;
      var x1 = b - gr * (b - a), x2 = a + gr * (b - a);
      var f1 = this.profit(x1, p), f2 = this.profit(x2, p);
      for (var it = 0; it < 80; it++) {
        if (f1 > f2) { b = x2; x2 = x1; f2 = f1; x1 = b - gr * (b - a); f1 = self.profit(x1, p); }
        else { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (b - a); f2 = self.profit(x2, p); }
      }
      var Bopt = 0.5 * (a + b);
      var surv = 1 - this.cdf(Bopt, p.k, p.lam), dens = this.pdf(Bopt, p.k, p.lam);
      return {
        B: Bopt, win: surv, profit: (Bopt - p.C) * surv,
        margin: Bopt - p.C, invHazard: surv / dens,
        mean: p.k / p.lam, sd: Math.sqrt(p.k) / p.lam
      };
    },

    // One Gamma(k, 1) draw, Marsaglia and Tsang (2000).
    sampleGamma: function (k, r) {
      if (k < 1) return this.sampleGamma(k + 1, r) * Math.pow(r.uniform(), 1 / k);
      var d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        var x, v;
        do { x = r.normal(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        var u = r.uniform();
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
      }
    },

    // Simulated auctions: draw the lowest competing ask, win when B < B*.
    auctions: function (p, B, n, seed) {
      var r = FIG.ctx.rng(seed), draws = new Float64Array(n), wins = 0, total = 0;
      for (var i = 0; i < n; i++) {
        var bs = this.sampleGamma(p.k, r) / p.lam;
        draws[i] = bs;
        if (B < bs) { wins++; total += B - p.C; }
      }
      return { draws: draws, winRate: wins / n, meanProfit: total / n };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var p = { k: D.k, lam: D.lam, C: D.C };
    var hoverB = null, cache = null;

    var row = ctx.controls(stage);
    ctx.slider(row, { label: "shape k", min: 1, max: 12, step: 0.5, value: D.k,
      fmt: function (v) { return v.toFixed(1); }, oninput: function (v) { p.k = v; draw(); } });
    ctx.slider(row, { label: "rate λ", min: 0.5, max: 6, step: 0.1, value: D.lam,
      fmt: function (v) { return v.toFixed(1); }, oninput: function (v) { p.lam = v; draw(); } });
    ctx.slider(row, { label: "marginal cost C", min: 0, max: 5, step: 0.05, value: D.C,
      fmt: function (v) { return v.toFixed(2); }, oninput: function (v) { p.C = v; draw(); } });

    var cv = ctx.canvas(stage, { aspect: 0.72, minHeight: 330, maxHeight: 470,
      label: "Expected profit and win probability against the ask B, above the density of the lowest competing ask B*" });
    var legend = ctx.el("div", "fig__legend", stage);
    var ro = ctx.readout(stage);
    cv.onresize = draw;

    // round up to a multiple of half a power of ten: 5.2 -> 5.5, 12.3 -> 15
    function niceCeil(v) {
      var unit = Math.pow(10, Math.floor(Math.log10(v))) / 2;
      return Math.ceil(v / unit - 1e-9) * unit;
    }

    function setLegend(col) {
      legend.innerHTML = "";
      [[col.c[0], "expected profit (B − C)·P(B < B*)", false],
       [col.c[1], "win probability P(B < B*), right axis", false],
       [col.c[2], "simulated auction won", true],
       [col.c[3], "simulated auction lost", true]].forEach(function (it) {
        var s = ctx.el("span", null, legend);
        var i = ctx.el("i", it[2] ? "dot" : null, s);
        i.style.background = it[0];
        s.appendChild(document.createTextNode(it[1]));
      });
    }

    function draw() {
      var col = ctx.colors(), g = cv.g;
      cv.clear();
      setLegend(col);
      if (!cache || cache.k !== p.k || cache.lam !== p.lam || cache.C !== p.C) {
        var oo = C.optimum(p);
        cache = { k: p.k, lam: p.lam, C: p.C, o: oo, sim: C.auctions(p, oo.B, D.auctions, D.seed),
          xmax: niceCeil(Math.max(C.quantile(0.995, p.k, p.lam), oo.B * 1.12, p.C * 1.15 + 0.5 / p.lam)) };
      }
      var o = cache.o, sim = cache.sim, xmax = cache.xmax;

      var ml = 44, mr = 40, mt = 14, mb = 34, gap = 22;
      var inner = cv.h - mt - mb - gap;
      var h1 = Math.round(inner * 0.56), h2 = inner - h1;
      var box1 = { x: ml, y: mt, w: cv.w - ml - mr, h: h1 };
      var box2 = { x: ml, y: mt + h1 + gap, w: cv.w - ml - mr, h: h2 };

      // sample both curves
      var N = Math.max(160, Math.floor(box1.w)), xs = [], prof = [], surv = [], dens = [];
      var pmax = 0, dmax = 0;
      for (var i = 0; i <= N; i++) {
        var B = xmax * i / N, s = 1 - C.cdf(B, p.k, p.lam), d = C.pdf(B, p.k, p.lam);
        xs.push(B); surv.push(s); dens.push(d); prof.push((B - p.C) * s);
        if ((B - p.C) * s > pmax) pmax = (B - p.C) * s;
        if (isFinite(d) && d > dmax) dmax = d;
      }
      var ymax1 = pmax * 1.3;
      var ymin1 = -0.3 * pmax;

      var tick = ctx.niceTicks(0, xmax, Math.max(3, Math.floor(box1.w / 70)));
      var ax1 = ctx.axes(cv, { box: box1, xmin: 0, xmax: xmax, ymin: ymin1, ymax: ymax1,
        xticks: tick, xfmt: function () { return ""; }, ylabel: "expected profit" });
      var ax2 = ctx.axes(cv, { box: box2, xmin: 0, xmax: xmax, ymin: 0, ymax: dmax * 1.15,
        xticks: tick, ylabel: "density of B*", xlabel: "ask B", yticks: ctx.niceTicks(0, dmax * 1.15, 3) });
      ax1.draw(); ax2.draw();

      g.save();
      g.font = "11px " + sans;

      // zero line for profit
      g.strokeStyle = col.rule; g.lineWidth = 1;
      g.beginPath(); g.moveTo(box1.x, Math.round(ax1.Y(0)) + .5); g.lineTo(box1.x + box1.w, Math.round(ax1.Y(0)) + .5); g.stroke();

      // right axis: win probability 0..1 mapped onto the top box
      function Yp(v) { return box1.y + box1.h - v * box1.h * 0.92; }
      g.fillStyle = col.muted; g.textAlign = "left"; g.textBaseline = "middle";
      [0, 0.5, 1].forEach(function (v) { g.fillText(Math.round(v * 100) + "%", box1.x + box1.w + 5, Yp(v)); });

      // win region under the density: B* above the best ask
      g.save();
      g.beginPath(); g.rect(box2.x, box2.y, box2.w, box2.h); g.clip();
      g.fillStyle = ctx.alpha(col.c[2], 0.18);
      g.beginPath(); g.moveTo(ax2.X(o.B), ax2.Y(0));
      for (i = 0; i <= N; i++) if (xs[i] >= o.B) g.lineTo(ax2.X(xs[i]), ax2.Y(dens[i]));
      g.lineTo(ax2.X(xmax), ax2.Y(0)); g.closePath(); g.fill();
      g.strokeStyle = col.ink; g.lineWidth = 1.6;
      g.beginPath();
      var started = false;
      for (i = 0; i <= N; i++) {
        if (!isFinite(dens[i])) continue;
        var X = ax2.X(xs[i]), Y = ax2.Y(Math.min(dens[i], dmax * 1.2));
        if (!started) { g.moveTo(X, Y); started = true; } else g.lineTo(X, Y);
      }
      g.stroke();
      // rug of simulated lowest competing asks
      var nr = Math.min(160, sim.draws.length);
      for (i = 0; i < nr; i++) {
        var b = sim.draws[i];
        if (b > xmax) continue;
        g.strokeStyle = ctx.alpha(o.B < b ? col.c[2] : col.c[3], 0.8);
        g.beginPath(); g.moveTo(Math.round(ax2.X(b)) + .5, box2.y + box2.h); g.lineTo(Math.round(ax2.X(b)) + .5, box2.y + box2.h - 7); g.stroke();
      }
      g.restore();

      // top: win probability and expected profit
      g.save();
      g.beginPath(); g.rect(box1.x, box1.y, box1.w, box1.h); g.clip();
      g.strokeStyle = col.c[1]; g.lineWidth = 1.5; g.setLineDash([5, 4]);
      g.beginPath();
      for (i = 0; i <= N; i++) { if (i) g.lineTo(ax1.X(xs[i]), Yp(surv[i])); else g.moveTo(ax1.X(xs[i]), Yp(surv[i])); }
      g.stroke(); g.setLineDash([]);
      g.strokeStyle = col.c[0]; g.lineWidth = 2.2;
      g.beginPath();
      for (i = 0; i <= N; i++) {
        var yv = Math.max(prof[i], ymin1 - 1);
        if (i) g.lineTo(ax1.X(xs[i]), ax1.Y(yv)); else g.moveTo(ax1.X(xs[i]), ax1.Y(yv));
      }
      g.stroke();
      g.restore();

      // cost line and best-ask line through both panels
      function vline(x, color, dash, width) {
        var X = Math.round(ax1.X(x)) + .5;
        g.strokeStyle = color; g.lineWidth = width || 1; g.setLineDash(dash || []);
        g.beginPath(); g.moveTo(X, box1.y); g.lineTo(X, box1.y + box1.h); g.moveTo(X, box2.y); g.lineTo(X, box2.y + box2.h); g.stroke();
        g.setLineDash([]);
      }
      if (p.C <= xmax) vline(p.C, col.muted, [3, 3]);
      vline(o.B, col.ink, [], 1);

      // labels
      var cX = ax2.X(p.C), bX = ax2.X(o.B);
      g.fillStyle = col.muted; g.textBaseline = "top";
      var cLeft = cX - 4 > box2.x + 30;
      g.textAlign = cLeft ? "right" : "left";
      g.fillText("cost C", cLeft ? cX - 4 : cX + 4, box2.y + 4);
      g.fillStyle = col.c[2]; g.textAlign = "left";
      var winTxt = "win " + Math.round(o.win * 100) + "%";
      var wx = bX + 5;
      if (wx + g.measureText(winTxt).width > box2.x + box2.w - 2) { g.textAlign = "right"; wx = box2.x + box2.w - 4; }
      g.fillText(winTxt, wx, box2.y + 4);

      // optimum marker
      var px = ax1.X(o.B), py = ax1.Y(o.profit);
      g.fillStyle = col.c[0];
      g.beginPath(); g.arc(px, py, 4.5, 0, 2 * Math.PI); g.fill();
      g.strokeStyle = col.paper; g.lineWidth = 1.5; g.stroke();
      g.fillStyle = col.ink; g.textBaseline = "bottom";
      var lab = "best ask " + ctx.fmt(o.B, 2);
      var lw = g.measureText(lab).width;
      var lx = px + 7, ly = py - 6;
      g.textAlign = "left";
      if (lx + lw > box1.x + box1.w - 2) { g.textAlign = "right"; lx = px - 7; }
      if (ly - 12 < box1.y) ly = box1.y + 14;
      g.fillText(lab, lx, ly);

      // hover: a trial ask
      if (hoverB !== null && hoverB >= 0 && hoverB <= xmax) {
        var hs = 1 - C.cdf(hoverB, p.k, p.lam), hp = (hoverB - p.C) * hs;
        var hX = Math.round(ax1.X(hoverB)) + .5;
        g.strokeStyle = ctx.alpha(col.accent, 0.7); g.lineWidth = 1;
        g.beginPath(); g.moveTo(hX, box1.y); g.lineTo(hX, box1.y + box1.h); g.moveTo(hX, box2.y); g.lineTo(hX, box2.y + box2.h); g.stroke();
        if (hp > ymin1) {
          g.fillStyle = col.accent;
          g.beginPath(); g.arc(hX, ax1.Y(hp), 3, 0, 2 * Math.PI); g.fill();
        }
        var txt = "B " + ctx.fmt(hoverB, 2) + ": profit " + ctx.fmt(hp, 3) + ", win " + Math.round(hs * 100) + "%";
        var tw = g.measureText(txt).width;
        var tx = hX + 6;
        g.textAlign = "left";
        if (tx + tw > box1.x + box1.w - 2) { tx = hX - 6; g.textAlign = "right"; }
        g.fillStyle = col.paper;
        var rx = g.textAlign === "left" ? tx - 2 : tx - tw - 2;
        g.fillRect(rx, box1.y + box1.h - 19, tw + 4, 15);
        g.fillStyle = col.accent; g.textBaseline = "bottom";
        g.fillText(txt, tx, box1.y + box1.h - 5);
      }
      g.restore();

      ro.set({
        "best ask B": ctx.fmt(o.B, 3),
        "win probability": (o.win * 100).toFixed(1) + "%",
        "expected profit": ctx.fmt(o.profit, 3),
        "mean of B*, k/λ": ctx.fmt(o.mean, 3),
        "B − C": ctx.fmt(o.margin, 3),
        "(1 − F(B)) / f(B)": ctx.fmt(o.invHazard, 3),
        "simulated win rate": (sim.winRate * 100).toFixed(1) + "%",
        "simulated profit": ctx.fmt(sim.meanProfit, 3)
      });
    }

    function pointer(e) {
      var r = cv.el.getBoundingClientRect();
      var x = e.clientX - r.left;
      var ml = 44, mr = 40;
      if (x < ml || x > cv.w - mr) { hoverB = null; draw(); return; }
      hoverB = (x - ml) / (cv.w - ml - mr) * (cache ? cache.xmax : 1);
      draw();
    }
    cv.el.addEventListener("pointermove", pointer);
    cv.el.addEventListener("pointerleave", function () { hoverB = null; draw(); });

    draw();
    return { draw: draw };
  }
});
