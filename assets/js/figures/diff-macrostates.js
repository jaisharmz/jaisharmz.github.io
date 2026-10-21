FIG.register("diff-macrostates", {
  defaults: { n: 25, nmin: 3, nmax: 40, levelsShown: 12 },

  // Pure math, no DOM. Notation follows the text: n balls, bins with energies
  // E_j = j for j = 0..n, total energy exactly n. A macrostate is an occupancy
  // vector (n_0, ..., n_n) with sum n_j = n and sum j n_j = n, and its weight is
  // W = n! / prod n_j!. The balls with E_j >= 1 form an integer partition of n,
  // so enumerating partitions enumerates every macrostate exactly once.
  compute: {
    logFactorials: function (n) {
      var lf = new Float64Array(n + 1);
      for (var k = 1; k <= n; k++) lf[k] = lf[k - 1] + Math.log(k);
      return lf;
    },

    // Every macrostate of n balls. Returns the count, the total number of
    // microstates, the microstate-weighted mean occupancy <n_j>/n, and the
    // three macrostates with the largest W. W is built in log space.
    enumerate: function (n) {
      var lf = this.logFactorials(n);
      var occ = new Int32Array(n + 1);        // occ[j] for j >= 1; n_0 is n - (number of parts)
      var distinct = new Int32Array(n + 1);   // distinct energies in use, a stack
      var nd = 0;
      var offset = lf[n];                     // W <= n!, so every weight below is <= 1
      var sum = 0;                            // sum of W / n!
      var acc = new Float64Array(n + 1);      // sum of (W / n!) n_j
      var count = 0;
      var top = [];                           // up to 3 of {lw, occ}
      // equal W (to 1e-9 in log) is a tie, broken by more balls in the ground state
      function byWeight(a, b) {
        if (Math.abs(a.lw - b.lw) > 1e-9) return b.lw - a.lw;
        for (var i = 0; i <= n; i++) if (a.occ[i] !== b.occ[i]) return b.occ[i] - a.occ[i];
        return 0;
      }

      function leaf(lwParts, k) {
        var n0 = n - k;
        var lw = lf[n] - lf[n0] + lwParts;
        var w = Math.exp(lw - offset);
        count++;
        sum += w;
        acc[0] += w * n0;
        for (var d = 0; d < nd; d++) { var j = distinct[d]; acc[j] += w * occ[j]; }
        if (top.length < 3 || lw > top[top.length - 1].lw + 1e-9) {
          var v = new Array(n + 1);
          v[0] = n0;
          for (var i = 1; i <= n; i++) v[i] = occ[i];
          top.push({ lw: lw, occ: v });
          top.sort(byWeight);
          if (top.length > 3) top.pop();
        }
      }

      // lwParts = - sum_{j>=1} log(n_j!), built up one ball at a time
      function rec(rem, maxPart, k, lwParts) {
        if (rem === 0) { leaf(lwParts, k); return; }
        for (var p = Math.min(rem, maxPart); p >= 1; p--) {
          if (occ[p] === 0) distinct[nd++] = p;
          occ[p]++;
          rec(rem - p, p, k + 1, lwParts - Math.log(occ[p]));
          occ[p]--;
          if (occ[p] === 0) nd--;
        }
      }
      rec(n, n, 0, 0);

      var logTotal = offset + Math.log(sum);
      var tied = 0;
      for (var t = 0; t < top.length; t++) if (Math.abs(top[t].lw - top[0].lw) <= 1e-9) tied++;
      var mean = new Float64Array(n + 1);
      for (var j = 0; j <= n; j++) mean[j] = acc[j] / sum / n;
      return {
        n: n,
        count: count,
        logTotal: logTotal,
        total: Math.exp(logTotal),
        mean: mean,
        tied: tied,
        top: top.map(function (t) {
          return { occ: t.occ, W: Math.exp(t.lw), share: Math.exp(t.lw - logTotal) };
        })
      };
    },

    // The text's Lagrange solution n_j / n = c e^{-lambda_2 E_j}, with c and
    // lambda_2 fixed by the two constraints on levels 0..n: the fractions sum
    // to 1 and the mean energy per ball is 1. Solved by bisection.
    lagrange: function (n) {
      function meanEnergy(lam) {
        var z = 0, e = 0;
        for (var j = 0; j <= n; j++) { var w = Math.exp(-lam * j); z += w; e += j * w; }
        return e / z;
      }
      var lo = 0, hi = 20;
      for (var it = 0; it < 200; it++) {
        var mid = (lo + hi) / 2;
        if (meanEnergy(mid) > 1) lo = mid; else hi = mid;
      }
      var lam = (lo + hi) / 2, z = 0;
      for (var j = 0; j <= n; j++) z += Math.exp(-lam * j);
      return { lambda: lam, c: 1 / z };
    },

    // Fit log(<n_j>/n) = log c - lambda_2 j by least squares, each level
    // weighted by its occupancy so the nearly empty top levels do not dominate.
    fit: function (mean) {
      var sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (var j = 0; j < mean.length; j++) {
        var p = mean[j];
        if (!(p > 0)) continue;
        var y = Math.log(p);
        sw += p; sx += p * j; sy += p * y; sxx += p * j * j; sxy += p * j * y;
      }
      var slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
      var icpt = (sy - slope * sx) / sw;
      return { lambda: -slope, c: Math.exp(icpt) };
    },

    // Stars and bars, used only as a check: there are C(2n-1, n-1) microstates,
    // and one ball sits at energy j in C(2n-2-j, n-2) of them.
    logChoose: function (a, b) {
      if (b < 0 || b > a) return -Infinity;
      var s = 0;
      for (var i = 1; i <= b; i++) s += Math.log(a - b + i) - Math.log(i);
      return s;
    },
    closedForm: function (n) {
      var lt = this.logChoose(2 * n - 1, n - 1), mean = [];
      for (var j = 0; j <= n; j++) mean.push(Math.exp(this.logChoose(2 * n - 2 - j, n - 2) - lt));
      return { total: Math.exp(lt), mean: mean };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var cache = {};
    function get(n) {
      if (!cache[n]) {
        var e = C.enumerate(n);
        e.lag = C.lagrange(n);
        e.fit = C.fit(e.mean);
        cache[n] = e;
      }
      return cache[n];
    }

    var row = ctx.controls(stage);
    var sN = ctx.slider(row, {
      label: "number of balls n", min: D.nmin, max: D.nmax, step: 1, value: D.n,
      fmt: function (v) { return String(v); }, oninput: draw
    });
    var scale = ctx.toggle(row, {
      label: "scale", options: [{ value: "lin", label: "linear" }, { value: "log", label: "log" }],
      value: "lin", onchange: draw
    });

    var cv = ctx.canvas(stage, { aspect: 0.52, minHeight: 250, maxHeight: 380,
      label: "Mean occupancy of each energy level, the Lagrange exponential, and the most probable macrostate" });
    var legend = ctx.el("div", "fig__legend", stage);
    var lgBars = ctx.el("span", null, legend); var iBars = ctx.el("i", null, lgBars); iBars.style.height = ".55rem";
    lgBars.appendChild(document.createTextNode("mean occupancy over all microstates"));
    var lgCurve = ctx.el("span", null, legend); var iCurve = ctx.el("i", null, lgCurve);
    var lgCurveText = ctx.el("span", null, lgCurve);
    lgCurveText.style.display = "inline";
    lgCurveText.innerHTML = "Lagrange solution&nbsp;<em>C e<sup>\u2212\u03BB\u2082E<sub>j</sub></sup></em>";
    var lgDot = ctx.el("span", null, legend); var iDot = ctx.el("i", "dot", lgDot);
    var dotText = document.createTextNode("most probable macrostate");
    lgDot.appendChild(dotText);

    var ro = ctx.readout(stage);

    var tblWrap = ctx.el("div", null, stage);
    tblWrap.style.overflowX = "auto";
    tblWrap.style.margin = ".85rem 0 0";
    var tbl = ctx.el("table", null, tblWrap);
    tbl.style.fontSize = ".8rem";
    tbl.style.margin = "0";
    var cap = ctx.el("caption", null, tbl, "Largest macrostates");
    cap.style.cssText = "caption-side: top; text-align: left; color: var(--muted); font-size: .78rem; padding: 0 0 .3rem;";
    var thead = ctx.el("thead", null, tbl);
    var hr = ctx.el("tr", null, thead);
    ["macrostate", "(n₀, n₁, n₂, …)", "W", "share of microstates"].forEach(function (h) { ctx.el("th", null, hr, h); });
    var tbody = ctx.el("tbody", null, tbl);

    cv.onresize = draw;

    var SUP = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵",
      "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻" };
    function sci(v, d) {
      if (v < 1e6) return String(Math.round(v));
      var e = Math.floor(Math.log10(v)), m = v / Math.pow(10, e);
      if (+m.toFixed(d) >= 10) { m /= 10; e += 1; }
      return m.toFixed(d) + "\u00D710" + String(e).split("").map(function (c) { return SUP[c]; }).join("");
    }
    function vec(o, n) {
      var last = 0;
      for (var j = 0; j <= n; j++) if (o[j]) last = j;
      if (n <= 5) return "(" + o.slice(0, n + 1).join(", ") + ")";
      return "(" + o.slice(0, last + 1).join(", ") + (last < n ? ", 0, …" : "") + ")";
    }
    // the text's names for the three macrostates of n = 3
    var TEXT3 = { "0,3,0,0": "A", "1,1,1,0": "B", "2,0,0,1": "C" };

    function draw() {
      var col = ctx.colors();
      var n = sN.value, e = get(n), isLog = scale.value === "log";
      var J = Math.min(n, D.levelsShown);
      var best = e.top[0];
      var g = cv.g;
      cv.clear();

      var ymax, ymin;
      if (isLog) { ymin = -4; ymax = 0; }
      else {
        ymin = 0;
        ymax = Math.max(e.mean[0], best.occ[0] / n, e.lag.c) * 1.12;
        ymax = Math.ceil(ymax * 10) / 10;
      }
      var narrow = cv.w < 420;
      var ax = ctx.axes(cv, {
        xmin: -0.6, xmax: J + 0.6, ymin: ymin, ymax: ymax,
        xlabel: "energy level j", ylabel: "fraction of balls at level j",
        margin: { l: narrow ? 46 : 52, r: 10, t: 12, b: 36 },
        xticks: (function () { var t = [], st = J > 10 && narrow ? 2 : 1; for (var j = 0; j <= J; j += st) t.push(j); return t; })(),
        yticks: isLog ? [-4, -3, -2, -1, 0] : null,
        yfmt: isLog ? function (t) { return t === 0 ? "1" : "10" + String(t).split("").map(function (c) { return SUP[c]; }).join(""); } : null
      });
      ax.draw();
      function Yv(p) {
        if (!isLog) return ax.Y(p);
        if (!(p > 0)) return ax.Y(ymin);
        return ax.Y(Math.max(ymin, Math.min(ymax, Math.log10(p))));
      }

      var bw = Math.min(40, (ax.box.w / (J + 1.2)) * 0.62);
      var font = getComputedStyle(document.documentElement).getPropertyValue("--sans");
      g.save();
      g.beginPath(); g.rect(ax.box.x, ax.box.y, ax.box.w, ax.box.h); g.clip();

      // bars: mean occupancy over all microstates
      g.fillStyle = ctx.alpha(col.c[0], 0.78);
      for (var j = 0; j <= J; j++) {
        var top = Yv(e.mean[j]), base = ax.Y(ymin);
        g.fillRect(ax.X(j) - bw / 2, top, bw, base - top);
      }

      // the Lagrange exponential, fixed by the two constraints
      g.strokeStyle = col.c[1]; g.lineWidth = 2;
      g.beginPath();
      for (var s = 0; s <= 240; s++) {
        var x = -0.4 + (J + 0.8) * s / 240, p = e.lag.c * Math.exp(-e.lag.lambda * x);
        var px = ax.X(x), py = Yv(p);
        if (s === 0) g.moveTo(px, py); else g.lineTo(px, py);
      }
      g.stroke();

      // the single most probable macrostate
      g.fillStyle = col.c[2]; g.strokeStyle = col.paper; g.lineWidth = 1.5;
      var lastOcc = 0;
      for (j = 0; j <= n; j++) if (best.occ[j]) lastOcc = j;
      for (j = 0; j <= Math.min(J, lastOcc + 1); j++) {
        var q = best.occ[j] / n;
        if (isLog && q === 0) continue;
        g.beginPath(); g.arc(ax.X(j), Yv(q), 4, 0, 2 * Math.PI); g.fill(); g.stroke();
      }
      g.restore();

      // values over the bars for the few-ball cases, where they are the point
      if (n <= 6 && bw >= 26) {
        g.save();
        g.font = "11px " + font; g.fillStyle = col.ink; g.textAlign = "center"; g.textBaseline = "bottom";
        for (j = 0; j <= J; j++) {
          var yy = Math.min(Yv(e.mean[j]), Yv(best.occ[j] / n)) - 6;
          if (yy > ax.box.y + 10) g.fillText(e.mean[j].toFixed(2), ax.X(j), yy);
        }
        g.restore();
      }

      iBars.style.background = ctx.alpha(col.c[0], 0.78);
      iCurve.style.background = col.c[1];
      iDot.style.background = col.c[2];
      dotText.nodeValue = e.tied > 1 ? "a most probable macrostate (" + e.tied + " tie)" : "most probable macrostate";

      ro.set({
        "macrostates": String(e.count),
        "microstates": sci(e.total, 2),
        "largest W share": (100 * best.share).toFixed(1) + "%",
        "λ₂ fitted": e.fit.lambda.toFixed(3),
        "λ₂ from constraints": e.lag.lambda.toFixed(3)
      });

      tbody.textContent = "";
      e.top.forEach(function (t, i) {
        var tr = ctx.el("tr", null, tbody);
        var key = t.occ.slice(0, 4).join(",");
        var name = n === 3 && TEXT3[key] ? TEXT3[key]
          : e.tied > 1 && i < e.tied ? "tied for most probable"
          : ["most probable", "second", "third"][i];
        ctx.el("td", null, tr, name);
        var tdv = ctx.el("td", null, tr, vec(t.occ, n));
        tdv.style.fontFamily = "var(--mono)"; tdv.style.whiteSpace = "nowrap";
        ctx.el("td", null, tr, sci(t.W, 2));
        ctx.el("td", null, tr, (100 * t.share).toFixed(1) + "%");
      });
    }

    draw();

    // Fill the cache one n per tick, so dragging the slider never waits on
    // the 37338 partitions of 40.
    var nextN = D.nmin;
    (function warm() {
      while (nextN <= D.nmax && cache[nextN]) nextN++;
      if (nextN > D.nmax) return;
      get(nextN);
      setTimeout(warm, 30);
    })();

    return { draw: draw };
  }
});
