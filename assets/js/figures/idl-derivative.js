FIG.register("idl-derivative", {
  defaults: { x: 5, h: 2, side: "right", xmin: -5, xmax: 6, hmin: 0.001, hmax: 3 },

  // Pure math, no DOM. The function is the book's own example, f(x) = x^2 + 2x + 3,
  // whose derivative the text works out as 2x + 2 (12 at x = 5).
  compute: {
    f: function (x) { return x * x + 2 * x + 3; },
    fprime: function (x) { return 2 * x + 2; },
    // slope of the secant through (x, f(x)) and (x + h, f(x + h))
    secant: function (x, h) { return (this.f(x + h) - this.f(x)) / h; },
    table: function (x, hs) {
      var self = this;
      return hs.map(function (h) { var m = self.secant(x, h); return { h: h, slope: m, gap: m - self.fprime(x) }; });
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var sans = getComputedStyle(document.documentElement).getPropertyValue("--sans");
    var st = { x: D.x, habs: D.h, side: D.side };
    var L0 = Math.log10(D.hmin), L1 = Math.log10(D.hmax);

    var row = ctx.controls(stage);
    var sX = ctx.slider(row, { label: "point x", min: D.xmin, max: D.xmax, step: 0.1, value: D.x,
      fmt: function (v) { return ctx.fmt(v, 1); }, oninput: function (v) { st.x = v; draw(); } });
    // continuous log slider that snaps to 0.001, 0.002, 0.005, ..., 1, 2, 3 when close to one
    var NICE = [];
    [0.001, 0.01, 0.1, 1].forEach(function (p) { [1, 2, 5].forEach(function (k) { if (k * p <= D.hmax) NICE.push(Math.log10(k * p)); }); });
    NICE.push(Math.log10(D.hmax));
    function snap(u) { for (var i = 0; i < NICE.length; i++) if (Math.abs(u - NICE[i]) < 0.035) return NICE[i]; return u; }
    var sH = ctx.slider(row, { label: "gap |h| (log scale)", min: L0, max: L1, step: "any", value: Math.log10(D.h),
      fmt: function (u) { return fmtH(Math.pow(10, snap(u))); },
      oninput: function (u) { var v = snap(u); if (v !== u) sH.set(v); st.habs = +Math.pow(10, v).toPrecision(12); draw(); } });
    var tS = ctx.toggle(row, { label: "approach from", value: D.side,
      options: [{ value: "right", label: "right, h > 0" }, { value: "left", label: "left, h < 0" }],
      onchange: function (v) { st.side = v; draw(); } });
    ctx.button(row, "Reset", function () {
      st.x = D.x; st.habs = D.h; st.side = D.side; sX.set(D.x); sH.set(Math.log10(D.h)); tS.set(D.side); draw();
    });

    var grid = ctx.el("div", "fig__row fig__row--3-2", stage);
    var left = ctx.el("div", null, grid), right = ctx.el("div", null, grid);
    var cvF = ctx.canvas(left, { aspect: 0.78, minHeight: 250, maxHeight: 400, label: "The curve f(x), the secant line and the tangent line" });
    var optS = { aspect: 1.2, minHeight: 230, maxHeight: 400, label: "Secant slope against h on a log scale, approaching the derivative" };
    var cvS = ctx.canvas(right, optS);
    var legend = ctx.el("div", "fig__legend", stage);
    var ro = ctx.readout(stage);

    function fmtH(h) { return h >= 0.1 ? h.toFixed(3) : h >= 0.01 ? h.toFixed(4) : h.toFixed(5); }
    function hval() { return st.side === "right" ? st.habs : -st.habs; }

    // drag along the curve to move the point
    var dragging = false, axF = null;
    function setFromPointer(e) {
      var r = cvF.el.getBoundingClientRect();
      var v = axF.invX(e.clientX - r.left);
      v = Math.round(Math.max(D.xmin, Math.min(D.xmax, v)) * 10) / 10;
      st.x = v; sX.set(v); draw();
    }
    cvF.el.addEventListener("pointerdown", function (e) { if (!axF) return; dragging = true; cvF.el.setPointerCapture(e.pointerId); setFromPointer(e); });
    cvF.el.addEventListener("pointermove", function (e) { if (dragging) setFromPointer(e); });
    cvF.el.addEventListener("pointerup", function () { dragging = false; });
    cvF.el.addEventListener("pointercancel", function () { dragging = false; });
    cvF.el.style.cursor = "ew-resize";

    var XW = [-6, 9.5], YW = [-8, 108];

    function lineThrough(g, ax, x0, y0, m) {
      g.beginPath();
      g.moveTo(ax.X(XW[0]), ax.Y(y0 + m * (XW[0] - x0)));
      g.lineTo(ax.X(XW[1]), ax.Y(y0 + m * (XW[1] - x0)));
      g.stroke();
    }

    function drawF(col) {
      var g = cvF.g, x = st.x, h = hval(), fx = C.f(x), fxh = C.f(x + h);
      var m = C.secant(x, h), d = C.fprime(x);
      cvF.clear();
      axF = ctx.axes(cvF, { xmin: XW[0], xmax: XW[1], ymin: YW[0], ymax: YW[1], xlabel: "x", ylabel: "f(x)", margin: { l: 38, r: 10, t: 10, b: 32 } });
      axF.draw();
      var ax = axF, b = ax.box;
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      // f
      g.strokeStyle = col.ink; g.lineWidth = 2; g.beginPath();
      for (var i = 0; i <= 200; i++) { var t = XW[0] + (XW[1] - XW[0]) * i / 200; if (i === 0) g.moveTo(ax.X(t), ax.Y(C.f(t))); else g.lineTo(ax.X(t), ax.Y(C.f(t))); }
      g.stroke();
      // tangent
      g.strokeStyle = col.c[0]; g.lineWidth = 1.6; g.setLineDash([6, 4]);
      lineThrough(g, ax, x, fx, d);
      g.setLineDash([]);
      // secant
      g.strokeStyle = col.c[1]; g.lineWidth = 1.8;
      lineThrough(g, ax, x, fx, m);
      // rise over run
      g.strokeStyle = ctx.alpha(col.soft, 0.9); g.lineWidth = 1; g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(ax.X(x), ax.Y(fx)); g.lineTo(ax.X(x + h), ax.Y(fx)); g.lineTo(ax.X(x + h), ax.Y(fxh)); g.stroke();
      g.setLineDash([]);
      var runPx = Math.abs(ax.X(x + h) - ax.X(x)), risePx = Math.abs(ax.Y(fxh) - ax.Y(fx));
      g.font = "11px " + sans; g.fillStyle = col.soft;
      if (runPx > 26) { g.textAlign = "center"; g.textBaseline = h > 0 ? "top" : "top"; g.fillText("h", (ax.X(x) + ax.X(x + h)) / 2, ax.Y(fx) + 3); }
      if (risePx > 30) {
        // outside the triangle when there is room, otherwise inside it on a paper backing
        var lab = "f(x + h) − f(x)", tw = g.measureText(lab).width, lx = ax.X(x + h), ly = (ax.Y(fx) + ax.Y(fxh)) / 2;
        var outRight = h > 0;
        var fits = outRight ? lx + 5 + tw <= b.x + b.w : lx - 5 - tw >= b.x;
        var toRight = fits ? outRight : !outRight;
        var tx = toRight ? lx + 5 : lx - 5 - tw;
        if (!fits) { g.fillStyle = ctx.alpha(col.paper, 0.88); g.fillRect(tx - 2, ly - 8, tw + 4, 16); }
        g.fillStyle = col.soft; g.textAlign = "left"; g.textBaseline = "middle";
        g.fillText(lab, tx, ly);
      }
      // points
      g.fillStyle = col.c[1];
      g.beginPath(); g.arc(ax.X(x + h), ax.Y(fxh), 4.2, 0, 7); g.fill();
      g.fillStyle = col.ink;
      g.beginPath(); g.arc(ax.X(x), ax.Y(fx), 4.8, 0, 7); g.fill();
      g.restore();
      g.font = "12px " + sans; g.fillStyle = col.soft; g.textAlign = "left"; g.textBaseline = "top";
      g.fillText("f(x) = x² + 2x + 3", b.x + 8, b.y + 6);
    }

    function drawS(col) {
      var g = cvS.g, x = st.x, d = C.fprime(x);
      cvS.clear();
      var span = D.hmax * 1.12;
      var ax = ctx.axes(cvS, { xmin: L0, xmax: L1, ymin: d - span, ymax: d + span, xlabel: "|h| (log scale)", ylabel: "secant slope",
        margin: { l: 42, r: 14, t: 22, b: 32 }, xticks: [-3, -2, -1, 0], xfmt: function (u) { return String(+Math.pow(10, u).toPrecision(1)); } });
      ax.draw();
      var b = ax.box;
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      g.strokeStyle = col.c[0]; g.lineWidth = 1.6; g.setLineDash([6, 4]);
      g.beginPath(); g.moveTo(b.x, ax.Y(d)); g.lineTo(b.x + b.w, ax.Y(d)); g.stroke(); g.setLineDash([]);
      ["right", "left"].forEach(function (side) {
        var on = side === st.side, sgn = side === "right" ? 1 : -1;
        g.strokeStyle = on ? col.c[1] : ctx.alpha(col.c[1], 0.35); g.lineWidth = on ? 2 : 1.2;
        g.beginPath();
        for (var i = 0; i <= 120; i++) {
          var u = L0 + (L1 - L0) * i / 120, h = sgn * Math.pow(10, u);
          var X = ax.X(u), Y = ax.Y(C.secant(x, h));
          if (i === 0) g.moveTo(X, Y); else g.lineTo(X, Y);
        }
        g.stroke();
      });
      var uc = Math.log10(st.habs), mc = C.secant(x, hval());
      g.fillStyle = col.c[1]; g.beginPath(); g.arc(ax.X(uc), ax.Y(mc), 4.5, 0, 7); g.fill();
      g.restore();
      g.font = "11px " + sans; g.textBaseline = "bottom"; g.textAlign = "left"; g.fillStyle = col.c[0];
      g.fillText("f′(x) = " + ctx.fmt(d, 1), b.x + 4, ax.Y(d) - 3);
      // side labels sit beside the curves at |h| = 1, where each is 1 away from f'(x)
      g.fillStyle = col.muted; g.textAlign = "right";
      g.fillText("from the right", ax.X(0) - 8, ax.Y(d + 1.45));
      g.textBaseline = "top";
      g.fillText("from the left", ax.X(0) - 8, ax.Y(d - 1.45));
    }

    function draw() {
      var aspect = right.offsetTop > left.offsetTop + 10 ? 0.7 : 1.2;
      if (optS.aspect !== aspect) { optS.aspect = aspect; cvS.resize(); }
      var col = ctx.colors(), h = hval(), m = C.secant(st.x, h), d = C.fprime(st.x);
      drawF(col); drawS(col);
      ro.set({
        "h": (h < 0 ? "−" : "") + fmtH(st.habs),
        "secant slope": m.toFixed(st.habs < 0.01 ? 5 : st.habs < 0.1 ? 4 : 3).replace("-", "−"),
        "f′(x) exact": ctx.fmt(d, 3),
        "difference": (m - d).toFixed(st.habs < 0.01 ? 5 : st.habs < 0.1 ? 4 : 3).replace("-", "−")
      });
    }

    function paintLegend() {
      var col = ctx.colors();
      legend.innerHTML = "";
      [[col.ink, "f(x), and the point (x, f(x))"], [col.c[1], "secant through (x + h, f(x + h))"], [col.c[0], "tangent, slope f′(x)"]].forEach(function (s) {
        var sp = ctx.el("span", null, legend), ic = ctx.el("i", null, sp);
        ic.style.background = s[0];
        sp.appendChild(document.createTextNode(s[1]));
      });
    }

    cvF.onresize = draw; cvS.onresize = draw;
    paintLegend();
    draw();
    return { draw: function () { paintLegend(); draw(); } };
  }
});
