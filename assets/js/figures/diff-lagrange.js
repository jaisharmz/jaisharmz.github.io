FIG.register("diff-lagrange", {
  defaults: { thetaDeg: 150, c: 4 },

  // Pure math, no DOM. Notation follows the text: maximize f(x, y) subject to
  // g(x, y) = c. Here f(x, y) = x + 2y and g(x, y) = x^2 + 4y^2 with c = 4, an
  // ellipse with semi-axes 2 and 1, walked as (2 cos theta, sin theta).
  compute: {
    f: function (x, y) { return x + 2 * y; },
    gradF: function () { return [1, 2]; },
    g: function (x, y) { return x * x + 4 * y * y; },
    gradG: function (x, y) { return [2 * x, 8 * y]; },
    point: function (thetaDeg) {
      var t = thetaDeg * Math.PI / 180;
      return [2 * Math.cos(t), Math.sin(t)];
    },

    // Everything the figure shows at one point of the constraint.
    at: function (thetaDeg) {
      var t = thetaDeg * Math.PI / 180;
      var p = this.point(thetaDeg), x = p[0], y = p[1];
      var gf = this.gradF(), gg = this.gradG(x, y);
      var tx = -2 * Math.sin(t), ty = Math.cos(t), tn = Math.hypot(tx, ty);
      tx /= tn; ty /= tn;                                  // unit tangent, toward larger theta
      var nf = Math.hypot(gf[0], gf[1]), ng = Math.hypot(gg[0], gg[1]);
      var cos = (gf[0] * gg[0] + gf[1] * gg[1]) / (nf * ng);
      var angle = Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
      return {
        theta: thetaDeg, x: x, y: y, f: this.f(x, y), g: this.g(x, y),
        gradF: gf, gradG: gg, tangent: [tx, ty],
        angle: angle,                                      // between grad f and grad g, degrees
        along: gf[0] * tx + gf[1] * ty,                    // grad f . tangent: df per unit arc length
        lambda: (gf[0] * gg[0] + gf[1] * gg[1]) / (ng * ng), // the multiplier that best fits grad f = lambda grad g
        parallel: angle < 0.25 || angle > 179.75
      };
    },

    // f around the whole constraint, for the side plot.
    profile: function (steps) {
      var out = [];
      for (var i = 0; i <= steps; i++) {
        var th = 360 * i / steps, p = this.point(th);
        out.push([th, this.f(p[0], p[1])]);
      }
      return out;
    },

    // The constrained maximum and minimum: scan theta, then golden-section search.
    optima: function () {
      var self = this;
      function fth(th) { var p = self.point(th); return self.f(p[0], p[1]); }
      function refine(th0, sign) {
        var a = th0 - 1, b = th0 + 1, r = (Math.sqrt(5) - 1) / 2;
        for (var i = 0; i < 80; i++) {
          var c = b - r * (b - a), d = a + r * (b - a);
          if (sign * fth(c) > sign * fth(d)) b = d; else a = c;
        }
        var th = (a + b) / 2;
        return ((th % 360) + 360) % 360;
      }
      var best = -Infinity, worst = Infinity, tb = 0, tw = 0;
      for (var i = 0; i < 3600; i++) {
        var th = i / 10, v = fth(th);
        if (v > best) { best = v; tb = th; }
        if (v < worst) { worst = v; tw = th; }
      }
      return { max: this.at(refine(tb, 1)), min: this.at(refine(tw, -1)) };
    }
  },

  mount: function (stage, ctx) {
    var C = this.compute, D = this.defaults;
    var opt = C.optima();
    var prof = C.profile(360);
    var theta = D.thetaDeg;

    var row = ctx.controls(stage);
    var sT = ctx.slider(row, {
      label: "point on the curve θ", min: 0, max: 360, step: 0.5, value: theta,
      fmt: function (v) { return v.toFixed(1) + "°"; },
      oninput: function (v) { stopTween(); theta = v; draw(); }
    });
    var bMax = ctx.button(row, "jump to maximum", function () { tweenTo(opt.max.theta); });
    var bMin = ctx.button(row, "jump to minimum", function () { tweenTo(opt.min.theta); });
    bMax.classList.add("fig__btn--primary");

    var cv = ctx.canvas(stage, { aspect: 0.5, minHeight: 200, maxHeight: 360,
      label: "Level lines of f, the constraint ellipse, and the two gradients at a point you can drag" });
    var legend = ctx.el("div", "fig__legend", stage);
    function key(text, cls) {
      var s = ctx.el("span", null, legend), i = ctx.el("i", cls || null, s);
      s.appendChild(document.createTextNode(text));
      return i;
    }
    var kF = key("direction of ∇f"), kG = key("direction of ∇g"),
        kA = key("part of ∇f along the curve"), kMax = key("maximum", "dot"), kMin = key("minimum", "dot");

    var cv2 = ctx.canvas(stage, { aspect: 0.24, minHeight: 130, maxHeight: 170,
      label: "f along the constraint as a function of theta" });
    cv2.wrap.style.marginTop = ".6rem";
    var ro = ctx.readout(stage);

    cv.onresize = draw;
    cv2.onresize = draw;
    var font = getComputedStyle(document.documentElement).getPropertyValue("--sans");

    // equal scale on both axes, so right angles look like right angles
    var view = null;
    function makeView() {
      var m = { l: 8, r: 8, t: 8, b: 8 };
      var bw = cv.w - m.l - m.r, bh = cv.h - m.t - m.b;
      var ppu = Math.min(bw / 5.9, bh / 3.0);
      var cx = m.l + bw / 2, cy = m.t + bh / 2;
      return {
        ppu: ppu, box: { x: m.l, y: m.t, w: bw, h: bh },
        X: function (x) { return cx + x * ppu; }, Y: function (y) { return cy - y * ppu; },
        invX: function (px) { return (px - cx) / ppu; }, invY: function (py) { return (cy - py) / ppu; }
      };
    }

    function arrow(g, x0, y0, dx, dy, color, width) {
      var L = Math.hypot(dx, dy);
      if (L < 1) return;
      var ux = dx / L, uy = dy / L, hl = Math.min(10, L * 0.45), hw = hl * 0.55;
      var x1 = x0 + dx, y1 = y0 + dy;
      g.strokeStyle = color; g.fillStyle = color; g.lineWidth = width || 2.2; g.lineCap = "round";
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1 - ux * hl * 0.8, y1 - uy * hl * 0.8); g.stroke();
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x1 - ux * hl - uy * hw, y1 - uy * hl + ux * hw);
      g.lineTo(x1 - ux * hl + uy * hw, y1 - uy * hl - ux * hw);
      g.closePath(); g.fill();
    }

    function label(g, text, x, y, color, align) {
      g.font = "11px " + font; g.textAlign = align || "center"; g.textBaseline = "middle";
      g.lineWidth = 3; g.strokeStyle = ctx.colors().paper; g.lineJoin = "round";
      g.strokeText(text, x, y); g.fillStyle = color; g.fillText(text, x, y);
    }

    function drawMain(col, a) {
      var g = cv.g, v = view = makeView(), b = v.box;
      cv.clear();
      g.save();
      g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();

      // f shading: f is linear, so its heat map is a linear gradient along grad f
      var gf = C.gradF(), nf = Math.hypot(gf[0], gf[1]);
      var ux = gf[0] / nf, uy = gf[1] / nf, span = 3.4;
      var grad = g.createLinearGradient(v.X(-ux * span), v.Y(-uy * span), v.X(ux * span), v.Y(uy * span));
      grad.addColorStop(0, ctx.alpha(col.c[1], 0));
      grad.addColorStop(1, ctx.alpha(col.c[1], 0.24));
      g.fillStyle = grad; g.fillRect(b.x, b.y, b.w, b.h);

      // level lines x + 2y = k
      var xr = v.invX(b.x + b.w) + 0.1, yr = v.invY(b.y) + 0.1;
      g.lineWidth = 1;
      for (var k = -8; k <= 8; k++) {
        g.strokeStyle = ctx.alpha(col.c[1], k === 0 ? 0.55 : 0.32);
        g.beginPath();
        g.moveTo(v.X(-xr), v.Y((k + xr) / 2)); g.lineTo(v.X(xr), v.Y((k - xr) / 2));
        g.stroke();
      }
      // label the level lines where they meet the side edges: negative f on
      // the left, positive f on the right, away from the arrows at the maximum
      var xl = v.invX(b.x), xR = v.invX(b.x + b.w);
      for (k = -6; k <= 6; k += 2) {
        var side = k < 0 ? xl : xR, yk = (k - side) / 2, py = v.Y(yk);
        if (py < b.y + 10 || py > b.y + b.h - 10) continue;
        label(g, "f = " + String(k).replace("-", "\u2212"), k < 0 ? b.x + 5 : b.x + b.w - 5, py - 7, col.muted, k < 0 ? "left" : "right");
      }

      // the constraint g(x, y) = c
      g.strokeStyle = col.ink; g.lineWidth = 2;
      g.beginPath(); g.ellipse(v.X(0), v.Y(0), 2 * v.ppu, 1 * v.ppu, 0, 0, 2 * Math.PI); g.stroke();
      // the curve's name goes in whichever of two spots is farther from the point
      var spot = Math.hypot(a.x - 1.6, a.y + 0.9) > Math.hypot(a.x + 1.6, a.y - 0.9)
        ? [1.42, -0.98, "left"] : [-1.42, 0.98, "right"];
      label(g, "g(x, y) = c", v.X(spot[0]), v.Y(spot[1]), col.ink, spot[2]);

      // the two parallel points
      [[opt.max, col.c[2], "maximum, λ > 0"], [opt.min, col.c[3], "minimum, λ < 0"]].forEach(function (o) {
        var p = o[0];
        g.fillStyle = o[1]; g.strokeStyle = col.paper; g.lineWidth = 1.5;
        g.beginPath(); g.arc(v.X(p.x), v.Y(p.y), 5, 0, 2 * Math.PI); g.fill(); g.stroke();
        // inside the ellipse, clear of the outward arrows and of the canvas edge
        // (sideways along the major axis, where the steep grad f arrow never goes)
        var lx = v.X(p.x - Math.sign(p.x) * 0.3), ly = v.Y(p.y - Math.sign(p.y) * 0.13);
        label(g, o[2], lx, ly, o[1], p.x > 0 ? "right" : "left");
      });

      // the point, the tangent, and the arrows
      var X0 = v.X(a.x), Y0 = v.Y(a.y), L = Math.max(38, Math.min(64, v.ppu * 0.75));
      g.setLineDash([4, 4]); g.strokeStyle = col.muted; g.lineWidth = 1.2;
      g.beginPath();
      g.moveTo(X0 - a.tangent[0] * L * 1.5, Y0 + a.tangent[1] * L * 1.5);
      g.lineTo(X0 + a.tangent[0] * L * 1.5, Y0 - a.tangent[1] * L * 1.5);
      g.stroke(); g.setLineDash([]);

      var ngg = Math.hypot(a.gradG[0], a.gradG[1]);
      var fdx = gf[0] / nf * L, fdy = -gf[1] / nf * L;
      var gdx = a.gradG[0] / ngg * L * 0.72, gdy = -a.gradG[1] / ngg * L * 0.72;
      var along = a.along / nf;                            // cos of the angle between grad f and the tangent
      var adx = a.tangent[0] * along * L, ady = -a.tangent[1] * along * L;
      arrow(g, X0, Y0, fdx, fdy, col.c[1], 2.4);
      arrow(g, X0, Y0, gdx, gdy, col.c[0], 2.4);
      arrow(g, X0, Y0, adx, ady, col.c[4], 3);

      // arrow labels, pushed apart when the arrows nearly overlap
      var sep = a.angle < 20 || a.angle > 160 ? 9 : 0;
      var nx = -fdy / L, ny = fdx / L;
      label(g, "∇f", X0 + fdx * 1.22 + nx * sep, Y0 + fdy * 1.22 + ny * sep, col.c[1]);
      label(g, "∇g", X0 + gdx * 1.3 - nx * sep, Y0 + gdy * 1.3 - ny * sep, col.c[0]);

      g.fillStyle = col.ink; g.strokeStyle = col.paper; g.lineWidth = 2;
      g.beginPath(); g.arc(X0, Y0, 6, 0, 2 * Math.PI); g.fill(); g.stroke();
      g.restore();
    }

    function drawSide(col, a) {
      var g = cv2.g;
      cv2.clear();
      var ax = ctx.axes(cv2, {
        xmin: 0, xmax: 360, ymin: -3.3, ymax: 3.3,
        xlabel: "position on the curve θ (degrees)",
        margin: { l: 42, r: 10, t: 8, b: 32 },
        xticks: cv2.w < 420 ? [0, 90, 180, 270, 360] : [0, 45, 90, 135, 180, 225, 270, 315, 360],
        yticks: [-2, 0, 2]
      });
      ax.draw();
      g.save();
      g.font = "italic 12px " + font; g.fillStyle = col.muted; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText("f", 10, ax.box.y + ax.box.h / 2);
      g.strokeStyle = col.c[1]; g.lineWidth = 2;
      g.beginPath();
      prof.forEach(function (q, i) { if (i === 0) g.moveTo(ax.X(q[0]), ax.Y(q[1])); else g.lineTo(ax.X(q[0]), ax.Y(q[1])); });
      g.stroke();
      [[opt.max, col.c[2]], [opt.min, col.c[3]]].forEach(function (o) {
        g.fillStyle = o[1]; g.strokeStyle = col.paper; g.lineWidth = 1.5;
        g.beginPath(); g.arc(ax.X(o[0].theta), ax.Y(o[0].f), 4.5, 0, 2 * Math.PI); g.fill(); g.stroke();
      });
      // slope of f along the curve at the current point, as a short tangent segment
      var h = 0.01, p1 = C.point(a.theta - h), p2 = C.point(a.theta + h);
      var dfd = (C.f(p2[0], p2[1]) - C.f(p1[0], p1[1])) / (2 * h);   // df / dtheta, per degree
      var sx = ax.box.w / 360, sy = ax.box.h / 6.6;
      var vx = 1, vy = -dfd * sy / sx, vn = Math.hypot(vx, vy), half = 26;
      g.strokeStyle = col.c[4]; g.lineWidth = 2.5;
      g.beginPath();
      g.moveTo(ax.X(a.theta) - vx / vn * half, ax.Y(a.f) - vy / vn * half);
      g.lineTo(ax.X(a.theta) + vx / vn * half, ax.Y(a.f) + vy / vn * half);
      g.stroke();
      g.strokeStyle = ctx.alpha(col.ink, 0.35); g.lineWidth = 1;
      g.beginPath(); g.moveTo(ax.X(a.theta) + .5, ax.box.y); g.lineTo(ax.X(a.theta) + .5, ax.box.y + ax.box.h); g.stroke();
      g.fillStyle = col.ink; g.strokeStyle = col.paper; g.lineWidth = 2;
      g.beginPath(); g.arc(ax.X(a.theta), ax.Y(a.f), 5, 0, 2 * Math.PI); g.fill(); g.stroke();
      g.restore();
      cv2.ax = ax;
    }

    function draw() {
      var col = ctx.colors();
      var a = C.at(theta);
      drawMain(col, a);
      drawSide(col, a);
      kF.style.background = col.c[1]; kG.style.background = col.c[0]; kA.style.background = col.c[4];
      kMax.style.background = col.c[2]; kMin.style.background = col.c[3];
      ro.set({
        "point": "(" + ctx.fmt(a.x, 2) + ", " + ctx.fmt(a.y, 2) + ")",
        "f": ctx.fmt(a.f, 3),
        "angle from \u2207f to \u2207g": a.angle.toFixed(1) + "\u00B0",
        "slope along the curve": Math.abs(a.along).toFixed(3),
        "λ": a.parallel ? ctx.fmt(a.lambda, 3) : "not parallel"
      });
    }

    // dragging: the pointer's angle around the ellipse picks the point
    function thetaFromMain(e) {
      var r = cv.el.getBoundingClientRect();
      var x = view.invX(e.clientX - r.left), y = view.invY(e.clientY - r.top);
      var th = Math.atan2(y, x / 2) * 180 / Math.PI;
      return (th + 360) % 360;
    }
    function thetaFromSide(e) {
      var r = cv2.el.getBoundingClientRect();
      var th = cv2.ax.invX(e.clientX - r.left);
      return Math.max(0, Math.min(360, th));
    }
    function setTheta(th) { theta = Math.round(th * 10) / 10; sT.set(theta); draw(); }
    function bindDrag(el, pick) {
      var dragging = false;
      el.style.cursor = "grab";
      el.addEventListener("pointerdown", function (e) {
        dragging = true; stopTween(); el.style.cursor = "grabbing";
        if (el.setPointerCapture) el.setPointerCapture(e.pointerId);
        setTheta(pick(e));
      });
      el.addEventListener("pointermove", function (e) { if (dragging) setTheta(pick(e)); });
      function end() { dragging = false; el.style.cursor = "grab"; }
      el.addEventListener("pointerup", end);
      el.addEventListener("pointercancel", end);
    }
    bindDrag(cv.el, thetaFromMain);
    bindDrag(cv2.el, thetaFromSide);

    // the jump buttons glide along the shorter way round
    var raf = 0;
    function stopTween() { if (raf) cancelAnimationFrame(raf); raf = 0; }
    function tweenTo(target) {
      stopTween();
      if (ctx.reducedMotion) { setTheta(target); return; }
      var from = theta, delta = ((target - from + 540) % 360) - 180, t0 = 0;
      function frame(t) {
        if (!t0) t0 = t;
        var u = Math.min(1, (t - t0) / 500), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        theta = ((from + delta * e) % 360 + 360) % 360;
        if (u >= 1) theta = target;
        sT.set(Math.round(theta * 10) / 10);
        draw();
        raf = u < 1 ? requestAnimationFrame(frame) : 0;
      }
      raf = requestAnimationFrame(frame);
    }

    draw();
    return { draw: draw };
  }
});
