/* Live playgrounds: drawing code for each playground in playgrounds/<id>.yml.
 *
 * Plots are drawn in the browser on a <canvas> (not by Python), so they
 * redraw on every slider movement. Each playground registers
 *   draw(plot, values) -> text shown under the plot
 * where `values` maps the yml's param names to slider values. The yml's
 * optional `python` code is a Python version the reader can open and run.
 *
 * Appended to _static/d2l.js through `include_js` (before exercise.js).
 */
// Leading semicolon: this file is appended to other scripts in d2l.js.
;(function () {
  'use strict';

  const COLORS = {
    text: '#1f2328', muted: '#59636e', grid: '#e7ebef', axis: '#9aa4ae',
    blue: '#1f6feb', orange: '#d9730d', green: '#1a7f37', red: '#cf222e', gray: '#8c959f',
  };

  // ------------------------------------------------------------ plot helper

  function niceStep(span, target) {
    const raw = span / target;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  }

  function fmt(v) {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e4 || a < 1e-3) return v.toExponential(0);
    return String(Number(v.toPrecision(4)));
  }

  /* A 2-D plot on a canvas with linear (or log-x) axes. */
  class Plot {
    constructor(canvas, opts) {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      // Assigning the size also resets all canvas state (clip, styles).
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      this.ctx = canvas.getContext('2d');
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.ctx.clearRect(0, 0, w, h);
      this.w = w;
      this.h = h;
      this.o = Object.assign({xlog: false, ylog: false, equal: false, xlabel: '', ylabel: ''}, opts);
      this.pad = {l: 44, r: 12, t: 10, b: this.o.xlabel ? 38 : 26};
      let [x0, x1] = this.o.xlim;
      let [y0, y1] = this.o.ylim;
      if (this.o.equal) {  // same scale on both axes
        const pw = w - this.pad.l - this.pad.r;
        const ph = h - this.pad.t - this.pad.b;
        const sx = (x1 - x0) / pw;
        const sy = (y1 - y0) / ph;
        if (sx > sy) {
          const c = (y0 + y1) / 2;
          y0 = c - sx * ph / 2; y1 = c + sx * ph / 2;
        } else {
          const c = (x0 + x1) / 2;
          x0 = c - sy * pw / 2; x1 = c + sy * pw / 2;
        }
      }
      this.xlim = [x0, x1];
      this.ylim = [y0, y1];
      this.legendItems = [];
      this.axes();
    }

    fx(x) {
      const [a, b] = this.xlim;
      const t = this.o.xlog ? (Math.log10(x) - Math.log10(a)) / (Math.log10(b) - Math.log10(a))
        : (x - a) / (b - a);
      return this.pad.l + t * (this.w - this.pad.l - this.pad.r);
    }

    fy(y) {
      const [a, b] = this.ylim;
      const t = this.o.ylog ? (Math.log10(y) - Math.log10(a)) / (Math.log10(b) - Math.log10(a))
        : (y - a) / (b - a);
      return this.h - this.pad.b - t * (this.h - this.pad.t - this.pad.b);
    }

    axes() {
      const c = this.ctx;
      c.save();
      c.font = '11px system-ui, -apple-system, Roboto, sans-serif';
      c.fillStyle = COLORS.muted;
      c.strokeStyle = COLORS.grid;
      c.lineWidth = 1;
      const [x0, x1] = this.xlim;
      const [y0, y1] = this.ylim;
      let xticks = [];
      if (this.o.xlog) {
        for (let e = Math.ceil(Math.log10(x0)); e <= Math.floor(Math.log10(x1)); e++) xticks.push(Math.pow(10, e));
      } else {
        const s = niceStep(x1 - x0, Math.max(3, Math.floor(this.w / 80)));
        for (let v = Math.ceil(x0 / s) * s; v <= x1 + 1e-9; v += s) xticks.push(Math.abs(v) < 1e-12 ? 0 : v);
      }
      const yticks = [];
      if (this.o.ylog) {
        for (let e = Math.ceil(Math.log10(y0)); e <= Math.floor(Math.log10(y1)); e++) yticks.push(Math.pow(10, e));
      } else {
        const ys = niceStep(y1 - y0, Math.max(3, Math.floor(this.h / 55)));
        for (let v = Math.ceil(y0 / ys) * ys; v <= y1 + 1e-9; v += ys) yticks.push(Math.abs(v) < 1e-12 ? 0 : v);
      }
      c.textAlign = 'center';
      c.textBaseline = 'top';
      for (const v of xticks) {
        const X = Math.round(this.fx(v)) + 0.5;
        c.beginPath(); c.moveTo(X, this.pad.t); c.lineTo(X, this.h - this.pad.b); c.stroke();
        c.fillText(fmt(v), X, this.h - this.pad.b + 5);
      }
      c.textAlign = 'right';
      c.textBaseline = 'middle';
      for (const v of yticks) {
        const Y = Math.round(this.fy(v)) + 0.5;
        c.beginPath(); c.moveTo(this.pad.l, Y); c.lineTo(this.w - this.pad.r, Y); c.stroke();
        c.fillText(fmt(v), this.pad.l - 6, Y);
      }
      // Zero axes, slightly darker, when visible.
      c.strokeStyle = COLORS.axis;
      if (!this.o.xlog && x0 < 0 && x1 > 0) {
        const X = Math.round(this.fx(0)) + 0.5;
        c.beginPath(); c.moveTo(X, this.pad.t); c.lineTo(X, this.h - this.pad.b); c.stroke();
      }
      if (!this.o.ylog && y0 < 0 && y1 > 0) {
        const Y = Math.round(this.fy(0)) + 0.5;
        c.beginPath(); c.moveTo(this.pad.l, Y); c.lineTo(this.w - this.pad.r, Y); c.stroke();
      }
      if (this.o.xlabel) {
        c.textAlign = 'center';
        c.textBaseline = 'bottom';
        c.fillText(this.o.xlabel, (this.pad.l + this.w - this.pad.r) / 2, this.h - 2);
      }
      c.restore();
      // Clip drawing to the plot area.
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.rect(this.pad.l, this.pad.t, this.w - this.pad.l - this.pad.r, this.h - this.pad.t - this.pad.b);
      this.ctx.clip();
    }

    line(xs, ys, {color = COLORS.blue, width = 2, dash = [], label = null, alpha = 1} = {}) {
      const c = this.ctx;
      c.save();
      c.globalAlpha = alpha;
      c.strokeStyle = color;
      c.lineWidth = width;
      c.setLineDash(dash);
      c.lineJoin = 'round';
      c.beginPath();
      let started = false;
      for (let i = 0; i < xs.length; i++) {
        const X = this.fx(xs[i]);
        const Y = this.fy(ys[i]);
        if (!isFinite(X) || !isFinite(Y)) { started = false; continue; }
        if (started) c.lineTo(X, Y); else { c.moveTo(X, Y); started = true; }
      }
      c.stroke();
      c.restore();
      if (label) this.legendItems.push({label, color, dash});
    }

    band(xs, lo, hi, {color = COLORS.blue, alpha = 0.15, label = null} = {}) {
      const c = this.ctx;
      c.save();
      c.globalAlpha = alpha;
      c.fillStyle = color;
      c.beginPath();
      xs.forEach((x, i) => (i ? c.lineTo(this.fx(x), this.fy(hi[i])) : c.moveTo(this.fx(x), this.fy(hi[i]))));
      for (let i = xs.length - 1; i >= 0; i--) c.lineTo(this.fx(xs[i]), this.fy(lo[i]));
      c.closePath();
      c.fill();
      c.restore();
      if (label) this.legendItems.push({label, color, band: true});
    }

    point(x, y, {color = COLORS.orange, r = 4.5} = {}) {
      const c = this.ctx;
      c.save();
      c.fillStyle = color;
      c.beginPath();
      c.arc(this.fx(x), this.fy(y), r, 0, 2 * Math.PI);
      c.fill();
      c.restore();
    }

    arrow(x0, y0, x1, y1, {color = COLORS.red, width = 2, label = null} = {}) {
      const c = this.ctx;
      const X0 = this.fx(x0), Y0 = this.fy(y0), X1 = this.fx(x1), Y1 = this.fy(y1);
      const ang = Math.atan2(Y1 - Y0, X1 - X0);
      const len = Math.hypot(X1 - X0, Y1 - Y0);
      const head = Math.min(10, len * 0.4);
      c.save();
      c.strokeStyle = c.fillStyle = color;
      c.lineWidth = width;
      c.beginPath(); c.moveTo(X0, Y0); c.lineTo(X1 - Math.cos(ang) * head * 0.6, Y1 - Math.sin(ang) * head * 0.6); c.stroke();
      c.beginPath();
      c.moveTo(X1, Y1);
      c.lineTo(X1 - head * Math.cos(ang - 0.4), Y1 - head * Math.sin(ang - 0.4));
      c.lineTo(X1 - head * Math.cos(ang + 0.4), Y1 - head * Math.sin(ang + 0.4));
      c.closePath(); c.fill();
      if (label) {
        c.font = '12px system-ui, -apple-system, Roboto, sans-serif';
        c.textAlign = 'left';
        c.textBaseline = 'bottom';
        c.fillText(label, X1 + 4, Y1 - 2);
      }
      c.restore();
    }

    legend() {
      this.ctx.restore();  // undo clip
      if (!this.legendItems.length) return;
      const c = this.ctx;
      c.save();
      c.font = '12px system-ui, -apple-system, Roboto, sans-serif';
      const rowH = 18;
      const width = Math.max(...this.legendItems.map((i) => c.measureText(i.label).width)) + 34;
      const boxH = rowH * this.legendItems.length + 6;
      const right = this.o.legend === 'lower right';
      const x = right ? this.w - this.pad.r - width - 8 : this.pad.l + 8;
      const y = right ? this.h - this.pad.b - boxH - 8 : this.pad.t + 8;
      c.fillStyle = 'rgba(255,255,255,0.9)';
      c.fillRect(x, y, width, boxH);
      this.legendItems.forEach((it, i) => {
        const Y = y + 4 + rowH * i + rowH / 2;
        if (it.band) {
          c.globalAlpha = 0.25; c.fillStyle = it.color; c.fillRect(x + 6, Y - 5, 18, 10); c.globalAlpha = 1;
        } else {
          c.strokeStyle = it.color; c.lineWidth = 2; c.setLineDash(it.dash || []);
          c.beginPath(); c.moveTo(x + 6, Y); c.lineTo(x + 24, Y); c.stroke(); c.setLineDash([]);
        }
        c.fillStyle = COLORS.text;
        c.textBaseline = 'middle';
        c.fillText(it.label, x + 30, Y);
      });
      c.restore();
    }
  }

  function linspace(a, b, n) {
    return Array.from({length: n}, (_, i) => a + (b - a) * i / (n - 1));
  }

  function logspace(a, b, n) {
    return linspace(a, b, n).map((e) => Math.pow(10, e));
  }

  // Small seeded random generator (mulberry32), so a seed gives a fixed run.
  function rng(seed) {
    let s = (seed >>> 0) + 0x9e3779b9;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const n4 = (v) => (Math.round(v * 1e4) / 1e4).toFixed(4);

  // Standard normal samples (Box-Muller) from a uniform generator.
  function gaussian(rand) {
    let u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  /* Ridge least squares: minimize sum (Phi w - y)^2 + lambda ||w||^2 by
   * solving (Phi^T Phi + lambda I) w = Phi^T y with Gaussian elimination. */
  function ridge(Phi, y, lambda) {
    const d = Phi[0].length;
    const A = Array.from({length: d}, (_, i) => Array.from({length: d + 1}, (_, j) => {
      if (j === d) return Phi.reduce((acc, row, k) => acc + row[i] * y[k], 0);
      return Phi.reduce((acc, row) => acc + row[i] * row[j], 0) + (i === j ? lambda : 0);
    }));
    for (let c = 0; c < d; c++) {
      let piv = c;
      for (let r = c + 1; r < d; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]];
      for (let r = 0; r < d; r++) {
        if (r === c || A[c][c] === 0) continue;
        const f = A[r][c] / A[c][c];
        for (let j = c; j <= d; j++) A[r][j] -= f * A[c][j];
      }
    }
    return A.map((row, i) => row[d] / row[i]);
  }

  const polyFeatures = (x, degree) => Array.from({length: degree + 1}, (_, k) => Math.pow(x, k));
  const mse = (pred, y) => pred.reduce((acc, p, i) => acc + (p - y[i]) ** 2, 0) / y.length;

  // ------------------------------------------------------------ playgrounds

  const registry = {};

  registry['matrix-transform'] = {
    aspect: 1,
    draw(canvas, {a, b, c, d}) {
      const t = linspace(0, 2 * Math.PI, 160);
      const cx = t.map(Math.cos), cy = t.map(Math.sin);
      const sq = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]];
      const A = (x, y) => [a * x + b * y, c * x + d * y];
      const tc = t.map((_, i) => A(cx[i], cy[i]));
      const ts = sq.map(([x, y]) => A(x, y));
      const lim = Math.max(2.2, ...tc.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y)) * 1.15));
      const p = new Plot(canvas, {xlim: [-lim, lim], ylim: [-lim, lim], equal: true});
      p.line(cx, cy, {color: COLORS.gray, width: 1.5, label: 'input x'});
      p.line(sq.map((q) => q[0]), sq.map((q) => q[1]), {color: COLORS.gray, width: 1.5});
      p.line(tc.map((q) => q[0]), tc.map((q) => q[1]), {color: COLORS.blue, width: 2.5, label: 'output Ax'});
      p.line(ts.map((q) => q[0]), ts.map((q) => q[1]), {color: COLORS.blue, width: 2.5});
      p.arrow(0, 0, a, c, {color: COLORS.orange, label: 'A e₁'});
      p.arrow(0, 0, b, d, {color: COLORS.green, label: 'A e₂'});
      p.legend();
      return `A = [[${a.toFixed(1)}, ${b.toFixed(1)}], [${c.toFixed(1)}, ${d.toFixed(1)}]]` +
        `   ·   A e₁ = (${a.toFixed(1)}, ${c.toFixed(1)}) is the first column, A e₂ = (${b.toFixed(1)}, ${d.toFixed(1)}) the second`;
    },
  };

  registry['lp-norm-balls'] = {
    aspect: 1,
    draw(canvas, {p, x1, x2}) {
      const ball = (q) => {
        const t = linspace(0, 2 * Math.PI, 400);
        return t.map((s) => {
          const u = Math.cos(s), v = Math.sin(s);
          const n = Math.pow(Math.pow(Math.abs(u), q) + Math.pow(Math.abs(v), q), 1 / q);
          return [u / n, v / n];
        });
      };
      const P = new Plot(canvas, {xlim: [-1.4, 1.4], ylim: [-1.4, 1.4], equal: true});
      for (const [q, dash] of [[1, [6, 4]], [2, [2, 3]]]) {
        const b = ball(q);
        P.line(b.map((z) => z[0]), b.map((z) => z[1]), {color: COLORS.gray, width: 1.2, dash, label: `p = ${q}`});
      }
      const b = ball(p);
      P.line(b.map((z) => z[0]), b.map((z) => z[1]), {color: COLORS.blue, width: 2.5, label: `p = ${p}`});
      P.legend();
      const norm = (q) => (q === Infinity ? Math.max(Math.abs(x1), Math.abs(x2))
        : Math.pow(Math.pow(Math.abs(x1), q) + Math.pow(Math.abs(x2), q), 1 / q));
      return `x = (${x1}, ${x2}):   ‖x‖₁ = ${norm(1).toFixed(3)}   ‖x‖₂ = ${norm(2).toFixed(3)}   ` +
        `‖x‖_${p} = ${norm(p).toFixed(3)}   ‖x‖_∞ = ${norm(Infinity).toFixed(3)}`;
    },
  };

  registry['derivative-limit'] = {
    aspect: 0.62,
    draw(canvas, {x0, h}) {
      const f = (x) => 3 * x * x - 4 * x;
      const xs = linspace(-1.5, 3.5, 200);
      const secant = (f(x0 + h) - f(x0)) / h;
      const exact = 6 * x0 - 4;
      const P = new Plot(canvas, {xlim: [-1.5, 3.5], ylim: [-4, 20], xlabel: 'x'});
      P.line(xs, xs.map(f), {color: COLORS.blue, label: 'f(x) = 3x² − 4x'});
      P.line(xs, xs.map((x) => f(x0) + secant * (x - x0)), {color: COLORS.orange, label: `secant, slope ${secant.toFixed(3)}`});
      P.line(xs, xs.map((x) => f(x0) + exact * (x - x0)), {color: COLORS.green, dash: [6, 4], label: `tangent, slope ${exact.toFixed(3)}`});
      P.point(x0, f(x0));
      P.point(x0 + h, f(x0 + h));
      P.legend();
      return `secant slope = ${secant.toFixed(5)}   exact derivative f'(x₀) = ${exact.toFixed(5)}   ` +
        `error = ${(secant - exact).toFixed(5)} (exactly 3h)`;
    },
  };

  registry['coin-convergence'] = {
    aspect: 0.6,
    draw(canvas, {p, n, seed}) {
      const rand = rng(seed);
      const ks = [], est = [], lo = [], hi = [];
      let heads = 0;
      for (let k = 1; k <= n; k++) {
        heads += rand() < p ? 1 : 0;
        // Keep about 600 points for drawing, spaced evenly on the log axis.
        if (k === n || k < 20 || Math.floor(Math.log(k) * 120) !== Math.floor(Math.log(k + 1) * 120)) {
          const sd = Math.sqrt(p * (1 - p) / k);
          ks.push(k); est.push(heads / k); lo.push(p - 2 * sd); hi.push(p + 2 * sd);
        }
      }
      const P = new Plot(canvas, {xlim: [1, Math.max(10, n)], ylim: [0, 1], xlog: true, xlabel: 'number of tosses k (log scale)'});
      P.band(ks, lo, hi, {label: 'p ± 2 std'});
      P.line([1, n], [p, p], {color: COLORS.text, width: 1, dash: [5, 4], label: 'true p'});
      P.line(ks, est, {color: COLORS.blue, label: 'running estimate'});
      P.legend();
      return `after ${n} tosses: estimate = ${n4(heads / n)}, true p = ${p}, ` +
        `expected std = ${n4(Math.sqrt(p * (1 - p) / n))}`;
    },
  };

  registry['bayes-test'] = {
    aspect: 0.6,
    draw(canvas, {prevalence, sensitivity, fpr}) {
      const post = (q) => sensitivity * q / (sensitivity * q + fpr * (1 - q));
      const qs = logspace(-4, Math.log10(0.5), 300);
      const P = new Plot(canvas, {xlim: [1e-4, 0.5], ylim: [0, 1], xlog: true, legend: 'lower right',
        xlabel: 'prevalence P(H=1) (log scale)'});
      P.line(qs, qs.map(post), {color: COLORS.blue, label: 'P(disease | positive)'});
      const two = (q) => sensitivity ** 2 * q / (sensitivity ** 2 * q + fpr ** 2 * (1 - q));
      P.line(qs, qs.map(two), {color: COLORS.green, dash: [6, 4], label: 'after 2 independent positives'});
      P.point(prevalence, post(prevalence), {color: COLORS.orange});
      P.point(prevalence, two(prevalence), {color: COLORS.green});
      P.legend();
      return `P(disease | 1 positive) = ${n4(post(prevalence))}   ·   ` +
        `P(disease | 2 positives) = ${n4(two(prevalence))}`;
    },
  };

  // ------------------------------------------------- Chapter 3 playgrounds

  registry['sgd-learning-rate'] = {
    aspect: 0.6,
    draw(canvas, {lr, batch, epochs}) {
      // Same setup as the section: y = X [2, -3.4] + 4.2 + noise, 1000 examples.
      const rand = rng(7);
      const n = 1000, wTrue = [2, -3.4], bTrue = 4.2;
      const X = Array.from({length: n}, () => [gaussian(rand), gaussian(rand)]);
      const y = X.map((x) => x[0] * wTrue[0] + x[1] * wTrue[1] + bTrue + 0.01 * gaussian(rand));
      const fullLoss = (w, b) => X.reduce((acc, x, i) => acc + 0.5 * (x[0] * w[0] + x[1] * w[1] + b - y[i]) ** 2, 0) / n;
      let w = [0, 0], b = 0, step = 0;
      const steps = [0], losses = [fullLoss(w, b)];
      const perEpoch = Math.ceil(n / batch);
      const every = Math.max(1, Math.floor(perEpoch * epochs / 300));
      const order = Array.from({length: n}, (_, i) => i);
      let diverged = false;
      for (let e = 0; e < epochs && !diverged; e++) {
        for (let i = n - 1; i > 0; i--) {  // shuffle each epoch
          const j = Math.floor(rand() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
        for (let s0 = 0; s0 < n; s0 += batch) {
          const idx = order.slice(s0, s0 + batch);
          let g0 = 0, g1 = 0, gb = 0;
          for (const i of idx) {
            const r = X[i][0] * w[0] + X[i][1] * w[1] + b - y[i];
            g0 += r * X[i][0]; g1 += r * X[i][1]; gb += r;
          }
          w = [w[0] - lr * g0 / idx.length, w[1] - lr * g1 / idx.length];
          b -= lr * gb / idx.length;
          step++;
          if (step % every === 0) {
            const l = fullLoss(w, b);
            if (!isFinite(l) || l > 1e6) { diverged = true; steps.push(step); losses.push(1e6); break; }
            steps.push(step); losses.push(l);
          }
        }
      }
      const P = new Plot(canvas, {xlim: [0, Math.max(10, steps[steps.length - 1])], ylim: [1e-5, 1e3],
        ylog: true, xlabel: 'SGD steps (minibatch updates)', legend: 'lower right'});
      P.line([0, steps[steps.length - 1]], [0.5e-4, 0.5e-4], {color: COLORS.gray, width: 1, dash: [4, 4], label: 'noise floor (σ²/2)'});
      P.line(steps, losses.map((l) => Math.max(l, 1.1e-5)), {color: diverged ? COLORS.red : COLORS.blue,
        label: diverged ? 'training loss (diverged!)' : 'training loss'});
      P.legend();
      if (diverged) return `lr = ${lr} is too large: the updates overshoot and the loss explodes. Try a smaller learning rate.`;
      return `after ${step} steps: w = (${w[0].toFixed(3)}, ${w[1].toFixed(3)}) vs true (2, -3.4), ` +
        `b = ${b.toFixed(3)} vs 4.2, loss = ${losses[losses.length - 1].toExponential(2)}`;
    },
  };

  registry['loss-shapes'] = {
    aspect: 0.6,
    draw(canvas, {sigma, outlier}) {
      const rs = linspace(-4, 4, 400);
      const huber = (r) => (Math.abs(r) > sigma ? Math.abs(r) - sigma / 2 : r * r / (2 * sigma));
      const P = new Plot(canvas, {xlim: [-4, 4], ylim: [0, 6], xlabel: 'residual  ŷ − y'});
      P.line(rs, rs.map((r) => r * r / 2), {color: COLORS.blue, label: 'squared  ½r²'});
      P.line(rs, rs.map(Math.abs), {color: COLORS.orange, label: 'absolute  |r|'});
      P.line(rs, rs.map(huber), {color: COLORS.green, width: 2.5, label: `Huber (σ = ${sigma})`});
      P.legend();
      // Best constant b for data [1, 2, 3, 4, outlier] under each loss.
      const data = [1, 2, 3, 4, outlier];
      const mean = data.reduce((a, v) => a + v, 0) / data.length;
      const median = [...data].sort((a, v) => a - v)[2];
      let hb = median;  // minimize Huber loss by gradient steps on b
      for (let i = 0; i < 400; i++) {
        const g = data.reduce((a, v) => { const r = hb - v; return a + (Math.abs(r) > sigma ? Math.sign(r) : r / sigma); }, 0);
        hb -= 0.05 * g;
      }
      return `fit one number b to data [1, 2, 3, 4, ${outlier}]:  squared → ${mean.toFixed(2)} (mean),  ` +
        `absolute → ${median.toFixed(2)} (median),  Huber → ${hb.toFixed(2)}`;
    },
  };

  function polyTask(seed, nTrain, noise) {
    const rand = rng(seed);
    const f = (x) => Math.sin(Math.PI * x) + 0.3 * x;
    const xs = Array.from({length: nTrain}, () => 2 * rand() - 1);
    const ys = xs.map((x) => f(x) + noise * gaussian(rand));
    const vx = linspace(-1, 1, 200);
    const vy = vx.map((x) => f(x) + noise * gaussian(rand));
    return {f, xs, ys, vx, vy};
  }

  function drawPolyFit(canvas, task, degree, lambda) {
    const {f, xs, ys, vx, vy} = task;
    const w = ridge(xs.map((x) => polyFeatures(x, degree)), ys, lambda);
    const predict = (x) => polyFeatures(x, degree).reduce((acc, v, k) => acc + v * w[k], 0);
    const grid = linspace(-1, 1, 300);
    const P = new Plot(canvas, {xlim: [-1, 1], ylim: [-2.2, 2.2], xlabel: 'x'});
    P.line(grid, grid.map(f), {color: COLORS.gray, width: 1.5, dash: [5, 4], label: 'true function'});
    P.line(grid, grid.map(predict), {color: COLORS.blue, width: 2.5, label: `degree-${degree} fit`});
    xs.forEach((x, i) => P.point(x, ys[i], {color: COLORS.orange, r: 3.5}));
    P.legend();
    return {
      train: mse(xs.map(predict), ys), val: mse(vx.map(predict), vy),
      norm: Math.sqrt(w.slice(1).reduce((a, v) => a + v * v, 0)),
    };
  }

  registry['polynomial-fit'] = {
    aspect: 0.6,
    draw(canvas, {degree, n, noise, seed}) {
      const r = drawPolyFit(canvas, polyTask(seed, n, noise), degree, 1e-9);
      const verdict = r.val > 3 * Math.max(r.train, noise * noise) && degree > 3 ? ' → overfitting'
        : r.train > 0.05 && degree < 3 ? ' → underfitting' : '';
      return `training error ${r.train.toFixed(4)}   ·   validation error ${r.val.toFixed(4)}${verdict}`;
    },
  };

  registry['weight-decay-poly'] = {
    aspect: 0.6,
    draw(canvas, {log_lambda}) {
      const lambda = Math.pow(10, log_lambda);
      const r = drawPolyFit(canvas, polyTask(3, 15, 0.25), 12, lambda);
      return `λ = ${lambda.toExponential(1)}:   training error ${r.train.toFixed(4)}   ·   ` +
        `validation error ${r.val.toFixed(4)}   ·   ‖w‖ = ${r.norm.toFixed(2)}`;
    },
  };

  window.D2LPlaygrounds = registry;
})();
