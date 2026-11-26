// Easing functions: map normalized time t in [0,1] to progress in [0,1].
// In animation terms these describe *spacing*: how far the drawing moves
// between frames. "Ease in" = slow start (frames bunched near the start key),
// "ease out" = slow finish (frames bunched near the end key).

export const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const invLerp = (a, b, v) => (a === b ? 0 : (v - a) / (b - a));
export const remap = (v, a0, a1, b0, b1) => lerp(b0, b1, invLerp(a0, a1, v));

export const linear = (t) => t;
export const easeInQuad = (t) => t * t;
export const easeOutQuad = (t) => 1 - (1 - t) * (1 - t);
export const easeInOutQuad = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
export const easeInCubic = (t) => t * t * t;
export const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
export const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeInOutSine = (t) => -(Math.cos(Math.PI * t) - 1) / 2;

// Overshoots the target then settles — great for "snappy" anime poses.
export const easeOutBack = (t, s = 1.70158) => 1 + (s + 1) * Math.pow(t - 1, 3) + s * Math.pow(t - 1, 2);

export const easeOutElastic = (t) => {
  if (t === 0 || t === 1) return t;
  const c4 = (2 * Math.PI) / 3;
  return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
};

export const easeOutBounce = (t) => {
  const n1 = 7.5625, d1 = 2.75;
  if (t < 1 / d1) return n1 * t * t;
  if (t < 2 / d1) return n1 * (t -= 1.5 / d1) * t + 0.75;
  if (t < 2.5 / d1) return n1 * (t -= 2.25 / d1) * t + 0.9375;
  return n1 * (t -= 2.625 / d1) * t + 0.984375;
};

// CSS-style cubic-bezier(x1, y1, x2, y2) — the same curve you edit in
// Blender's Graph Editor or Clip Studio's tween curve dialog.
export function cubicBezier(x1, y1, x2, y2) {
  const cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
  const cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
  const sampleX = (s) => ((ax * s + bx) * s + cx) * s;
  const sampleY = (s) => ((ay * s + by) * s + cy) * s;
  const dX = (s) => (3 * ax * s + 2 * bx) * s + cx;
  return (t) => {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    // Newton-Raphson, falling back to bisection.
    let s = t;
    for (let i = 0; i < 8; i++) {
      const err = sampleX(s) - t;
      if (Math.abs(err) < 1e-6) return sampleY(s);
      const d = dX(s);
      if (Math.abs(d) < 1e-6) break;
      s -= err / d;
    }
    let lo = 0, hi = 1;
    s = t;
    while (hi - lo > 1e-7) {
      const x = sampleX(s);
      if (x < t) lo = s; else hi = s;
      s = (lo + hi) / 2;
    }
    return sampleY(s);
  };
}

export const EASINGS = {
  linear, easeInQuad, easeOutQuad, easeInOutQuad, easeInCubic, easeOutCubic,
  easeInOutCubic, easeInOutSine, easeOutBack, easeOutElastic, easeOutBounce,
};

export function getEasing(nameOrFn) {
  if (typeof nameOrFn === 'function') return nameOrFn;
  if (!nameOrFn) return linear;
  const fn = EASINGS[nameOrFn];
  if (!fn) throw new Error(`Unknown easing: ${nameOrFn}`);
  return fn;
}
