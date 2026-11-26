// Cel shading ("anime shading"): instead of a smooth gradient, light is
// quantized into 2–3 flat tones with hard edges — base, shadow, and sometimes
// a second darker shadow and a crisp highlight. This is what a "toon shader"
// does in Blender (Shader to RGB -> Color Ramp set to Constant) and what you
// paint by hand in Clip Studio with a "shadow" layer set to Multiply.

import { hexToRgb, shadowColor, highlightColor } from '../utils/color.js';

const norm = (v) => {
  const l = Math.hypot(...v) || 1;
  return v.map((c) => c / l);
};
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Toon ramp: pick a color index from a lighting value.
 * @param {number} ndl N·L in [-1, 1]
 * @param {number[]} thresholds ascending, e.g. [0.0] for 2 tones, [-0.3, 0.25] for 3
 * @returns {number} band index 0 (darkest) .. thresholds.length (lightest)
 */
export function toonBand(ndl, thresholds) {
  let i = 0;
  while (i < thresholds.length && ndl > thresholds[i]) i++;
  return i;
}

/**
 * Render a lit sphere, either smooth (realistic Lambert) or cel-shaded.
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} o
 * @param {number} o.cx
 * @param {number} o.cy
 * @param {number} o.r
 * @param {number[]} [o.light] direction TO the light, [x, y, z] (y down, z toward viewer)
 * @param {'smooth'|'cel'} [o.mode]
 * @param {number} [o.bands] 2 or 3 tones
 * @param {number} [o.threshold] where the main shadow edge falls (-1..1)
 * @param {string} [o.base]
 * @param {boolean} [o.highlight] add a specular "pop" highlight
 * @param {boolean} [o.rim] add a rim light on the shadow side
 * @param {number} [o.outline] outline width in px
 * @param {boolean} [o.hueShift] use hue-shifted shadows (true) or plain darkening (false)
 */
export function renderShadedSphere(ctx, o) {
  const {
    cx, cy, r, light = [-0.6, -0.7, 0.6], mode = 'cel', bands = 2, threshold = 0.1,
    base = '#f6a5c0', highlight = true, rim = false, outline = 3, hueShift = true,
  } = o;
  const L = norm(light);
  const shadowHex = hueShift ? shadowColor(base) : shadowColor(base, { hueShift: 0, saturate: 0 });
  const deepHex = hueShift ? shadowColor(base, { darken: 0.32, hueShift: 0.4 }) : shadowColor(base, { darken: 0.32, hueShift: 0, saturate: 0 });
  const palette = {
    base: hexToRgb(base),
    shadow: hexToRgb(shadowHex),
    deep: hexToRgb(deepHex),
    hi: hexToRgb(highlightColor(base, { lighten: 0.25 })),
  };
  const thresholds = bands >= 3 ? [threshold - 0.45, threshold] : [threshold];
  const colors = bands >= 3 ? [palette.deep, palette.shadow, palette.base] : [palette.shadow, palette.base];
  const H = norm([L[0], L[1], L[2] + 1]); // half vector for viewer at +z

  const size = Math.ceil(r * 2) + 2;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const x = (px - size / 2) / r, y = (py - size / 2) / r;
      const rr = x * x + y * y;
      if (rr > 1) continue;
      const N = [x, y, Math.sqrt(1 - rr)];
      const ndl = dot(N, L);
      let c;
      if (mode === 'smooth') {
        const k = Math.max(0, ndl) * 0.75 + 0.25;
        c = palette.base.map((v, i) => palette.deep[i] + (v - palette.deep[i]) * k);
        const spec = Math.pow(Math.max(0, dot(N, H)), 40);
        if (highlight) c = c.map((v) => v + (255 - v) * spec * 0.8);
      } else {
        c = colors[toonBand(ndl, thresholds)];
        if (highlight && dot(N, H) > 0.995) c = [255, 255, 255];
        else if (highlight && dot(N, H) > 0.975) c = palette.hi;
        if (rim && ndl < threshold && N[2] < 0.35) c = palette.hi;
      }
      const i = (py * size + px) * 4;
      d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2];
      // Anti-alias the silhouette edge
      d[i + 3] = Math.min(255, Math.max(0, (1 - Math.sqrt(rr)) * r * 255 * 1.5));
    }
  }
  // putImageData ignores transforms, so render via a scratch canvas.
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = size;
  tmp.getContext('2d').putImageData(img, 0, 0);
  ctx.drawImage(tmp, cx - size / 2, cy - size / 2);
  if (outline > 0) {
    ctx.save();
    ctx.lineWidth = outline;
    ctx.strokeStyle = '#2a1b2a';
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

/**
 * Hand-painted style cel shadow on any 2D shape: fill the shape, then draw
 * the same shape offset AWAY from the light, clipped to the original, in the
 * shadow color. This is the fastest way to "fake" form in 2D.
 */
//   pathFn(ctx) must add the shape's sub-path WITHOUT calling beginPath().
export function celShadeShape(ctx, pathFn, { base, light = [-1, -1], depth = 10, outline = 2, lineColor = '#2a1b2a' }) {
  const len = Math.hypot(light[0], light[1]) || 1;
  const ox = (-light[0] / len) * depth, oy = (-light[1] / len) * depth;
  ctx.save();
  ctx.beginPath();
  pathFn(ctx);
  ctx.fillStyle = base;
  ctx.fill();
  ctx.clip();
  // Everything in the shape EXCEPT a copy shifted toward the light is shadow.
  ctx.fillStyle = shadowColor(base);
  ctx.beginPath();
  ctx.rect(-1e4, -1e4, 2e4, 2e4);
  ctx.translate(-ox, -oy);
  pathFn(ctx);
  ctx.fill('evenodd');
  ctx.restore();
  if (outline) {
    ctx.save();
    ctx.beginPath();
    pathFn(ctx);
    ctx.lineWidth = outline;
    ctx.strokeStyle = lineColor;
    ctx.stroke();
    ctx.restore();
  }
}
