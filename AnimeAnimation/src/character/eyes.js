// The anime eye — the single most important feature of the style.
// Anatomy (outside in): thick upper lash line, small lower lash line,
// white sclera, a TALL iris (taller than wide), dark gradient at the top of
// the iris (shadow cast by the lid), pupil, and 1–3 hard white highlights.

import { lerp } from '../core/easing.js';
import { shadowColor, highlightColor } from '../utils/color.js';

export const EYE_STYLES = {
  // h: height/width ratio; iris: iris width fraction; lash: lash thickness
  shoujo: { h: 1.05, iris: 0.62, lash: 0.11, lowerLash: true, highlights: 3 },
  shounen: { h: 0.62, iris: 0.5, lash: 0.09, lowerLash: false, highlights: 1 },
  chibi: { h: 1.2, iris: 0.72, lash: 0.1, lowerLash: false, highlights: 2 },
  mature: { h: 0.55, iris: 0.42, lash: 0.07, lowerLash: true, highlights: 1 },
};

function bez(ctx, pts) {
  ctx.bezierCurveTo(pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1]);
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} o
 * @param {number} o.x center x
 * @param {number} o.y center y
 * @param {number} o.size eye width in px
 * @param {number} [o.openness=1] 0 closed .. 1 open (animate this for blinks)
 * @param {{x:number,y:number}} [o.look] -1..1 gaze direction
 * @param {string} [o.irisColor]
 * @param {boolean} [o.mirror] true for the character's left eye (screen right)
 * @param {string} [o.style] key of EYE_STYLES
 * @param {'normal'|'happy'|'sharp'} [o.shape]
 */
export function drawAnimeEye(ctx, o) {
  const {
    x, y, size, openness = 1, look = { x: 0, y: 0 }, irisColor = '#3d7fd6',
    mirror = false, style = 'shoujo', shape = 'normal', lashColor = '#2a1b1f',
  } = o;
  const S = EYE_STYLES[style] || EYE_STYLES.shoujo;
  const w = size, h = size * S.h;

  ctx.save();
  ctx.translate(x, y);
  if (mirror) ctx.scale(-1, 1);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = lashColor;
  ctx.fillStyle = lashColor;

  if (shape === 'happy') {
    // ^ ^ — closed, upward arc
    ctx.lineWidth = w * S.lash;
    ctx.beginPath();
    ctx.moveTo(-w * 0.45, h * 0.2);
    ctx.quadraticCurveTo(0, -h * 0.45, w * 0.45, h * 0.2);
    ctx.stroke();
    ctx.restore();
    return;
  }

  const sharp = shape === 'sharp' ? 1 : 0;
  // Lower lid (fixed)
  const lower = [
    [-w * 0.46, h * 0.18], [-w * 0.2, h * 0.52], [w * 0.22, h * 0.5], [w * 0.48, h * 0.12],
  ];
  // Upper lid (open pose); "sharp" flattens the top for an intense look.
  const upperOpen = [
    [-w * 0.5, h * 0.12],
    [-w * 0.3, -h * lerp(0.62, 0.3, sharp)],
    [w * 0.22, -h * lerp(0.72, 0.5, sharp)],
    [w * 0.52, -h * lerp(0.08, 0.2, sharp)],
  ];
  // Closing: control points drop toward a line just above the lower lid.
  const closeY = h * 0.3;
  const o2 = Math.max(0, Math.min(1, openness));
  const upper = upperOpen.map(([px, py], i) => [px, i === 0 || i === 3 ? lerp(closeY, py, o2) : lerp(closeY + h * 0.05, py, o2)]);

  if (o2 < 0.08) {
    // Closed eye: a single down-curved lash line.
    ctx.lineWidth = w * S.lash * 0.8;
    ctx.beginPath();
    ctx.moveTo(-w * 0.48, h * 0.15);
    ctx.quadraticCurveTo(0, h * 0.45, w * 0.5, h * 0.12);
    ctx.stroke();
    ctx.restore();
    return;
  }

  // --- Eye opening (clip region) ---
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(...upper[0]);
  bez(ctx, upper);
  ctx.lineTo(...lower[3]);
  bez(ctx, [lower[3], lower[2], lower[1], lower[0]]);
  ctx.closePath();
  ctx.fillStyle = '#fbfbff';
  ctx.fill();
  ctx.clip();

  // Iris
  const iw = w * S.iris, ih = h * 0.95;
  const ix = look.x * w * 0.2, iy = h * 0.08 + look.y * h * 0.12;
  const g = ctx.createLinearGradient(0, iy - ih / 2, 0, iy + ih / 2);
  g.addColorStop(0, shadowColor(irisColor, { darken: 0.35 }));
  g.addColorStop(0.45, irisColor);
  g.addColorStop(1, highlightColor(irisColor, { lighten: 0.25 }));
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.ellipse(ix, iy, iw / 2, ih / 2, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = w * 0.025;
  ctx.strokeStyle = shadowColor(irisColor, { darken: 0.4 });
  ctx.stroke();
  // Pupil
  ctx.fillStyle = shadowColor(irisColor, { darken: 0.5 });
  ctx.beginPath();
  ctx.ellipse(ix, iy + ih * 0.04, iw * 0.2, ih * 0.24, 0, 0, Math.PI * 2);
  ctx.fill();
  // Lid shadow on the eyeball (cel-shaded band)
  ctx.fillStyle = 'rgba(70, 60, 120, 0.28)';
  ctx.beginPath();
  ctx.moveTo(...upper[0]);
  bez(ctx, upper);
  ctx.lineTo(w, -h);
  ctx.lineTo(-w, -h);
  ctx.closePath();
  ctx.translate(0, h * 0.16);
  ctx.fill();
  ctx.translate(0, -h * 0.16);
  // Highlights: always on the same side for both eyes (light comes from one
  // direction!), so we undo the mirror for them.
  if (mirror) ctx.scale(-1, 1);
  const hx = mirror ? -ix : ix;
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.ellipse(hx - iw * 0.18, iy - ih * 0.2, iw * 0.16, ih * 0.13, -0.4, 0, Math.PI * 2);
  ctx.fill();
  if (S.highlights > 1) {
    ctx.beginPath();
    ctx.arc(hx + iw * 0.2, iy + ih * 0.2, iw * 0.07, 0, Math.PI * 2);
    ctx.fill();
  }
  if (S.highlights > 2) {
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.ellipse(hx, iy + ih * 0.32, iw * 0.22, ih * 0.05, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.restore();

  // --- Upper lash line: thick, tapering, with a flick at the outer corner ---
  ctx.fillStyle = lashColor;
  ctx.beginPath();
  ctx.moveTo(...upper[0]);
  bez(ctx, upper);
  const t = w * S.lash;
  ctx.lineTo(upper[3][0] + t * 0.9, upper[3][1] - t * 0.9); // flick
  ctx.bezierCurveTo(upper[2][0], upper[2][1] - t * 1.2, upper[1][0], upper[1][1] - t * 0.9, upper[0][0] + t * 0.2, upper[0][1] - t * 0.2);
  ctx.closePath();
  ctx.fill();

  if (S.lowerLash) {
    ctx.lineWidth = w * 0.03;
    ctx.beginPath();
    ctx.moveTo(lower[1][0] + w * 0.1, lower[1][1] + h * 0.02);
    ctx.quadraticCurveTo(w * 0.3, lower[2][1] - h * 0.02, lower[3][0], lower[3][1] + h * 0.03);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Blink curve for a given time, in seconds. Anime blinks are FAST and drawn
 * with very few frames: open → half → closed → half → open, timed on ones.
 * Returns eye openness 0..1.
 */
export function blinkOpenness(t, { interval = 3.2, frames = [1, 0.5, 0, 0, 0.4, 1], fps = 24 } = {}) {
  const local = t % interval;
  const f = Math.floor(local * fps);
  return f < frames.length ? frames[f] : 1;
}
