// Body proportions measured in "heads" — how many head-heights tall a
// character is. This single number does more to set the style than anything
// else: 2–3 heads reads as chibi/super-deformed, 5 as a child, 6.5–7 as a
// typical teen anime lead, 8+ as a tall adult or "heroic" action style.

import { lerp, clamp } from '../core/easing.js';

export const PROPORTION_PRESETS = {
  chibi: 2.5,
  child: 5,
  teen: 6.5,
  adult: 7.5,
  heroic: 8.5,
};

/**
 * Compute landmark heights (in head units from the top) for an H-heads figure.
 * Chibi figures shrink the legs most; tall figures get long legs.
 */
export function proportionLandmarks(H) {
  const k = clamp((H - 2) / 6.5); // 0 = chibi .. 1 = heroic
  const body = H - 1; // everything below the chin
  const legFrac = lerp(0.38, 0.58, k); // fraction of body that is legs
  const crotch = 1 + body * (1 - legFrac);
  const shoulder = 1 + lerp(0.08, 0.35, k);
  return {
    chin: 1,
    shoulder,
    chest: shoulder + (crotch - shoulder) * 0.3,
    waist: shoulder + (crotch - shoulder) * 0.62,
    crotch,
    knee: crotch + (H - crotch) * 0.5,
    fingertips: crotch + (H - crotch) * lerp(0.05, 0.3, k),
    floor: H,
    headWidth: lerp(1.15, 0.8, k), // in head units; chibi heads are wide
    shoulderWidth: lerp(1.0, 2.1, k),
    hipWidth: lerp(0.9, 1.45, k),
  };
}

/**
 * Draw a simple mannequin with a head-unit grid.
 * @param {CanvasRenderingContext2D} ctx
 * @param {{x:number, groundY:number, height:number, heads:number, color?:string,
 *          grid?:boolean, label?:string}} o
 */
export function drawProportionFigure(ctx, o) {
  const { x, groundY, height, heads: H, color = '#7c8cf8', grid = true, label } = o;
  const u = height / H; // pixels per head
  const L = proportionLandmarks(H);
  const top = groundY - height;
  const Y = (v) => top + v * u;

  ctx.save();
  if (grid) {
    ctx.strokeStyle = 'rgba(128,128,160,0.35)';
    ctx.fillStyle = 'rgba(128,128,160,0.9)';
    ctx.lineWidth = 1;
    ctx.font = '11px system-ui, sans-serif';
    for (let i = 0; i <= Math.ceil(H); i++) {
      const y = Y(Math.min(i, H));
      ctx.beginPath();
      const gw = Math.min(u * 1.4, 62);
      ctx.moveTo(x - gw, y);
      ctx.lineTo(x + gw, y);
      ctx.stroke();
      if (i < H) ctx.fillText(String(i + 1), x - gw, y + u / 2 + 4);
    }
  }

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.fillStyle = color;
  ctx.strokeStyle = color;

  // Legs
  const hipW = (L.hipWidth * u) / 2;
  ctx.lineWidth = u * lerp(0.42, 0.32, clamp((H - 2) / 6.5));
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(x + s * hipW * 0.5, Y(L.crotch) - u * 0.1);
    ctx.lineTo(x + s * hipW * 0.42, Y(L.knee));
    ctx.lineTo(x + s * hipW * 0.38, Y(L.floor) - ctx.lineWidth / 2);
    ctx.stroke();
  }
  // Torso
  const shW = (L.shoulderWidth * u) / 2;
  ctx.beginPath();
  ctx.moveTo(x - shW * 0.95, Y(L.shoulder));
  ctx.lineTo(x + shW * 0.95, Y(L.shoulder));
  ctx.quadraticCurveTo(x + shW * 0.7, Y(L.chest), x + hipW * 0.55, Y(L.waist));
  ctx.lineTo(x + hipW, Y(L.crotch));
  ctx.lineTo(x - hipW, Y(L.crotch));
  ctx.lineTo(x - hipW * 0.55, Y(L.waist));
  ctx.quadraticCurveTo(x - shW * 0.7, Y(L.chest), x - shW * 0.95, Y(L.shoulder));
  ctx.fill();
  // Neck
  ctx.lineWidth = u * 0.22;
  ctx.beginPath();
  ctx.moveTo(x, Y(0.9));
  ctx.lineTo(x, Y(L.shoulder));
  ctx.stroke();
  // Arms
  ctx.lineWidth = u * lerp(0.3, 0.22, clamp((H - 2) / 6.5));
  for (const s of [-1, 1]) {
    const elbowY = Y(L.waist);
    ctx.beginPath();
    ctx.moveTo(x + s * shW * 0.95, Y(L.shoulder) + ctx.lineWidth / 2);
    ctx.lineTo(x + s * shW * 1.1, elbowY);
    ctx.lineTo(x + s * shW * 1.15, Y(L.fingertips) - ctx.lineWidth / 2);
    ctx.stroke();
  }
  // Head (drawn last so it overlaps the neck)
  ctx.beginPath();
  ctx.ellipse(x, Y(0.5), (L.headWidth * u) / 2, u / 2, 0, 0, Math.PI * 2);
  ctx.fill();
  // Eye line — at the vertical middle of the head in anime style
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  ctx.lineWidth = Math.max(1, u * 0.03);
  ctx.beginPath();
  ctx.moveTo(x - (L.headWidth * u) / 2.4, Y(0.55));
  ctx.lineTo(x + (L.headWidth * u) / 2.4, Y(0.55));
  ctx.stroke();

  if (label) {
    ctx.fillStyle = getComputedStyleSafe(ctx);
    ctx.font = '600 13px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, groundY + 20);
  }
  ctx.restore();
}

function getComputedStyleSafe(ctx) {
  try {
    return getComputedStyle(ctx.canvas).color || '#333';
  } catch {
    return '#333';
  }
}
