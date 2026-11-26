// Front-view anime head built from simple construction shapes:
//   1. a circle for the cranium
//   2. a jaw that tapers to a small chin
//   3. an eye line LOW on the head (about halfway between crown and chin)
//   4. tiny nose, small mouth, big eyes, then hair on top
// Every proportion is a parameter so you can explore styles with sliders.

import { drawAnimeEye } from './eyes.js';
import { drawMouth } from './mouth.js';
import { shadowColor, highlightColor, SKIN_SHADOW } from '../utils/color.js';
import { lerp } from '../core/easing.js';

export const DEFAULT_FACE = {
  skin: '#ffe3d3',
  hairColor: '#f28fb1',
  eyeColor: '#3d7fd6',
  lineColor: '#4a2b2b',
  eyeStyle: 'shoujo',
  eyeSize: 1,        // multiplier
  eyeSpacing: 0.42,  // fraction of cranium radius from center
  eyeLine: 0.3,      // eye height below cranium center (fraction of R)
  chin: 0.6,         // 0 round .. 1 pointy
  jawWidth: 0.85,    // 0.6 narrow .. 1 wide
  expression: 'neutral',
  openness: 1,
  mouth: null,       // overrides expression mouth
  look: { x: 0, y: 0 },
  blush: 0.4,
  hairStyle: 'bob',
  ahoge: true,       // the single "antenna" strand
  tilt: 0,           // head tilt in degrees
};

export const EXPRESSIONS = {
  neutral: { brows: [0, 0], eyeShape: 'normal', open: 1, mouth: 'closed' },
  happy: { brows: [-4, -2], eyeShape: 'happy', open: 1, mouth: 'A', blush: 0.7 },
  smile: { brows: [-2, -1], eyeShape: 'normal', open: 0.85, mouth: 'smile' },
  angry: { brows: [10, -6], eyeShape: 'sharp', open: 0.8, mouth: 'shout', symbol: 'anger' },
  sad: { brows: [-9, 4], eyeShape: 'normal', open: 0.7, mouth: 'frown', look: { x: 0, y: 0.6 }, symbol: 'tear' },
  surprised: { brows: [-12, -12], eyeShape: 'normal', open: 1, mouth: 'O' },
  smug: { brows: [-3, 5], eyeShape: 'normal', open: 0.55, mouth: 'cat', look: { x: 0.5, y: 0 } },
  nervous: { brows: [-7, 3], eyeShape: 'normal', open: 0.9, mouth: 'I', symbol: 'sweat' },
};

function faceOutline(ctx, R, p) {
  const chinW = R * lerp(0.4, 0.04, p.chin);
  const chinY = R * lerp(1.2, 1.38, p.chin);
  const jw = R * p.jawWidth;
  ctx.beginPath();
  ctx.moveTo(-R, 0);
  ctx.arc(0, 0, R, Math.PI, 0);
  ctx.bezierCurveTo(R, R * 0.45, jw, R * 0.8, chinW, chinY - R * 0.05);
  ctx.quadraticCurveTo(0, chinY + R * 0.03, -chinW, chinY - R * 0.05);
  ctx.bezierCurveTo(-jw, R * 0.8, -R, R * 0.45, -R, 0);
  ctx.closePath();
  return chinY;
}

function drawBackHair(ctx, R, p) {
  const c = p.hairColor;
  ctx.fillStyle = shadowColor(c, { darken: 0.12, saturate: -0.15 });
  ctx.strokeStyle = p.lineColor;
  ctx.lineWidth = R * 0.025;
  ctx.beginPath();
  if (p.hairStyle === 'long') {
    ctx.moveTo(-R * 1.1, -R * 0.1);
    // Ends just above the bust's bottom edge so it tucks behind the shoulders.
    ctx.bezierCurveTo(-R * 1.3, R * 1.0, -R * 1.25, R * 1.8, -R * 1.15, R * 2.25);
    ctx.lineTo(R * 1.15, R * 2.25);
    ctx.bezierCurveTo(R * 1.25, R * 1.8, R * 1.3, R * 1.0, R * 1.1, -R * 0.1);
  } else if (p.hairStyle === 'spiky') {
    ctx.moveTo(-R * 1.05, 0);
    const spikes = [[-1.35, 0.45], [-1.1, 0.5], [-1.25, 1.0], [-0.85, 0.75]];
    for (const [sx, sy] of spikes) ctx.lineTo(R * sx, R * sy);
    for (const [sx, sy] of [...spikes].reverse()) ctx.lineTo(-R * sx, R * sy);
    ctx.lineTo(R * 1.05, 0);
  } else {
    // bob
    ctx.moveTo(-R * 1.08, -R * 0.1);
    ctx.bezierCurveTo(-R * 1.25, R * 0.6, -R * 1.15, R * 1.2, -R * 0.95, R * 1.35);
    ctx.lineTo(R * 0.95, R * 1.35);
    ctx.bezierCurveTo(R * 1.15, R * 1.2, R * 1.25, R * 0.6, R * 1.08, -R * 0.1);
  }
  ctx.arc(0, -R * 0.05, R * 1.1, 0, Math.PI, true);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawFrontHair(ctx, R, p) {
  const c = p.hairColor;
  const shadow = shadowColor(c);
  const hi = highlightColor(c, { lighten: 0.22 });
  // Cranium cap + bangs as one polygon with spiky locks along the bottom.
  // [x, y] in units of R: alternating lock tips (low) and gaps (high).
  // Tips must stay ABOVE the eye line or the bangs hide the eyes.
  const locks = p.hairStyle === 'spiky'
    ? [[-1.05, 0.15], [-0.85, -0.25], [-0.62, 0.12], [-0.4, -0.32], [-0.18, 0.08], [0.05, -0.35], [0.25, 0.1], [0.48, -0.3], [0.7, 0.12], [0.88, -0.2], [1.05, 0.15]]
    : [[-1.06, 0.45], [-0.88, -0.1], [-0.7, 0.1], [-0.48, -0.3], [-0.3, 0.06], [-0.08, -0.32], [0.1, 0.04], [0.32, -0.3], [0.52, 0.08], [0.76, -0.15], [0.92, 0.3], [1.06, 0.45]];
  const path = () => {
    ctx.beginPath();
    ctx.moveTo(R * locks[0][0], R * locks[0][1]);
    for (let i = 1; i < locks.length; i++) {
      const [x0, y0] = locks[i - 1], [x1, y1] = locks[i];
      // Curved locks: control point bows outward so strands look like clumps.
      ctx.quadraticCurveTo(R * (x0 + x1) / 2, R * Math.min(y0, y1) + R * 0.12, R * x1, R * y1);
    }
    ctx.bezierCurveTo(R * 1.2, -R * 0.4, R * 0.8, -R * 1.25, 0, -R * 1.18);
    ctx.bezierCurveTo(-R * 0.8, -R * 1.25, -R * 1.2, -R * 0.4, R * locks[0][0], R * locks[0][1]);
    ctx.closePath();
  };
  path();
  ctx.fillStyle = c;
  ctx.fill();
  // Cel shadow band along the bottom of the bangs (hair casts shadow downward).
  ctx.save();
  path();
  ctx.clip();
  ctx.fillStyle = shadow;
  ctx.beginPath();
  ctx.ellipse(0, R * 0.32, R * 1.4, R * 0.36, 0, 0, Math.PI * 2);
  ctx.fill();
  // The "angel ring" highlight — a jagged band around the crown.
  ctx.fillStyle = hi;
  ctx.beginPath();
  const ringY = -R * 0.62;
  ctx.moveTo(-R * 0.85, ringY + R * 0.2);
  for (let i = 0; i <= 10; i++) {
    const x = lerp(-0.85, 0.85, i / 10) * R;
    const arc = -Math.cos((i / 10 - 0.5) * Math.PI) * R * 0.18;
    ctx.lineTo(x, ringY + arc + (i % 2 ? R * 0.14 : 0));
  }
  for (let i = 10; i >= 0; i--) {
    const x = lerp(-0.85, 0.85, i / 10) * R;
    const arc = -Math.cos((i / 10 - 0.5) * Math.PI) * R * 0.18;
    ctx.lineTo(x, ringY + arc - R * 0.06 - (i % 2 ? 0 : R * 0.05));
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  path();
  ctx.strokeStyle = p.lineColor;
  ctx.lineWidth = R * 0.025;
  ctx.stroke();
}

/** Draw the ahoge (antenna strand). `sway` in radians lets physics drive it. */
export function drawAhoge(ctx, R, color, lineColor, sway = 0) {
  ctx.save();
  ctx.translate(0, -R * 1.15);
  ctx.rotate(sway);
  ctx.beginPath();
  ctx.moveTo(-R * 0.05, 0);
  ctx.bezierCurveTo(-R * 0.1, -R * 0.4, R * 0.35, -R * 0.45, R * 0.3, -R * 0.2);
  ctx.bezierCurveTo(R * 0.25, -R * 0.35, R * 0.02, -R * 0.3, R * 0.06, 0);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = R * 0.02;
  ctx.stroke();
  ctx.restore();
}

function drawBrow(ctx, x, y, w, innerDeg, outerDeg, mirror, color, R) {
  ctx.save();
  ctx.translate(x, y);
  if (mirror) ctx.scale(-1, 1);
  ctx.strokeStyle = color;
  ctx.lineWidth = R * 0.035;
  ctx.lineCap = 'round';
  // inner end is at +x (toward the nose) for the screen-left eye
  const innerY = innerDeg * R * 0.012, outerY = outerDeg * R * 0.012;
  ctx.beginPath();
  ctx.moveTo(-w / 2, outerY);
  ctx.quadraticCurveTo(0, Math.min(innerY, outerY) - R * 0.06, w / 2, innerY);
  ctx.stroke();
  ctx.restore();
}

export function drawSymbol(ctx, kind, R) {
  ctx.save();
  if (kind === 'anger') {
    // The cross-shaped "popping vein" mark.
    ctx.translate(R * 0.65, -R * 0.75);
    ctx.strokeStyle = '#e0283c';
    ctx.lineWidth = R * 0.06;
    ctx.lineCap = 'round';
    for (let i = 0; i < 4; i++) {
      ctx.rotate(Math.PI / 2);
      ctx.beginPath();
      ctx.moveTo(R * 0.05, R * 0.18);
      ctx.quadraticCurveTo(R * 0.06, R * 0.06, R * 0.18, R * 0.05);
      ctx.stroke();
    }
  } else if (kind === 'sweat') {
    ctx.translate(R * 0.95, -R * 0.35);
    ctx.fillStyle = '#bfe6ff';
    ctx.strokeStyle = '#3a8bc9';
    ctx.lineWidth = R * 0.02;
    ctx.beginPath();
    ctx.moveTo(0, -R * 0.2);
    ctx.bezierCurveTo(R * 0.12, 0, R * 0.12, R * 0.12, 0, R * 0.12);
    ctx.bezierCurveTo(-R * 0.12, R * 0.12, -R * 0.12, 0, 0, -R * 0.2);
    ctx.fill();
    ctx.stroke();
  } else if (kind === 'tear') {
    ctx.fillStyle = 'rgba(150, 210, 255, 0.85)';
    for (const sx of [-1, 1]) {
      ctx.beginPath();
      ctx.ellipse(sx * R * 0.3, R * 0.62, R * 0.04, R * 0.1, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

function drawBust(ctx, R, p) {
  // Simple school-uniform shoulders: white shirt + sailor collar + ribbon.
  const shirt = '#f6f6fb', collar = '#2f4a8a';
  ctx.save();
  ctx.lineWidth = R * 0.025;
  ctx.strokeStyle = p.lineColor;
  ctx.fillStyle = shirt;
  ctx.beginPath();
  ctx.moveTo(-R * 0.28, R * 1.55);
  ctx.bezierCurveTo(-R * 0.9, R * 1.6, -R * 1.25, R * 1.75, -R * 1.35, R * 2.3);
  ctx.lineTo(R * 1.35, R * 2.3);
  ctx.bezierCurveTo(R * 1.25, R * 1.75, R * 0.9, R * 1.6, R * 0.28, R * 1.55);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = collar;
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(s * R * 0.28, R * 1.55);
    ctx.lineTo(s * R * 0.95, R * 1.68);
    ctx.lineTo(s * R * 0.7, R * 2.0);
    ctx.lineTo(0, R * 2.15);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.fillStyle = '#e0457b';
  ctx.beginPath();
  ctx.moveTo(0, R * 2.05);
  ctx.lineTo(-R * 0.25, R * 1.92); ctx.lineTo(-R * 0.22, R * 2.2);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(0, R * 2.05);
  ctx.lineTo(R * 0.25, R * 1.92); ctx.lineTo(R * 0.22, R * 2.2);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.restore();
}

/**
 * Draw an anime head. (x, y) is the cranium center; R = 100 * scale.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Partial<typeof DEFAULT_FACE> & {x:number, y:number, scale?:number, ahogeSway?:number}} params
 */
export function drawAnimeHead(ctx, params) {
  const p = { ...DEFAULT_FACE, ...params };
  const R = 100 * (p.scale ?? 1);
  const ex = EXPRESSIONS[p.expression] || EXPRESSIONS.neutral;
  const skinShadow = shadowColor(p.skin, SKIN_SHADOW);

  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.rotate((p.tilt * Math.PI) / 180);
  ctx.lineJoin = 'round';

  drawBackHair(ctx, R, p);

  // Neck with a hard cel shadow under the jaw.
  ctx.fillStyle = p.skin;
  ctx.strokeStyle = p.lineColor;
  ctx.lineWidth = R * 0.025;
  ctx.beginPath();
  ctx.rect(-R * 0.28, R * 0.7, R * 0.56, R * 0.9);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = skinShadow;
  ctx.beginPath();
  ctx.moveTo(-R * 0.28, R * 0.95);
  ctx.quadraticCurveTo(0, R * 1.6, R * 0.28, R * 0.95);
  ctx.lineTo(R * 0.28, R * 0.7);
  ctx.lineTo(-R * 0.28, R * 0.7);
  ctx.fill();

  if (p.bust !== false) drawBust(ctx, R, p);

  // Ears
  for (const s of [-1, 1]) {
    ctx.fillStyle = p.skin;
    ctx.beginPath();
    ctx.ellipse(s * R * 0.97, R * 0.3, R * 0.12, R * 0.22, s * 0.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // Face
  faceOutline(ctx, R, p);
  ctx.fillStyle = p.skin;
  ctx.fill();
  ctx.stroke();

  // Blush: soft ellipses + three hatch lines (a very anime convention).
  const blush = ex.blush ?? p.blush;
  if (blush > 0) {
    for (const s of [-1, 1]) {
      const bx = s * R * 0.5, by = R * 0.62;
      const g = ctx.createRadialGradient(bx, by, 0, bx, by, R * 0.22);
      g.addColorStop(0, `rgba(255, 120, 150, ${0.55 * blush})`);
      g.addColorStop(1, 'rgba(255, 120, 150, 0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(bx, by, R * 0.24, R * 0.13, 0, 0, Math.PI * 2);
      ctx.fill();
      if (blush > 0.5) {
        ctx.strokeStyle = `rgba(230, 90, 120, ${blush})`;
        ctx.lineWidth = R * 0.015;
        for (let i = -1; i <= 1; i++) {
          ctx.beginPath();
          ctx.moveTo(bx + i * R * 0.07 + R * 0.03, by - R * 0.04);
          ctx.lineTo(bx + i * R * 0.07 - R * 0.02, by + R * 0.04);
          ctx.stroke();
        }
      }
    }
  }

  // Eyes
  const eyeY = R * p.eyeLine;
  const eyeW = R * 0.5 * p.eyeSize;
  const open = Math.min(p.openness, ex.open);
  const look = params.look ?? ex.look ?? p.look;
  for (const s of [-1, 1]) {
    drawAnimeEye(ctx, {
      x: s * R * p.eyeSpacing, y: eyeY, size: eyeW, openness: open, look: { x: s === -1 ? -look.x : look.x, y: look.y },
      irisColor: p.eyeColor, mirror: s === -1, style: p.eyeStyle, shape: ex.eyeShape, lashColor: p.lineColor,
    });
  }

  // Nose: a tiny shadow tick.
  ctx.strokeStyle = shadowColor(p.skin, { ...SKIN_SHADOW, darken: 0.25 });
  ctx.lineWidth = R * 0.02;
  ctx.beginPath();
  ctx.moveTo(R * 0.02, R * 0.72);
  ctx.lineTo(-R * 0.02, R * 0.78);
  ctx.stroke();

  drawMouth(ctx, { x: 0, y: R * 0.98, width: R * 0.5, shape: p.mouth || ex.mouth, lineColor: p.lineColor });

  drawFrontHair(ctx, R, p);
  // Brows are drawn OVER the bangs — a common anime convention so the
  // eyebrows can always act, even under heavy fringe.
  for (const s of [-1, 1]) {
    drawBrow(ctx, s * R * p.eyeSpacing, eyeY - eyeW * 0.85, eyeW * 0.8, ex.brows[0], ex.brows[1], s === 1, shadowColor(p.hairColor, { darken: 0.3 }), R);
  }
  if (p.ahoge) drawAhoge(ctx, R, p.hairColor, p.lineColor, p.ahogeSway ?? 0);
  if (ex.symbol) drawSymbol(ctx, ex.symbol, R);

  ctx.restore();
}
