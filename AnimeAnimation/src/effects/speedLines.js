// Anime motion effects: focus/speed lines, parallel speed lines, impact
// frames, and smears. These are cheap ways to sell motion and energy without
// extra in-betweens — key to the "limited animation" look.

import { mulberry32, randRange } from '../utils/random.js';

/**
 * Radial "focus lines" (shuuchuusen) pointing at a subject.
 * Re-seed every drawing (e.g. seed = drawing number) to make them "boil".
 */
export function drawFocusLines(ctx, { cx, cy, inner = 120, outer = 900, count = 90, seed = 1, color = '#111', thickness = 6 }) {
  const rand = mulberry32(seed);
  ctx.save();
  ctx.fillStyle = color;
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2 + randRange(rand, -0.03, 0.03);
    const start = inner * randRange(rand, 0.9, 1.6);
    const w = thickness * randRange(rand, 0.3, 1.2) * 0.005;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * start, cy + Math.sin(a) * start);
    ctx.lineTo(cx + Math.cos(a - w) * outer, cy + Math.sin(a - w) * outer);
    ctx.lineTo(cx + Math.cos(a + w) * outer, cy + Math.sin(a + w) * outer);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}

/** Horizontal speed lines behind a moving object (direction dx sign). */
export function drawParallelSpeedLines(ctx, { x, y, width, height, count = 30, seed = 1, color = 'rgba(40,40,60,0.7)', dir = -1 }) {
  const rand = mulberry32(seed);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineCap = 'round';
  for (let i = 0; i < count; i++) {
    const ly = y + rand() * height;
    const len = randRange(rand, 0.3, 1) * width;
    const lx = x + (dir < 0 ? 0 : width - len) + randRange(rand, -20, 20);
    ctx.lineWidth = randRange(rand, 1, 4);
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(lx + len, ly);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Impact frame: 1–3 frames of inverted / high-contrast image at the moment of
 * a hit. Call AFTER drawing the scene.
 */
export function applyImpactFrame(ctx, { mode = 'invert' } = {}) {
  const { width, height } = ctx.canvas;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (mode === 'invert') {
    ctx.globalCompositeOperation = 'difference';
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);
  } else {
    // "Red/black" style
    ctx.globalCompositeOperation = 'multiply';
    ctx.fillStyle = '#ff2a3a';
    ctx.fillRect(0, 0, width, height);
  }
  ctx.restore();
}

/**
 * Smear frame: stretch a round shape along its motion path so a fast move
 * reads in a single drawing. `prev` and `cur` are positions one frame apart.
 */
export function drawSmear(ctx, { prev, cur, radius, color = '#f28fb1', lineColor = '#2a1b2a', lines = 3 }) {
  const dx = cur.x - prev.x, dy = cur.y - prev.y;
  const dist = Math.hypot(dx, dy);
  const a = Math.atan2(dy, dx);
  ctx.save();
  ctx.translate(cur.x, cur.y);
  ctx.rotate(a);
  // Teardrop: full circle at the front, tapering tail behind.
  ctx.beginPath();
  ctx.arc(0, 0, radius, -Math.PI / 2, Math.PI / 2);
  ctx.quadraticCurveTo(-dist * 0.5, radius * 0.6, -dist, 0);
  ctx.quadraticCurveTo(-dist * 0.5, -radius * 0.6, 0, -radius);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 3;
  ctx.stroke();
  // Multiples / drybrush streaks in the tail
  ctx.lineWidth = 2;
  for (let i = 1; i <= lines; i++) {
    const off = (i / (lines + 1) - 0.5) * radius * 1.4;
    ctx.beginPath();
    ctx.moveTo(-radius * 0.2, off);
    ctx.lineTo(-dist * (0.5 + 0.4 * (1 - Math.abs(off) / radius)), off * 0.4);
    ctx.stroke();
  }
  ctx.restore();
}

/** Sparkles / "kira kira" for cute or dramatic moments. */
export function drawSparkle(ctx, x, y, size, color = '#fff6a8') {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = color;
  ctx.beginPath();
  for (let i = 0; i < 4; i++) {
    const a = (i * Math.PI) / 2;
    ctx.lineTo(Math.cos(a) * size, Math.sin(a) * size);
    ctx.lineTo(Math.cos(a + Math.PI / 4) * size * 0.2, Math.sin(a + Math.PI / 4) * size * 0.2);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}
