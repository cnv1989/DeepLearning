import { test } from 'node:test';
import assert from 'node:assert/strict';

import { EASINGS, cubicBezier, lerp } from '../src/core/easing.js';
import { Track, Clip, stepTime, drawingCount, timingChart } from '../src/core/keyframes.js';
import { hexToRgb, rgbToHex, rgbToHsl, hslToRgb, shadowColor } from '../src/utils/color.js';
import { Spring, Strand } from '../src/physics/spring.js';
import { createHumanoid, Bone, Skeleton } from '../src/rig/skeleton.js';
import { walkPose, plantOnGround } from '../src/rig/walkCycle.js';
import { toonBand } from '../src/effects/celShading.js';
import { textToVisemes } from '../src/character/mouth.js';
import { proportionLandmarks } from '../src/character/proportions.js';
import { blinkOpenness } from '../src/character/eyes.js';
import { mulberry32 } from '../src/utils/random.js';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} ≈ ${b}`);

test('every easing starts at 0 and ends at 1', () => {
  for (const [name, fn] of Object.entries(EASINGS)) {
    close(fn(0), 0, 1e-9);
    close(fn(1), 1, 1e-9);
    assert.ok(Number.isFinite(fn(0.5)), name);
  }
});

test('ease-in is slow at the start, ease-out is fast at the start', () => {
  assert.ok(EASINGS.easeInQuad(0.25) < 0.25);
  assert.ok(EASINGS.easeOutQuad(0.25) > 0.25);
});

test('cubicBezier(0,0,1,1) is linear and (.42,0,.58,1) is symmetric', () => {
  const lin = cubicBezier(0, 0, 1, 1);
  for (const t of [0.1, 0.3, 0.7]) close(lin(t), t, 1e-4);
  const io = cubicBezier(0.42, 0, 0.58, 1);
  close(io(0.5), 0.5, 1e-4);
  close(io(0.2) + io(0.8), 1, 1e-4);
});

test('Track interpolates numbers, arrays and holds strings', () => {
  const tr = new Track([{ t: 0, v: 0 }, { t: 1, v: 10 }]);
  close(tr.valueAt(0.5), 5);
  close(tr.valueAt(-1), 0);
  close(tr.valueAt(5), 10);
  const arr = new Track([{ t: 0, v: [0, 100] }, { t: 2, v: [10, 0] }]);
  assert.deepEqual(arr.valueAt(1), [5, 50]);
  const str = new Track([{ t: 0, v: 'A' }, { t: 1, v: 'O' }]);
  assert.equal(str.valueAt(0.99), 'A');
  assert.equal(str.valueAt(1), 'O');
});

test('Track hold keeps the value until the next key', () => {
  const tr = new Track([{ t: 0, v: 0, hold: true }, { t: 1, v: 10 }]);
  assert.equal(tr.valueAt(0.9), 0);
  assert.equal(tr.valueAt(1), 10);
});

test('Track easing applies to the segment starting at a key', () => {
  const tr = new Track([{ t: 0, v: 0, ease: 'easeInQuad' }, { t: 1, v: 100 }]);
  close(tr.valueAt(0.5), 25);
});

test('Clip samples all tracks', () => {
  const c = new Clip({ x: new Track([{ t: 0, v: 0 }, { t: 2, v: 2 }]), y: new Track([{ t: 0, v: 5 }]) });
  assert.deepEqual(c.sample(1), { x: 1, y: 5 });
  assert.equal(c.duration, 2);
});

test('stepTime holds drawings on twos and threes', () => {
  // at 24 fps on twos, frames 0,1 show drawing 0; frames 2,3 show drawing 2
  assert.equal(stepTime(0 / 24, 24, 2), 0);
  assert.equal(stepTime(1 / 24, 24, 2), 0);
  close(stepTime(2 / 24, 24, 2), 2 / 24);
  close(stepTime(5 / 24, 24, 3), 3 / 24);
  close(stepTime(5 / 24, 24, 1), 5 / 24);
});

test('drawingCount: one second on twos is 12 drawings', () => {
  assert.equal(drawingCount(1, 24, 2), 12);
  assert.equal(drawingCount(1, 24, 3), 8);
  assert.equal(drawingCount(1, 24, 1), 24);
});

test('timingChart spaces marks according to easing', () => {
  const lin = timingChart(4);
  assert.deepEqual(lin, [0, 0.25, 0.5, 0.75, 1]);
  const ease = timingChart(4, 'easeInQuad');
  assert.ok(ease[1] - ease[0] < ease[4] - ease[3], 'marks bunch up at the start');
});

test('color conversions round-trip', () => {
  for (const hex of ['#ff0000', '#3d7fd6', '#ffe3d3', '#000000', '#ffffff']) {
    assert.equal(rgbToHex(hexToRgb(hex)), hex);
    assert.equal(rgbToHex(hslToRgb(rgbToHsl(hexToRgb(hex)))), hex);
  }
});

test('shadowColor is darker and shifts hue by a bounded amount', () => {
  for (const base of ['#f28fb1', '#ffd34d', '#2f4a8a', '#4f9a6a']) {
    const [h0, , l0] = rgbToHsl(hexToRgb(base));
    const [h1, , l1] = rgbToHsl(hexToRgb(shadowColor(base)));
    assert.ok(l1 < l0, `${base} shadow is darker`);
    const dh = Math.abs(((h1 - h0 + 540) % 360) - 180);
    assert.ok(dh <= 21, `${base} hue shift ${dh} is bounded`);
  }
});

test('Spring settles at its target', () => {
  const s = new Spring({ stiffness: 100, damping: 12 });
  for (let i = 0; i < 300; i++) s.step(10, 1 / 60);
  close(s.value, 10, 1e-3);
});

test('Spring with low damping overshoots (follow-through)', () => {
  const s = new Spring({ stiffness: 100, damping: 2 });
  let max = 0;
  for (let i = 0; i < 120; i++) max = Math.max(max, s.step(1, 1 / 60));
  assert.ok(max > 1.2);
});

test('Strand keeps segment lengths and stays attached to the root', () => {
  const st = new Strand({ x: 0, y: 0 }, { segments: 5, length: 50 });
  let pts;
  for (let i = 0; i < 120; i++) pts = st.step({ x: Math.sin(i / 10) * 40, y: 0 }, 1 / 60);
  assert.equal(pts[0].x, Math.sin(119 / 10) * 40);
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    close(d, 10, 1.0);
  }
});

test('forward kinematics: child follows parent rotation', () => {
  const root = new Bone('upper', 10, { angle: 0 });
  root.add(new Bone('lower', 10, { angle: 90 }));
  const sk = new Skeleton(root).solve();
  close(sk.get('lower').end.x, 10);
  close(sk.get('lower').end.y, 10);
  sk.setPose({ upper: 90 }).solve();
  close(sk.get('lower').end.x, -10);
  close(sk.get('lower').end.y, 10);
});

test('humanoid stands upright: head above hips above feet', () => {
  const h = createHumanoid(300).solve();
  const head = h.get('head').end.y, hips = h.get('hips').start.y, foot = h.get('footL').start.y;
  assert.ok(head < hips && hips < foot);
});

test('walk cycle: half a cycle later the legs have swapped', () => {
  const a = walkPose(0.1), b = walkPose(0.6);
  close(a.thighL, b.thighR);
  close(a.shinL, b.shinR);
  close(a.upperArmL, b.upperArmR);
});

test('plantOnGround puts the lowest foot on the ground', () => {
  const h = createHumanoid(300);
  h.setPose(walkPose(0.3));
  plantOnGround(h, 400);
  let maxY = -Infinity;
  h.forEach((b) => { if (/foot|shin/.test(b.name)) maxY = Math.max(maxY, b.start.y, b.end.y); });
  close(maxY, 400, 1e-6);
});

test('toonBand quantizes lighting', () => {
  assert.equal(toonBand(-0.5, [0]), 0);
  assert.equal(toonBand(0.5, [0]), 1);
  assert.equal(toonBand(0.1, [-0.3, 0.25]), 1);
  assert.equal(toonBand(0.9, [-0.3, 0.25]), 2);
});

test('textToVisemes maps vowels and closes on m/b/p', () => {
  const keys = textToVisemes('mama');
  assert.deepEqual(keys.map((k) => k.v), ['closed', 'A', 'closed', 'A', 'closed']);
  assert.ok(keys.every((k, i) => i === 0 || k.t >= keys[i - 1].t));
  const flap = textToVisemes('aiueo', { mode: 'flap' });
  assert.ok(flap.every((k) => ['closed', 'half', 'A'].includes(k.v)));
});

test('proportions: taller characters have proportionally longer legs', () => {
  const chibi = proportionLandmarks(2.5), hero = proportionLandmarks(8.5);
  const legs = (L) => (L.floor - L.crotch) / L.floor;
  assert.ok(legs(hero) > legs(chibi));
  for (const L of [chibi, hero]) {
    assert.ok(L.chin < L.shoulder && L.shoulder < L.waist && L.waist < L.crotch && L.crotch < L.knee && L.knee < L.floor);
  }
});

test('blink is short and fully closes', () => {
  const values = Array.from({ length: 24 * 3 }, (_, f) => blinkOpenness(f / 24)); // one blink interval (3.2s)
  assert.ok(values.includes(0));
  assert.ok(values.filter((v) => v < 1).length <= 6);
});

test('seeded random is repeatable', () => {
  const a = mulberry32(42), b = mulberry32(42);
  for (let i = 0; i < 5; i++) assert.equal(a(), b());
  close(lerp(0, 10, 0.3), 3);
});
