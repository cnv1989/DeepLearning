// Keyframes and in-betweens.
//
// A Track holds key poses ("genga" in a Japanese studio) at specific times.
// valueAt(t) computes the in-between ("douga") for any time, using the easing
// attached to the key the segment starts from.

import { getEasing, lerp } from './easing.js';

function interpolate(a, b, p) {
  if (typeof a === 'number') return lerp(a, b, p);
  if (Array.isArray(a)) return a.map((v, i) => interpolate(v, b[i], p));
  if (a && typeof a === 'object') {
    const out = {};
    for (const k of Object.keys(a)) out[k] = k in b ? interpolate(a[k], b[k], p) : a[k];
    return out;
  }
  // Non-numeric values (strings like mouth shapes) hold until the next key.
  return p < 1 ? a : b;
}

export class Track {
  /**
   * @param {Array<{t:number, v:any, ease?:string|Function, hold?:boolean}>} keys
   *   t: time in seconds, v: value (number | array | object | string)
   *   ease: easing for the segment that STARTS at this key
   *   hold: if true, value is held (stepped) until the next key
   */
  constructor(keys = []) {
    this.keys = [...keys].sort((a, b) => a.t - b.t);
  }

  add(t, v, opts = {}) {
    this.keys.push({ t, v, ...opts });
    this.keys.sort((a, b) => a.t - b.t);
    return this;
  }

  get duration() {
    return this.keys.length ? this.keys[this.keys.length - 1].t : 0;
  }

  valueAt(t) {
    const k = this.keys;
    if (!k.length) return undefined;
    if (t <= k[0].t) return k[0].v;
    if (t >= k[k.length - 1].t) return k[k.length - 1].v;
    let i = 0;
    while (i < k.length - 1 && k[i + 1].t <= t) i++;
    const a = k[i], b = k[i + 1];
    if (a.hold) return a.v;
    const p = getEasing(a.ease)((t - a.t) / (b.t - a.t));
    return interpolate(a.v, b.v, p);
  }
}

// A named collection of tracks: e.g. { x: Track, y: Track, squash: Track }.
export class Clip {
  constructor(tracks = {}) {
    this.tracks = tracks;
  }
  get duration() {
    return Math.max(0, ...Object.values(this.tracks).map((tr) => tr.duration));
  }
  sample(t) {
    const out = {};
    for (const [name, tr] of Object.entries(this.tracks)) out[name] = tr.valueAt(t);
    return out;
  }
}

/**
 * Quantize time to "animating on N's".
 * Anime is typically timed at 24 fps but most drawings are held for 2 frames
 * ("on twos", 12 drawings/sec) or 3 frames ("on threes", 8 drawings/sec).
 * Returns the time of the drawing that should be on screen at time t.
 */
export function stepTime(t, fps = 24, onN = 1) {
  const frame = Math.floor(t * fps + 1e-9);
  const held = Math.floor(frame / onN) * onN;
  return held / fps;
}

/** Number of unique drawings needed for a shot. */
export function drawingCount(durationSec, fps = 24, onN = 2) {
  return Math.ceil((durationSec * fps) / onN);
}

/**
 * Timing chart: where do the in-betweens fall between two keys?
 * Returns progress values (0..1) for each in-between frame.
 * This is the little ruler animators draw next to key poses.
 */
export function timingChart(frames, ease = 'linear') {
  const e = getEasing(ease);
  const out = [];
  for (let i = 0; i <= frames; i++) out.push(e(i / frames));
  return out;
}
