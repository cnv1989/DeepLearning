// Color helpers. The big anime-coloring lesson lives here: shadows are NOT
// just "base color but darker". Studios pick a shadow color that shifts hue
// (usually toward blue/purple, or toward red for skin) and keeps saturation.

export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const v = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  const n = parseInt(v, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rgbToHex([r, g, b]) {
  const c = (x) => Math.round(Math.max(0, Math.min(255, x))).toString(16).padStart(2, '0');
  return `#${c(r)}${c(g)}${c(b)}`;
}

export function rgbToHsl([r, g, b]) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return [h * 60, s, l];
}

export function hslToRgb([h, s, l]) {
  h = ((h % 360) + 360) % 360 / 360;
  if (s === 0) return [l * 255, l * 255, l * 255];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const f = (t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [f(h + 1 / 3) * 255, f(h) * 255, f(h - 1 / 3) * 255];
}

/**
 * Pick an anime-style shadow color for a base color.
 * @param {string} base hex
 * @param {{darken?:number, hueShift?:number, saturate?:number, toward?:number}} o
 *   toward: target hue the shadow drifts toward (default 260 = blue-violet)
 *   hueShift: 0..1 how far to drift toward that hue
 */
export function shadowColor(base, { darken = 0.18, hueShift = 0.25, saturate = 0.0, toward = 260, maxShift = 20 } = {}) {
  const [h, s, l] = rgbToHsl(hexToRgb(base));
  return rgbToHex(hslToRgb([h + hueStep(h, toward, hueShift, maxShift), Math.min(1, s + saturate), Math.max(0, l - darken)]));
}

/** Move hue h toward `toward` by fraction `amount`, capped at maxShift degrees. */
export function hueStep(h, toward, amount, maxShift = 20) {
  const dh = ((toward - h + 540) % 360) - 180; // shortest way around the wheel
  const step = dh * amount;
  return Math.sign(step) * Math.min(Math.abs(step), maxShift * (amount > 0 ? 1 : 0));
}

export function highlightColor(base, { lighten = 0.15, hueShift = 0.1, toward = 55 } = {}) {
  const [h, s, l] = rgbToHsl(hexToRgb(base));
  return rgbToHex(hslToRgb([h + hueStep(h, toward, hueShift, 10), s, Math.min(1, l + lighten)]));
}

export const mixHex = (a, b, t) => {
  const A = hexToRgb(a), B = hexToRgb(b);
  return rgbToHex(A.map((v, i) => v + (B[i] - v) * t));
};

/** A small ready-made palette set: base, shadow and highlight per material. */
export function makeMaterial(base, opts) {
  return { base, shadow: shadowColor(base, opts), highlight: highlightColor(base) };
}

export const SKIN_SHADOW = { toward: 350, hueShift: 0.35, darken: 0.1, saturate: 0.1 };
