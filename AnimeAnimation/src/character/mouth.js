// Mouth shapes and lip flap.
//
// TV anime rarely does full lip-sync. It uses "kuchipaku" (lip flap):
// typically just 3 drawings — closed, half open, open — cycled roughly on
// threes while the character talks. Higher-budget work uses vowel shapes
// A / I / U / E / O, the five Japanese vowels.

export const MOUTH_SHAPES = ['closed', 'smile', 'frown', 'cat', 'A', 'I', 'U', 'E', 'O', 'half', 'shout'];
export const LIP_FLAP = ['closed', 'half', 'A'];

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {{x:number, y:number, width:number, shape?:string, lineColor?:string, innerColor?:string}} o
 */
export function drawMouth(ctx, o) {
  const { x, y, width: w, shape = 'closed', lineColor = '#5a2a2a', innerColor = '#8a2c3a' } = o;
  ctx.save();
  ctx.translate(x, y);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = Math.max(1.5, w * 0.07);

  const openShape = (top, bottom, halfW, flatTop = false) => {
    const outline = () => {
      ctx.beginPath();
      ctx.moveTo(-halfW, top);
      if (flatTop) ctx.quadraticCurveTo(0, top - w * 0.05, halfW, top);
      else ctx.quadraticCurveTo(0, top - (bottom - top) * 0.2, halfW, top);
      ctx.quadraticCurveTo(halfW * 0.9, bottom, 0, bottom);
      ctx.quadraticCurveTo(-halfW * 0.9, bottom, -halfW, top);
      ctx.closePath();
    };
    outline();
    ctx.fillStyle = innerColor;
    ctx.fill();
    // tongue, clipped to the mouth opening
    ctx.save();
    ctx.clip();
    ctx.fillStyle = '#e07383';
    ctx.beginPath();
    ctx.ellipse(0, bottom, halfW * 0.6, (bottom - top) * 0.45, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    outline(); // rebuild the path: the tongue ellipse replaced it
    ctx.stroke();
  };

  switch (shape) {
    case 'closed':
      ctx.beginPath();
      ctx.moveTo(-w * 0.22, 0);
      ctx.quadraticCurveTo(0, w * 0.04, w * 0.22, 0);
      ctx.stroke();
      break;
    case 'smile':
      ctx.beginPath();
      ctx.moveTo(-w * 0.3, -w * 0.04);
      ctx.quadraticCurveTo(0, w * 0.18, w * 0.3, -w * 0.04);
      ctx.stroke();
      break;
    case 'frown':
      ctx.beginPath();
      ctx.moveTo(-w * 0.25, w * 0.06);
      ctx.quadraticCurveTo(0, -w * 0.1, w * 0.25, w * 0.06);
      ctx.stroke();
      break;
    case 'cat': // the ":3" / "ω" mouth
      ctx.beginPath();
      ctx.moveTo(-w * 0.3, -w * 0.02);
      ctx.quadraticCurveTo(-w * 0.15, w * 0.18, 0, 0);
      ctx.quadraticCurveTo(w * 0.15, w * 0.18, w * 0.3, -w * 0.02);
      ctx.stroke();
      break;
    case 'half':
      openShape(0, w * 0.16, w * 0.22, true);
      break;
    case 'A':
      openShape(-w * 0.02, w * 0.38, w * 0.3);
      break;
    case 'shout':
      openShape(-w * 0.1, w * 0.5, w * 0.36);
      break;
    case 'I':
      openShape(0, w * 0.1, w * 0.34, true);
      ctx.fillStyle = '#fff';
      ctx.fillRect(-w * 0.28, w * 0.005, w * 0.56, w * 0.04);
      break;
    case 'U':
      ctx.beginPath();
      ctx.ellipse(0, w * 0.06, w * 0.08, w * 0.1, 0, 0, Math.PI * 2);
      ctx.fillStyle = innerColor;
      ctx.fill();
      ctx.stroke();
      break;
    case 'E':
      openShape(0, w * 0.22, w * 0.3, true);
      break;
    case 'O':
      ctx.beginPath();
      ctx.ellipse(0, w * 0.12, w * 0.15, w * 0.2, 0, 0, Math.PI * 2);
      ctx.fillStyle = innerColor;
      ctx.fill();
      ctx.stroke();
      break;
    default:
      throw new Error(`Unknown mouth shape: ${shape}`);
  }
  ctx.restore();
}

/**
 * Convert romanized text to mouth keyframes.
 * Vowels pick A/I/U/E/O, m/b/p close the lips, spaces/punctuation rest.
 * @returns {Array<{t:number, v:string, hold:true}>} usable with Track
 */
export function textToVisemes(text, { rate = 8, mode = 'vowels' } = {}) {
  const keys = [];
  const step = 1 / rate; // rate 8 = one mouth drawing every 3 frames @24fps
  let t = 0;
  const flap = ['half', 'A', 'half', 'closed'];
  let flapI = 0;
  for (const raw of text.toLowerCase()) {
    let v = null;
    if ('aiueo'.includes(raw)) v = mode === 'vowels' ? raw.toUpperCase() : flap[flapI++ % flap.length];
    else if ('mbp'.includes(raw)) v = 'closed';
    else if (' ,.!?'.includes(raw)) v = 'closed';
    if (v === null) continue;
    if (!keys.length || keys[keys.length - 1].v !== v) keys.push({ t, v, hold: true });
    t += raw === ' ' ? step * 0.5 : raw === ',' || raw === '.' ? step * 2 : step;
  }
  keys.push({ t, v: 'closed', hold: true });
  return keys;
}
