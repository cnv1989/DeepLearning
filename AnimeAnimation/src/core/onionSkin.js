// Onion skinning: show faded, tinted copies of neighbouring drawings so you
// can judge spacing — exactly the feature in Krita / Clip Studio / OpenToonz.

let scratch = null;

function getScratch(w, h) {
  if (!scratch) scratch = document.createElement('canvas');
  if (scratch.width !== w || scratch.height !== h) {
    scratch.width = w;
    scratch.height = h;
  }
  return scratch;
}

/**
 * @param {CanvasRenderingContext2D} ctx destination
 * @param {(ctx:CanvasRenderingContext2D, t:number)=>void} drawAt draws the scene at time t
 * @param {number} t current time
 * @param {{before?:number, after?:number, step?:number, alpha?:number,
 *          pastColor?:string, futureColor?:string}} opts step = seconds between ghosts
 */
export function drawOnionSkin(ctx, drawAt, t, opts = {}) {
  const {
    before = 3, after = 0, step = 1 / 12, alpha = 0.35,
    pastColor = '#e0457b', futureColor = '#2f9e6b',
  } = opts;
  const { width, height } = ctx.canvas;
  const s = getScratch(width, height);
  const sctx = s.getContext('2d');

  const ghost = (tg, color, a) => {
    sctx.setTransform(1, 0, 0, 1, 0, 0);
    sctx.globalCompositeOperation = 'source-over';
    sctx.clearRect(0, 0, width, height);
    sctx.setTransform(ctx.getTransform());
    drawAt(sctx, tg);
    sctx.setTransform(1, 0, 0, 1, 0, 0);
    sctx.globalCompositeOperation = 'source-in';
    sctx.fillStyle = color;
    sctx.fillRect(0, 0, width, height);
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = a;
    ctx.drawImage(s, 0, 0);
    ctx.restore();
  };

  for (let i = before; i >= 1; i--) {
    const tg = t - i * step;
    if (tg >= 0) ghost(tg, pastColor, alpha * (1 - (i - 1) / (before + 1)));
  }
  for (let i = 1; i <= after; i++) ghost(t + i * step, futureColor, alpha * (1 - (i - 1) / (after + 1)));
}
