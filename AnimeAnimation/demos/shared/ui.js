// Tiny helpers for building demo control panels. No framework needed.

/** Make a canvas crisp on high-DPI screens. Draw in CSS pixels afterwards. */
export function setupCanvas(canvas, width, height) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.aspectRatio = `${width} / ${height}`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.W = width;
  ctx.H = height;
  return ctx;
}

export function clear(ctx, color) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (color) {
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  } else {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }
  ctx.restore();
}

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function heading(parent, text) {
  const h = document.createElement('h2');
  h.textContent = text;
  parent.append(h);
  return h;
}

export function slider(parent, { label, min, max, step = 1, value, format = (v) => v, onInput }) {
  const wrap = document.createElement('label');
  wrap.className = 'ctrl';
  const span = document.createElement('span');
  span.textContent = label;
  const out = document.createElement('output');
  const input = document.createElement('input');
  Object.assign(input, { type: 'range', min, max, step, value });
  const update = () => {
    out.textContent = format(Number(input.value));
    onInput?.(Number(input.value));
  };
  input.addEventListener('input', update);
  wrap.append(span, out, input);
  parent.append(wrap);
  out.textContent = format(Number(value));
  return input;
}

export function select(parent, { label, options, value, onChange }) {
  const wrap = document.createElement('label');
  wrap.className = 'ctrl';
  const span = document.createElement('span');
  span.textContent = label;
  const sel = document.createElement('select');
  for (const o of options) {
    const opt = document.createElement('option');
    const [v, text] = Array.isArray(o) ? o : [o, o];
    opt.value = v;
    opt.textContent = text;
    sel.append(opt);
  }
  sel.value = value;
  sel.addEventListener('change', () => onChange?.(sel.value));
  wrap.append(span, document.createElement('span'), sel);
  parent.append(wrap);
  return sel;
}

export function checkbox(parent, { label, checked = false, onChange }) {
  const wrap = document.createElement('label');
  wrap.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = checked;
  input.addEventListener('change', () => onChange?.(input.checked));
  wrap.append(input, document.createTextNode(label));
  parent.append(wrap);
  return input;
}

export function buttons(parent, list) {
  const wrap = document.createElement('div');
  wrap.className = 'buttons';
  const els = list.map(({ label, onClick, primary }) => {
    const b = document.createElement('button');
    b.textContent = label;
    if (primary) b.className = 'primary';
    b.addEventListener('click', onClick);
    wrap.append(b);
    return b;
  });
  parent.append(wrap);
  return els;
}

export function textInput(parent, { label, value, onInput, multiline = false }) {
  const wrap = document.createElement('label');
  wrap.className = 'ctrl';
  const span = document.createElement('span');
  span.textContent = label;
  const input = document.createElement(multiline ? 'textarea' : 'input');
  if (!multiline) input.type = 'text';
  input.value = value;
  input.addEventListener('input', () => onInput?.(input.value));
  wrap.append(span, document.createElement('span'), input);
  parent.append(wrap);
  return input;
}
