// Secondary motion: things that follow the main action (hair, ribbons,
// skirts, earrings) lag behind, overshoot and settle. A damped spring is the
// simplest model of "follow-through and overlapping action".

export class Spring {
  /**
   * @param {{stiffness?:number, damping?:number, mass?:number, value?:number}} opts
   *  stiffness: how hard it pulls toward the target (higher = snappier)
   *  damping:   how quickly the wobble dies out (higher = less bounce)
   */
  constructor({ stiffness = 120, damping = 10, mass = 1, value = 0 } = {}) {
    this.k = stiffness;
    this.c = damping;
    this.m = mass;
    this.value = value;
    this.velocity = 0;
  }

  step(target, dt) {
    // Semi-implicit Euler with sub-steps for stability.
    const n = Math.max(1, Math.ceil(dt / (1 / 240)));
    const h = dt / n;
    for (let i = 0; i < n; i++) {
      const force = -this.k * (this.value - target) - this.c * this.velocity;
      this.velocity += (force / this.m) * h;
      this.value += this.velocity * h;
    }
    return this.value;
  }

  get settled() {
    return Math.abs(this.velocity) < 1e-3;
  }
}

/**
 * A hair strand / ribbon as a Verlet chain.
 * The root follows an anchor (the head); each segment tries to keep its
 * length and gently returns toward its rest direction (hair "style memory").
 */
export class Strand {
  /**
   * @param {{x:number,y:number}} root
   * @param {{segments?:number, length?:number, restAngle?:number, curl?:number,
   *          stiffness?:number, gravity?:number, drag?:number}} opts
   *  restAngle: radians, direction the strand points at rest (PI/2 = down)
   *  curl: extra angle added per segment (makes the strand curve)
   */
  constructor(root, opts = {}) {
    const {
      segments = 6, length = 60, restAngle = Math.PI / 2, curl = 0,
      stiffness = 0.25, gravity = 400, drag = 0.04,
    } = opts;
    this.segLen = length / segments;
    this.restAngle = restAngle;
    this.curl = curl;
    this.stiffness = stiffness;
    this.gravity = gravity;
    this.drag = drag;
    this.points = [];
    let a = restAngle, x = root.x, y = root.y;
    for (let i = 0; i <= segments; i++) {
      this.points.push({ x, y, px: x, py: y });
      x += Math.cos(a) * this.segLen;
      y += Math.sin(a) * this.segLen;
      a += curl;
    }
  }

  /** Rest position of point i given the current root and a head rotation. */
  restPoint(i, rotation = 0) {
    const r = this.points[0];
    let x = r.x, y = r.y, a = this.restAngle + rotation;
    for (let j = 0; j < i; j++) {
      x += Math.cos(a) * this.segLen;
      y += Math.sin(a) * this.segLen;
      a += this.curl;
    }
    return { x, y };
  }

  step(root, dt, rotation = 0, iterations = 4) {
    const p = this.points;
    p[0].x = p[0].px = root.x;
    p[0].y = p[0].py = root.y;
    for (let i = 1; i < p.length; i++) {
      const q = p[i];
      const vx = (q.x - q.px) * (1 - this.drag);
      const vy = (q.y - q.py) * (1 - this.drag);
      q.px = q.x;
      q.py = q.y;
      q.x += vx;
      q.y += vy + this.gravity * dt * dt;
    }
    for (let it = 0; it < iterations; it++) {
      // Shape memory: pull toward rest pose, weaker toward the tip.
      for (let i = 1; i < p.length; i++) {
        const rp = this.restPoint(i, rotation);
        const w = this.stiffness * (1 - (i - 1) / p.length) / iterations;
        p[i].x += (rp.x - p[i].x) * w;
        p[i].y += (rp.y - p[i].y) * w;
      }
      // Length constraints.
      for (let i = 1; i < p.length; i++) {
        const a = p[i - 1], b = p[i];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 1e-6;
        const diff = (d - this.segLen) / d;
        if (i === 1) {
          b.x -= dx * diff;
          b.y -= dy * diff;
        } else {
          a.x += dx * diff * 0.5;
          a.y += dy * diff * 0.5;
          b.x -= dx * diff * 0.5;
          b.y -= dy * diff * 0.5;
        }
      }
      p[0].x = root.x;
      p[0].y = root.y;
    }
    return p;
  }
}
