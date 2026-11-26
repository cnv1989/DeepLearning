// 2D cut-out rig with forward kinematics (FK) — the idea behind bones in
// Blender, Spine, Live2D deformers, Toon Boom pegs and Krita's transform masks.
// Each bone has a local angle relative to its parent; rotating a parent
// carries every child with it.

const DEG = Math.PI / 180;

export class Bone {
  constructor(name, length, { angle = 0, offset = null, width = 10 } = {}) {
    this.name = name;
    this.length = length;
    this.angle = angle; // degrees, local to parent
    this.offset = offset; // [along, side] attach point on the parent; null = parent's end
    this.width = width;
    this.parent = null;
    this.children = [];
    // World transform results:
    this.start = { x: 0, y: 0 };
    this.end = { x: 0, y: 0 };
    this.worldAngle = 0; // radians
  }

  add(child) {
    child.parent = this;
    this.children.push(child);
    return child;
  }
}

export class Skeleton {
  constructor(root) {
    this.root = root;
    this.position = { x: 0, y: 0 };
    this.bones = new Map();
    this._index(root);
  }

  _index(b) {
    this.bones.set(b.name, b);
    b.children.forEach((c) => this._index(c));
  }

  get(name) {
    return this.bones.get(name);
  }

  /** Apply a pose: { boneName: angleDegrees }. */
  setPose(pose) {
    for (const [name, angle] of Object.entries(pose)) {
      const b = this.bones.get(name);
      if (b) b.angle = angle;
    }
    return this;
  }

  /** Forward kinematics: compute world start/end for every bone. */
  solve() {
    const visit = (b, ox, oy, parentAngle) => {
      const wa = parentAngle + b.angle * DEG;
      b.worldAngle = wa;
      b.start = { x: ox, y: oy };
      b.end = { x: ox + Math.cos(wa) * b.length, y: oy + Math.sin(wa) * b.length };
      for (const c of b.children) {
        // offset[0] = distance along parent from its START (default: its end)
        const along = c.offset ? c.offset[0] : b.length;
        const side = c.offset ? c.offset[1] : 0;
        const cx = ox + Math.cos(wa) * along - Math.sin(wa) * side;
        const cy = oy + Math.sin(wa) * along + Math.cos(wa) * side;
        visit(c, cx, cy, wa);
      }
    };
    visit(this.root, this.position.x, this.position.y, 0);
    return this;
  }

  forEach(fn) {
    const visit = (b) => { fn(b); b.children.forEach(visit); };
    visit(this.root);
  }
}

/** Build a simple side-view humanoid. Units are pixels for a ~`h` tall figure. */
export function createHumanoid(h = 300) {
  const u = h / 7; // ~7 heads tall
  const hips = new Bone('hips', 1, { angle: -90, width: u * 0.9 });
  const spine = hips.add(new Bone('spine', u * 2.1, { angle: 0, offset: [0, 0], width: u * 0.8 }));
  const neck = spine.add(new Bone('neck', u * 0.35, { angle: 0, width: u * 0.3 }));
  neck.add(new Bone('head', u * 0.9, { angle: 0, width: u * 0.85 }));

  for (const side of ['L', 'R']) {
    const upperArm = spine.add(new Bone(`upperArm${side}`, u * 1.3, { angle: 180, offset: [u * 1.9, 0], width: u * 0.32 }));
    const foreArm = upperArm.add(new Bone(`foreArm${side}`, u * 1.15, { angle: 0, width: u * 0.28 }));
    foreArm.add(new Bone(`hand${side}`, u * 0.4, { angle: 0, width: u * 0.25 }));
    const thigh = hips.add(new Bone(`thigh${side}`, u * 1.75, { angle: 180, offset: [0, 0], width: u * 0.45 }));
    const shin = thigh.add(new Bone(`shin${side}`, u * 1.75, { angle: 0, width: u * 0.36 }));
    shin.add(new Bone(`foot${side}`, u * 0.55, { angle: -90, width: u * 0.2 }));
  }
  return new Skeleton(hips);
}
