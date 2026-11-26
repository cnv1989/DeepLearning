// Procedural walk cycle for the humanoid skeleton (side view, facing right).
//
// The four classic walk key poses, per step (half cycle):
//   CONTACT  (q = 0)     front heel touches the ground, legs spread widest
//   DOWN     (q ~ .125)  weight lands, knee bends, body is LOWEST
//   PASSING  (q = .25)   free leg passes the standing leg
//   UP       (q ~ .375)  pushing off, body is HIGHEST
// The second half (q = .5 .. 1) mirrors it with the other leg.

const TAU = Math.PI * 2;

/**
 * @param {number} phase 0..1 through one full cycle (two steps)
 * @param {{stride?:number, kneeLift?:number, armSwing?:number, lean?:number, bounce?:number}} style
 *   Tweak these to get different personalities: a big stride + big arm swing
 *   reads confident; low stride + forward lean reads tired or sneaky.
 */
export function walkPose(phase, style = {}) {
  const { stride = 25, kneeLift = 60, armSwing = 20, lean = 5, bounce = 1 } = style;
  const pose = {};
  const legs = { L: phase % 1, R: (phase + 0.5) % 1 };

  for (const [side, q] of Object.entries(legs)) {
    const swing = stride * Math.cos(TAU * q);
    const knee = q < 0.5
      ? 5 + 15 * bounce * Math.sin(TAU * q)          // stance: soft bend on "down"
      : 5 + kneeLift * Math.sin(TAU * (q - 0.5));    // swing: lift the foot clear
    const thighWorld = 90 - swing;
    const shinWorld = thighWorld + knee;
    const toeDrop = q >= 0.5 ? 20 * Math.sin(TAU * (q - 0.5)) : 0;
    pose[`thigh${side}`] = 180 - swing;
    pose[`shin${side}`] = knee;
    pose[`foot${side}`] = -shinWorld + toeDrop;

    // Arms swing opposite to the leg on the same side.
    const a = -armSwing * Math.cos(TAU * q);
    pose[`upperArm${side}`] = 180 - a;
    pose[`foreArm${side}`] = -(12 + 12 * Math.max(0, a / Math.max(armSwing, 1e-6)));
    pose[`hand${side}`] = -5;
  }
  pose.spine = lean;
  pose.neck = -lean * 0.4;
  pose.head = -lean * 0.4;
  return pose;
}

/**
 * Plant the skeleton on the ground: shift it vertically so the lowest foot
 * point touches groundY. This automatically creates the up/down body bob.
 */
export function plantOnGround(skeleton, groundY) {
  skeleton.solve();
  let maxY = -Infinity;
  skeleton.forEach((b) => {
    if (b.name.startsWith('foot') || b.name.startsWith('shin')) maxY = Math.max(maxY, b.end.y, b.start.y);
  });
  skeleton.position.y += groundY - maxY;
  skeleton.solve();
  return skeleton;
}

export const WALK_STYLES = {
  normal: {},
  confident: { stride: 32, armSwing: 30, lean: 2, kneeLift: 55 },
  tired: { stride: 15, armSwing: 6, lean: 14, kneeLift: 35, bounce: 0.4 },
  sneaky: { stride: 20, armSwing: 8, lean: 18, kneeLift: 75, bounce: 1.6 },
  bouncy: { stride: 28, armSwing: 26, lean: 0, kneeLift: 80, bounce: 2 },
};
