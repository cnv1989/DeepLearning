# Lesson 8 · Rigging and the walk cycle

**Demo:** `demos/07-walk-cycle.html` · **Code:** `src/rig/skeleton.js`, `src/rig/walkCycle.js`

## Goals
- Understand a bone hierarchy and **forward kinematics**.
- Know the four key poses of a walk and where the body bob comes from.
- Give a walk **personality**.

## Concepts

### Bones and forward kinematics (FK)
A rig is a tree:
```
hips ─┬─ spine ─┬─ neck ─ head
      │         ├─ upperArmL ─ foreArmL ─ handL
      │         └─ upperArmR ─ foreArmR ─ handR
      ├─ thighL ─ shinL ─ footL
      └─ thighR ─ shinR ─ footR
```
Each bone stores an angle **relative to its parent**. `solve()` walks the tree and accumulates the angles, so rotating the thigh moves the shin and foot with it. Blender armatures, Spine, Moho, Toon Boom pegs and Live2D deformers all work this way.

(*Inverse kinematics*, IK, solves the opposite problem: "put the foot *here*, work out the angles". It's useful for planting feet, and `plantOnGround()` is a tiny cousin of it.)

### The walk: four key poses per step
| Pose | What happens | Body height |
|---|---|---|
| **Contact** | Front heel touches down, legs widest | Middle |
| **Down** | Weight lands, front knee bends | **Lowest** |
| **Passing** | Back leg swings past the standing leg | Middle |
| **Up** | Standing leg pushes off | **Highest** |

Then the same again with the other leg. Arms swing **opposite** to the legs.

### Budget walks
Classic TV anime walks are often **8 drawings per cycle on threes** (one second per two steps), with the background panning behind. Set the demo to 8 drawings and compare with 24.

### Personality comes from the poses
`WALK_STYLES` changes only five numbers: stride, arm swing, lean, knee lift and bounce. Lean forward with a small stride reads as tired. A big stride, big arms and an upright back read as confident.

## Try it
1. Show bones. Watch how the foot follows the thigh.
2. Step drawing by drawing and name each key pose.
3. Make a new style in `WALK_STYLES` (e.g. "sad": big lean, tiny stride, almost no arms).

## Practice in your software
- **Frame-by-frame:** draw contact → passing → contact (3 keys) for one step, then add down and up. Mirror for the second step. Loop it on twos.
- **Krita / CSP cut-out:** split a character into limb layers and rotate them per key (use pivot points at the joints).
- **Blender:** a Rigify or simple armature, keyframe the four poses, add a *Cycles* modifier on the F-curves to loop, and add *Stepped Interpolation* for anime timing.

## Checklist
- [ ] I can draw the four key poses from memory.
- [ ] My character is lowest on *down* and highest on *up*.
- [ ] I made two walks with different personalities.
