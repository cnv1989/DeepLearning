# Lesson 7 · Secondary motion: hair, cloth and follow-through

**Demo:** `demos/06-secondary-motion.html` · **Code:** `src/physics/spring.js`

## Goals
- Separate **primary** action from **secondary** action.
- Use **drag, overshoot and settle** to make hair and cloth feel real.
- Map those ideas to physics settings in Live2D and Blender.

## Concepts

### Follow-through and overlapping action
When the head stops, the hair doesn't. It keeps going, **overshoots**, swings back and **settles**. Parts further from the root lag more. This is one of Disney's 12 principles, and anime pushes it hard: long hair, ribbons, skirts and coat tails are a big part of the style's appeal.

### Model 1: a damped spring
```js
const sway = new Spring({ stiffness: 60, damping: 3 });
angle = sway.step(targetAngle, dt);
```
- **Stiffness**: how hard it pulls back. Higher is snappier.
- **Damping**: how fast the wobble dies. Lower is bouncier and more cartoony.
The demo drives the ahoge (antenna strand) with one spring.

### Model 2: a chain (Verlet strand)
A twin-tail is a chain of points. Each frame we:
1. move points by their velocity plus gravity,
2. pull them toward the **rest shape** (hair "remembers" its style, stronger at the root),
3. fix segment lengths so the strand can't stretch.

### Hand-drawn equivalent
In frame-by-frame animation you don't simulate. You **offset**: draw the hair reaching the pose a couple of frames *after* the head, then overshoot a little, then settle. Use onion skin to check the arcs are smooth.

### Snappy primary, loose secondary
The demo's auto mode moves the head **pose-to-pose**: hold, quick move, hold. That's very anime. The fast head stop gives the hair lots of energy to show follow-through. A floaty head kills the effect.

## Try it
1. Set stiffness high and damping high. The hair looks gelled.
2. Set stiffness low and damping low. It looks like silk in water.
3. Find settings that match *your* character's hair.

## Practice in your software
- **Frame-by-frame (Krita / CSP / OpenToonz):** animate a 1-second head turn on twos with a ponytail that lags 2 frames behind, overshoots, and settles.
- **Live2D:** *Physics settings*: input = head angle X, output = hair parameters. Tune the pendulum length, sway and delay. They map closely to `Strand` and `Spring`.
- **Blender:** spring bones (add-on), cloth simulation on a hair mesh, or keyframe the bones with offset timing.

## Checklist
- [ ] I can point to primary vs secondary action in a clip.
- [ ] My hair arcs smoothly and settles instead of stopping dead.
