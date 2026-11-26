# Lesson 5 · Timing, spacing, keys and in-betweens

**Demo:** `demos/04-timing.html` · **Code:** `src/core/easing.js`, `src/core/keyframes.js`, `src/core/onionSkin.js`

## Goals
- Separate **timing** (how many frames) from **spacing** (where the drawings fall).
- Understand **keys** (*genga*) vs **in-betweens** (*douga*).
- Know why anime is animated **on twos and threes**.

## Concepts

### Timing vs spacing
- **Timing** = how long an action takes, in frames. A blink takes about 6 frames and a punch about 4.
- **Spacing** = how far the object moves between drawings. Close spacing is slow and wide spacing is fast.

An easing function *is* a spacing chart: `easeInQuad` puts the early drawings close together (slow start), and `easeOutQuad` puts them close at the end (slow finish). The vertical ruler in the demo is the **timing chart** animators draw next to their key poses. `timingChart()` computes it.

### Keys and in-betweens
```js
const y = new Track([
  { t: 0.0, v: 110, ease: 'easeInQuad' },   // key: top of arc
  { t: 0.5, v: 440, ease: 'easeOutQuad' },  // key: contact with ground
  { t: 0.9, v: 230 },                       // key: top of next arc
]);
y.valueAt(0.25);  // an in-between, computed
```
A key artist draws the extremes and writes a timing chart. An in-betweener draws the rest. In software, *you* set keys and the app tweens them.

### Ones, twos and threes
Anime is projected at 24 fps, but most drawings are **held**:
- **On ones**: 24 drawings/sec. Smooth and expensive, kept for big action or fast camera pans.
- **On twos**: 12 drawings/sec. The everyday standard.
- **On threes**: 8 drawings/sec. Common in limited TV animation. Mixing in threes gives anime its snappy texture.

`stepTime(t, 24, 2)` returns the time of the drawing on screen, and `drawingCount(1, 24, 2) === 12`.

> A 3D animation viewed **on ones** often looks "too smooth" next to anime. Stepped timing is one of the biggest reasons 2D and toon-shaded 3D look different.

### Squash and stretch
Objects stretch along their motion when moving fast and squash on impact. Keep the **volume** constant (`sx = 1/sy`).

### Onion skinning
Seeing faded previous drawings is how you check spacing. The ghosts should get closer together as the object slows down.

## Try it
1. Switch spacing to *linear*. Why does it look weightless?
2. Step through with onion skin on. Find where spacing is widest.
3. Compare ones, twos and threes, and watch the drawing count.

## Practice in your software
1. **The bouncing ball** is the classic first exercise. In Krita (or OpenToonz):
   - 24 fps, about 2 seconds.
   - Draw keys first (tops and contacts), then in-betweens using the timing chart.
   - Hold every drawing 2 frames (on twos).
2. Redo it as a **heavy bowling ball** (less bounce, less squash) and a **beach ball** (more hang time at the top).

## Checklist
- [ ] I can explain timing vs spacing in one sentence each.
- [ ] My ball slows at the top and is fastest at contact.
- [ ] I animated something on twos.
