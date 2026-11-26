# Lesson 3 · The anime face, eyes and expressions

**Demo:** `demos/02-face-builder.html` · **Code:** `src/character/face.js`, `eyes.js`, `mouth.js`

## Goals
- Construct a front-view anime head from simple shapes.
- Draw the anime eye in its correct layer order.
- Build a set of expressions for a model sheet.

## Concepts

### Head construction (turn on "construction lines" in the demo)
1. **Circle** for the cranium.
2. **Center line** and **eye line**. The eye line sits around the circle's lower half.
3. **Jaw**: from the sides of the circle, curve in to a small chin. Chin pointiness and jaw width are the strongest "age/style" dials after eye size.
4. **Features**: big eyes, a nose that is often just a tiny shadow tick, a small mouth.
5. **Hair** goes on last and sits *outside* the skull, because hair has volume. Draw the skull first even when bangs will cover it.

### Anatomy of the anime eye (drawn in this order in `drawAnimeEye`)
1. **Sclera**: the white shape, clipped between the upper and lower lid curves.
2. **Iris**: *taller than wide*, gradient dark at the top and light at the bottom.
3. **Pupil**.
4. **Lid shadow**: a flat band across the top of the eyeball. This one cel shadow makes the eye look round.
5. **Highlights**: 1–3 pure white shapes, on the **same side in both eyes** (one light source!). The code un-mirrors them for that reason.
6. **Upper lash line**: the thickest line on the face, tapered, with a flick at the outer corner.
7. **Lower lash**: thin and partial, or omitted for shounen styles.

Style comparison (`EYE_STYLES`): *shoujo* = tall eyes, big iris, many highlights. *Shounen* = narrower, sharper. *Mature* = small iris, more lid.

### Expressions = brows + mouth + eye shape
Look at `EXPRESSIONS` in `face.js`. Each one is only a few numbers:
- **Brows** carry most of the emotion: inner ends down = angry, inner ends up = sad or worried.
- **Eye shape**: `happy` replaces the eye with a ^ arc, and `sharp` flattens the top lid.
- **Symbols**: anger vein, sweat drop, tears and blush lines are a shared visual vocabulary. Use them!

## Try it
1. Make the face look older using only three sliders.
2. Flip between *shoujo* and *shounen* eye styles.
3. Click every expression and say which feature does the most work.

## Practice in your software
1. Draw **one character, eight expressions** on one page (a model sheet). Keep the head shape identical across all eight. Trace your own construction circle.
2. Draw the eye in **separate layers** in the order above. This is also exactly how you'd prepare a Live2D model (Lesson 6).

## Checklist
- [ ] My eye highlights are on the same side in both eyes.
- [ ] I can draw angry, sad, happy and surprised by changing only brows, mouth and eye shape.
