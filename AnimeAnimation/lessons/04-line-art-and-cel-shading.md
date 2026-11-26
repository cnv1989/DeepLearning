# Lesson 4 · Line art and cel shading

**Demo:** `demos/03-cel-shading.html` · **Code:** `src/effects/celShading.js`, `src/utils/color.js`

## Goals
- Understand cel shading as **quantized lighting**.
- Pick shadow colors like a studio color designer.
- Set up the layer structure for coloring.

## Concepts

### Cel shading = lighting with steps
Real light changes smoothly across a surface. Anime splits it into **2–3 flat tones** with hard edges:

```
ndl = dot(surfaceNormal, lightDirection)   // -1 (facing away) .. 1 (facing light)
tone = ndl > threshold ? BASE : SHADOW      // toonBand() in the code
```

The **threshold** decides how much of the form is in shadow: low for a bright, cheerful look and high for a dramatic one. In Blender this is a *Color Ramp* set to *Constant*.

### Shadow color is a design choice
Don't make shadows by "base color + black". That gives grey, muddy color. Studios pick a shadow color that is:
- **darker**,
- **hue-shifted**, usually toward blue/purple for cool ambient light, or toward red for skin (blood under the surface),
- about **as saturated** as the base.

`shadowColor()` does this. Toggle *hue-shifted shadows* in the demo and compare the swatches.

### Highlights and rim light
- **Specular highlight**: a small hard white or light shape. Use it on hair (the "angel ring"), eyes, metal, lips.
- **Rim light**: a thin bright edge on the shadow side. It separates the character from a dark background and appears constantly in action scenes.

### Line art
- **Outer silhouette lines thicker, inner detail lines thinner.** This one rule makes line art look professional.
- Taper line ends. Most apps have pressure-sensitive brushes; make sure pressure is on.
- Close your shapes so the fill bucket works. Krita has *Colorize Mask* and *Close Gap*, and Clip Studio has *Close gap* in the fill tool.

### Layer structure (Krita / Clip Studio)
```
Character
 ├─ Line art          (Normal, top)
 ├─ Highlights        (Normal or Add)
 ├─ Shadows           (Multiply, clipped to Flats; or painted with chosen shadow colors)
 └─ Flats             (one solid color per material: skin, hair, shirt...)
```
Paint the shadow as **one shape per area**. For example, the shadow under the bangs is one shape, not many soft strokes.

## Try it
1. Drag the light around. Notice the shadow shape stays simple.
2. Switch between 2 and 3 tones. Three tones look richer but cost more paint time per frame, which matters at 12 drawings a second.

## Practice in your software
1. Take your Lesson 3 character, make **flats**, then add **one shadow layer** with a light from the top-left.
2. Build a **color key**: base, shadow and highlight for skin, hair, eyes and each clothing piece. Studios hand this palette to every colorist.
3. (Blender) Put a toon shader on a sphere: *Diffuse BSDF → Shader to RGB → Color Ramp (Constant)*.

## Checklist
- [ ] My shadows are a chosen color, not a grey overlay.
- [ ] Silhouette lines are thicker than interior lines.
- [ ] I have a color key for my character.
