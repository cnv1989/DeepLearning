# Lesson 2 · Anime proportions: thinking in "heads"

**Demo:** `demos/01-proportions.html` · **Code:** `src/character/proportions.js`

## Goals
- Measure a character in **head-heights** and pick a head count on purpose.
- Know which body parts change most between chibi and tall styles.

## Concepts

**1. The head is the ruler.** Draw the head first, then stack copies of its height downward. The total number of heads defines the style:

| Heads | Reads as | Typical use |
|---|---|---|
| 2–3 | Chibi / super-deformed | Gags, merch, comedic reactions |
| 4–5 | Child, or a "cute" style | Kids' shows, mascots |
| 6–7 | Teen lead | Most school, isekai and shounen protagonists |
| 7.5–8 | Adult | Older characters, mature dramas |
| 8.5+ | Heroic / fashion | Action heroes, idealized adults |

**2. Legs change, torsos don't.** Going from 8 heads to 2.5, legs shrink fastest, the head gets *wider*, and shoulders narrow. Compare `legFrac` and `headWidth` in `proportionLandmarks()`.

**3. Landmarks to memorize at ~7 heads.** Chin at 1, nipples ≈ 2, navel/elbows ≈ 3, crotch ≈ 3.5–4, fingertips at mid-thigh, knees ≈ 5–5.5.

**4. The eye line is low.** In anime the eyes sit near the vertical **middle of the head** (crown to chin), with a big forehead above. Lower eyes and a bigger forehead read as younger.

## Try it
1. Open the demo and compare the presets side by side.
2. Set *your* slider to 2, then 9. Watch the legs.
3. In the browser console run `proportionLandmarks(6.5)` and read the numbers.

## Practice in your software
1. In Krita or Clip Studio, make a new canvas and draw **8 horizontal guide lines** evenly spaced.
2. Draw the same character at 7 heads, then redraw them at 2.5 heads (chibi) next to it.
3. Keep the *design* readable: hairstyle silhouette, outfit colors, one signature accessory. A design that only works at one proportion is fragile.

## Checklist
- [ ] I can sketch a 7-head mannequin without measuring.
- [ ] My chibi version is still recognizably the same character.
