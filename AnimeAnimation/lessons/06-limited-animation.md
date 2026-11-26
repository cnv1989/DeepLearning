# Lesson 6 · Limited animation: lip flap, blinks and holds

**Demo:** `demos/05-lip-sync.html` · **Code:** `src/character/mouth.js` (`textToVisemes`), `src/character/eyes.js` (`blinkOpenness`)

## Goals
- Animate dialogue with the fewest drawings that still sell it.
- Time a believable blink.
- Use **holds** deliberately, as a budget tool and an acting tool.

## Concepts

### Limited animation
TV anime makes hundreds of minutes a year, so it saves drawings everywhere:
- **Hold** the body and animate only the mouth and eyes.
- **Reuse** cycles and stock shots (*bank* footage).
- Use **camera moves over still drawings**: pans, slow zooms, and the "slide" of a still character.
- Spend the saved budget on the moments that matter, the *sakuga* scenes.

### Lip flap (kuchipaku)
The classic system uses **three mouth drawings**: closed, half and open. They cycle roughly **on threes** while the voice plays. It doesn't match the words exactly, and audiences accept that.

Higher-budget or game work uses **vowel shapes** for the five Japanese vowels: **A, I, U, E, O**. Close the lips for **m, b, p**.

```js
const keys = textToVisemes('konnichiwa', { rate: 8 });   // 8 mouth changes/sec = on threes
const mouth = new Track(keys);                           // string values hold until next key
mouth.valueAt(t);  // -> 'O' | 'I' | 'A' | 'closed' ...
```

### Blinks
A natural blink is about **6 frames on ones**: open → half → closed → closed → half → open (`blinkOpenness`). Use these rules:
- Blink on **head turns** and **changes of thought**. It reads as the character thinking.
- A slow blink reads as tired, flirty or unimpressed.
- Don't blink both eyes at mismatched times unless it's a wink.

## Try it
1. Type your own line in the demo. Compare the *vowel* and *3-shape flap* modes.
2. Lower *mouth changes / sec* to 4. Where does it stop reading as speech?

## Practice in your software
1. **Krita / Clip Studio:** draw your character's face once. On separate layers, draw 3 mouths (closed, half, open) and 4 eye states (open, half, closed, happy). Swap the layers on a timeline to a recorded line of dialogue, then sync to the audio track (both apps can import audio).
2. **Live2D path:** those same layers are exactly what Live2D needs. Rig *ParamMouthOpenY* and *ParamEyeLOpen/ROpen*, and use its built-in lip-sync from audio.

## Checklist
- [ ] I animated a line of dialogue with only 3 mouth drawings.
- [ ] My blink is 6 frames or fewer.
- [ ] I can name three ways TV anime saves drawings.
