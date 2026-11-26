# Lesson 9 · Action effects: anticipation, smears, speed lines, impact frames

**Demo:** `demos/08-effects.html` · **Code:** `src/effects/speedLines.js`, `src/utils/random.js`

## Goals
- Structure an action as **anticipation → action → impact → follow-through**.
- Use anime's effect toolkit to sell speed and force with few drawings.

## Concepts

### The action beat
1. **Anticipation**: wind up the opposite way, then **hold**. The hold is where tension builds, and anime holds longer than Western animation.
2. **Action**: very few frames (2–4). Speed comes from how *few* drawings there are.
3. **Impact**: the moment of contact gets 1–3 special frames.
4. **Follow-through**: the energy dissipates. Shake, debris, lines and sparkles settle.

### The toolkit
| Effect | What it is | Code |
|---|---|---|
| **Smear** | A single drawing stretched along the motion path, sometimes with "multiples" or drybrush streaks | `drawSmear` |
| **Speed lines** | Parallel streaks behind a moving object or across the frame | `drawParallelSpeedLines` |
| **Focus lines** (*shuuchuusen*) | Radial lines pointing at the subject, for impact, shock or reveals | `drawFocusLines` |
| **Impact frame** | 1–2 frames of inverted or high-contrast color | `applyImpactFrame` |
| **Screen shake** | Offset the whole frame randomly, decaying | demo `scene()` |
| **Boil** | Redraw lines every 2–3 frames with slight variation so they feel alive | new `seed` per drawing |
| **Sparkles** | *Kira-kira*, for cute or dramatic accents | `drawSparkle` |

### Why seeded randomness?
`mulberry32(seed)` returns the same "random" numbers for the same seed. Using `seed = Math.floor(frame / 3)` makes the focus lines **change every 3 frames**, which is exactly how hand-drawn effects boil on threes.

## Try it
1. Turn every effect off, then add them back one at a time. Which adds the most?
2. Turn off *anticipation + hold*. The hit feels weaker even though nothing else changed.

## Practice in your software
1. Animate a **punch** in 12 drawings or fewer: 4 for anticipation (plus a hold), 2 for the action as smears, 2 impact frames, and the rest for follow-through.
2. Use speed lines on their own layer. Make them in Clip Studio with the *Saturated line* / *Streamline* ruler, or in Krita with the *Multibrush* or *Perspective* assistants.
3. Composite the impact frame with an inverted adjustment layer (Krita / CSP), or a Negative effect in Resolve or After Effects.

## Checklist
- [ ] My action has a visible hold before it.
- [ ] I used at least one smear frame.
- [ ] I can explain why effects are re-drawn every few frames.
