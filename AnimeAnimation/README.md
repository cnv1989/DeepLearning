# Anime Animation Course

A hands-on course for learning **anime-style character design and animation**. It has:

- **10 written lessons** (`lessons/`): concepts, exercises and "do it in your software" practice
- **9 interactive browser demos** (`demos/`) you can play with
- **Reusable JavaScript components** (`src/`) that implement the core ideas (easing, keyframes, on-twos timing, onion skin, cel shading, anime eyes and faces, lip-sync, hair physics, bone rigs, walk cycles, action effects) in small, readable files
- **Unit tests** (`tests/`) for all the math

No build step and no dependencies: just a browser (and Node 18+ if you want to run the tests).

## Quick start

```bash
cd AnimeAnimation
python3 -m http.server 8000      # or: npm start
# open http://localhost:8000
```

ES modules don't load from `file://`, so serve the folder as shown above rather than double-clicking the HTML files.

Run the tests:

```bash
npm test      # node --test, no install needed
```

## What software should I use?

Read **[Lesson 1](lessons/01-software-and-pipeline.md)** for the full guide. Short version:

| You want to… | Free | Paid / industry |
|---|---|---|
| Draw and design characters | **Krita** | **Clip Studio Paint** |
| Frame-by-frame animation | **Krita**, **OpenToonz** | Clip Studio Paint EX, Toon Boom Harmony, TVPaint |
| Rig a 2D puppet (VTuber, games) | Live2D Cubism (free tier) | Live2D Pro, Spine, Moho |
| 3D anime / toon shading | **Blender**, **VRoid Studio** | Unity / Unreal toon shaders |
| Composite and edit | **DaVinci Resolve** | After Effects |

**Recommended free starter path:** Krita → OpenToonz → Blender → DaVinci Resolve, plus a pen tablet.

## Curriculum

| # | Lesson | Demo | Components |
|---|---|---|---|
| 1 | [Software and pipeline](lessons/01-software-and-pipeline.md) | n/a | n/a |
| 2 | [Proportions in heads](lessons/02-proportions.md) | [demo](demos/01-proportions.html) | `character/proportions.js` |
| 3 | [Face, eyes and expressions](lessons/03-face-and-eyes.md) | [demo](demos/02-face-builder.html) | `character/face.js`, `eyes.js`, `mouth.js` |
| 4 | [Line art and cel shading](lessons/04-line-art-and-cel-shading.md) | [demo](demos/03-cel-shading.html) | `effects/celShading.js`, `utils/color.js` |
| 5 | [Timing and spacing](lessons/05-timing-and-spacing.md) | [demo](demos/04-timing.html) | `core/easing.js`, `keyframes.js`, `onionSkin.js`, `player.js` |
| 6 | [Limited animation: lip flap and blinks](lessons/06-limited-animation.md) | [demo](demos/05-lip-sync.html) | `character/mouth.js`, `eyes.js` |
| 7 | [Secondary motion](lessons/07-secondary-motion.md) | [demo](demos/06-secondary-motion.html) | `physics/spring.js` |
| 8 | [Rigging and walk cycles](lessons/08-rigging-and-walk-cycles.md) | [demo](demos/07-walk-cycle.html) | `rig/skeleton.js`, `walkCycle.js` |
| 9 | [Action effects](lessons/09-action-effects.md) | [demo](demos/08-effects.html) | `effects/speedLines.js` |
| 10 | [Capstone: a full cut](lessons/10-capstone-scene.md) | [demo](demos/09-scene.html) | all of the above |

There's also a [glossary](lessons/glossary.md) of the Japanese production terms (genga, douga, kuchipaku, sakuga…).

## Project layout

```
AnimeAnimation/
├── index.html              course hub
├── lessons/                written lessons (Markdown)
├── demos/                  interactive pages, one per lesson
│   └── shared/             UI helpers + styles
├── src/
│   ├── core/               easing, keyframes/tracks, playback, onion skin
│   ├── character/          proportions, face, eyes, mouth (all procedural canvas drawing)
│   ├── effects/            cel shading, speed/focus lines, smears, impact frames
│   ├── physics/            springs + Verlet hair strands
│   ├── rig/                FK skeleton + procedural walk cycle
│   └── utils/              color (hue-shifted shadows), seeded random
└── tests/                  node:test unit tests
```

## Using the components in your own code

Everything is a plain ES module. Draw on any `<canvas>`:

```js
import { drawAnimeHead } from './src/character/face.js';
import { Track, stepTime } from './src/core/keyframes.js';

const zoom = new Track([{ t: 0, v: 1, ease: 'easeOutBack' }, { t: 0.25, v: 1.4 }]);
function render(seconds) {
  const t = stepTime(seconds, 24, 2);                 // animate on twos
  drawAnimeHead(ctx, { x: 300, y: 220, scale: zoom.valueAt(t), expression: 'surprised' });
}
```

## How to study with this repo

For each lesson:
1. **Read** the lesson (10 minutes).
2. **Play** with the demo until the concept is obvious.
3. **Read the code** it points to. Every file opens with comments explaining the animation idea before the math.
4. **Practice** the exercise in your own software. This is the part that makes you an animator.
5. Tick the checklist at the bottom of the lesson.
