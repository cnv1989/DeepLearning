# Lesson 10 · Capstone: build a complete cut

**Demo:** `demos/09-scene.html` · **Code:** everything, combined in the `CUT` object at the top of the demo's script

## Goals
- Plan a shot on an **exposure sheet** before animating.
- Combine character, acting, dialogue, camera and effects in one cut.
- Take the cut from this repo into your own software.

## The cut (4.5 seconds, 108 frames)
| Frames | Action | Techniques |
|---|---|---|
| 0–20 | Looks off-screen, still | Hold (Lesson 6) |
| 20–28 | Turns to camera | Ease-out, `easeOutBack` tilt (Lesson 5) |
| 30–70 | Speaks: "ne, kiite yo… e?!" | Lip-sync + blinks (Lesson 6) |
| 78–84 | Surprise! Camera push-in | Overshoot zoom, focus lines (Lesson 9) |
| 84–108 | Reaction hold, then nervous | Expression swap, sparkles (Lessons 3, 9) |

## Exposure sheets
An X-sheet has **one row per frame** (in this demo, one column per frame) and one column per element: dialogue, each animation layer, camera instructions. A mark means "new drawing here"; a blank means "hold the previous one". The strip under the demo shows this. Darker blocks are new keys or drawings.

Click the strip to scrub. Notice how *sparse* most rows are. That sparseness is limited animation.

## Assignment

### Part A: remix the code (30 min)
Edit `CUT` in `demos/09-scene.html`:
1. Change `LINE` to your own sentence.
2. Move the surprise 12 frames earlier (change `F(78)` keys).
3. Swap `'surprised'` for `'angry'` and the effect to focus lines plus an impact frame (import `applyImpactFrame`).

### Part B: make it for real (a weekend)
1. **Storyboard**: 4–6 thumbnail panels on paper.
2. **Character sheet**: your character from Lessons 2–4 (front view, 6 expressions, color key).
3. **Animate** in your chosen app at 24 fps:
   - Background on its own layer.
   - Head and body held, with mouth and eye layers swapped per the X-sheet.
   - One camera move (zoom or pan) and one effect beat.
   - Mostly on twos, with a few drawings on threes.
4. **Composite and sound**: add a voice line (record yourself) in Resolve or After Effects and check lip-sync.
5. Export an MP4 and share it.

## Where to go next
- **Study real cuts frame by frame.** Step through anime clips you like one frame at a time (most video players have a frame-step key) and write down the timing of a blink, a head turn and a punch.
- **Sakuga collections.** Search for "sakuga" clips of animators whose work you like, and note how they use smears, effects and timing.
- **Books:** *The Animator's Survival Kit* (Richard Williams) for timing and spacing; *Cartoon Animation* (Preston Blair) for construction and walks.
- **Extend this repo:** add a 3/4-view head, an IK solver for feet, a run cycle in `walkCycle.js`, or a cloth skirt using `Strand`s side by side.

## Final checklist
- [ ] I storyboarded the cut before animating.
- [ ] I filled out an X-sheet (even a rough one).
- [ ] My cut uses a hold, a lip-synced line, a blink, a camera move and one effect.
- [ ] I exported and watched it at full speed, then fixed the one thing that bothered me most.
