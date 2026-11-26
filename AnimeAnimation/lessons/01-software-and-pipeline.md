# Lesson 1 · Software and the anime production pipeline

**Goal:** choose your tools and understand how a shot travels from idea to screen, so every later lesson has a place in the bigger picture.

---

## 1. How a TV anime shot gets made

Every program below exists to do one or more of these steps. Learn the steps and you can learn any tool.

| Step | Japanese term | What happens | Covered in |
|---|---|---|---|
| Script and storyboard | *kyakuhon*, *ekonte* | Shots, camera, timing roughs | Lesson 10 |
| Character design | *settei* | Model sheets: turnarounds, expressions, color keys | Lessons 2–4 |
| Layout | *layout* | Background plus character placement, camera moves | Lesson 10 |
| Key animation | *genga* | The important poses (keyframes) | Lessons 5, 8 |
| In-betweening | *douga* | Drawings between the keys | Lesson 5 |
| Coloring | *shiage* | Flat colors plus cel shadows on the cleaned line art | Lesson 4 |
| Backgrounds | *bijutsu* | Painted backgrounds | n/a |
| Compositing | *satsuei* | Characters + BG + effects + camera, glow, blur | Lessons 9–10 |
| Editing and sound | *henshuu*, *oto* | Final cut, voice, lip-sync check | Lesson 6 |

The timing of everything is written on an **exposure sheet** (*timesheet* / X-sheet). Lesson 10's demo has one.

---

## 2. The software, by job

> Prices and licensing change often. Check each vendor's site before you buy. Everything marked **free** below is free to use today.

### Drawing and character design (start here)

| Tool | Cost | Why use it |
|---|---|---|
| **Krita** | Free, open source | Strong painting app with frame-by-frame animation, onion skin and a timeline built in. The best free starting point. |
| **Clip Studio Paint** | Paid (PRO / EX) | The de-facto standard for anime and manga illustration. Many Japanese studios now do *genga* and *douga* in it. PRO limits animations to a small number of frames; **EX** removes the limit. 3D posing models help with references. |
| **Procreate** / **Procreate Dreams** | Paid, one-time (iPad) | Great feel for sketching on iPad. Dreams adds an animation timeline. |
| **MediBang Paint** / **FireAlpaca** | Free | Lightweight manga-focused painting. Fine for line art practice. |

### Frame-by-frame 2D animation

| Tool | Cost | Why use it |
|---|---|---|
| **OpenToonz** | Free, open source | Built on the Toonz software used in Japanese production. Real X-sheet workflow, a good fit for learning "the studio way". |
| **Krita** | Free | Simpler than OpenToonz and fine for short clips and practice cycles. |
| **Clip Studio Paint EX** | Paid | Draw and animate in one place, and widely used in Japan. |
| **Toon Boom Harmony** | Paid, subscription | Industry standard for TV series (mostly Western studios, also some anime). Combines frame-by-frame and rigging. |
| **TVPaint** | Paid | A bitmap animation workhorse used by many feature studios. |
| **Blender Grease Pencil** | Free | 2D drawing inside a 3D scene, with camera moves and lighting. |

### Puppet / rigged 2D (fast results, VTubers, games)

| Tool | Cost | Why use it |
|---|---|---|
| **Live2D Cubism** | Free tier + paid Pro | Turns a layered illustration into a deformable puppet. The standard for VTuber models and many mobile games. |
| **Spine** | Paid | Skeletal 2D animation for games. |
| **Moho** | Paid | Bone rigs plus vector drawing for TV-style cut-out animation. |

### 3D anime style ("toon shading")

| Tool | Cost | Why use it |
|---|---|---|
| **Blender** | Free, open source | Model, rig, animate and render with toon shaders (*Shader to RGB* → *Color Ramp (Constant)*), plus Freestyle or inverted-hull outlines. |
| **VRoid Studio** | Free | Anime character creator with sliders for face, hair and outfit. Exports VRM for Blender, Unity and VTubing. |
| **Unity** / **Unreal** | Free tiers | Real-time toon shaders (e.g. Unity Toon Shader, lilToon) for games and VTuber apps. |

### Compositing and editing

| Tool | Cost | Why use it |
|---|---|---|
| **DaVinci Resolve** (with Fusion) | Free + paid Studio | Edit, color and composite. The free version is plenty to start. |
| **Adobe After Effects** | Paid, subscription | What most anime *satsuei* (compositing) departments use: glows, camera, particles. |
| **Blender compositor** | Free | Node compositing if you're already in Blender. |

### Hardware

A **pen tablet** is the one purchase that matters (Wacom, XP-Pen, Huion; a screenless tablet is fine to start). An iPad with Apple Pencil works with Procreate and Clip Studio.

---

## 3. Recommended learning paths

Pick **one** path and stick with it for the first three months.

**A. Totally free (recommended to start)**
1. **Krita** for character design, line art and coloring (Lessons 2–4)
2. **Krita** or **OpenToonz** for frame-by-frame animation (Lessons 5–9)
3. **Blender** later, for toon-shaded 3D and Grease Pencil
4. **DaVinci Resolve** to edit and add sound

**B. "Japanese studio" style**
1. **Clip Studio Paint EX** for design, *genga*, *douga* and coloring
2. **OpenToonz** to learn X-sheets
3. **After Effects** or **Resolve/Fusion** for compositing

**C. VTuber / puppet**
1. **Krita** or **Clip Studio** to draw a layered character (separate eyes, mouth, hair strands, etc.)
2. **Live2D Cubism** to rig the puppet (blinks, lip-sync, hair physics: Lessons 6–7)

**D. 3D anime**
1. **VRoid Studio** for a character in an afternoon
2. **Blender** for toon shading, rigging and animation (Lessons 4, 8)

---

## 4. Where this repo's code fits in

The code in `src/` re-implements, in about 1,800 readable lines, the core ideas these programs are built on. You learn the concept by playing with the demo and reading the code, then use the real feature in your app.

| Concept (file) | Krita / CSP / OpenToonz | Blender | Live2D / Spine |
|---|---|---|---|
| Easing curves (`core/easing.js`) | Tween curves, spacing | Graph Editor F-curves | Curve editor |
| Keyframes / holds (`core/keyframes.js`) | Timeline keys, frame holds | Keyframes, *Constant* interpolation | Timeline keys |
| On twos (`stepTime`) | Hold each drawing 2 frames | *Stepped Interpolation* modifier | Frame rate / stepped keys |
| Onion skin (`core/onionSkin.js`) | Onion skin panel | Grease Pencil onion skin | n/a |
| Cel shading (`effects/celShading.js`) | Shadow layer, Multiply | Shader to RGB + Color Ramp | Multiply color |
| Springs (`physics/spring.js`) | Hand-drawn follow-through | Cloth, spring bones | **Physics** settings |
| FK rig (`rig/skeleton.js`) | Transform masks (Krita) | Armatures | Deformers, bones |
| Lip flap (`character/mouth.js`) | Mouth layer swap | Shape keys / image swap | Lip-sync parameter |

## Checklist

- [ ] I picked a learning path and installed its first tool.
- [ ] I can name the steps genga → douga → shiage → satsuei.
- [ ] I opened `index.html` and ran the demos (see the README).
