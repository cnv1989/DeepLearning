// Playback clock for browser demos. Handles fps, "on N's" stepping, looping.

import { stepTime } from './keyframes.js';

export class Player {
  /**
   * @param {(t:number, info:{frame:number, rawT:number})=>void} render
   * @param {{fps?:number, onN?:number, duration?:number, loop?:boolean}} opts
   */
  constructor(render, { fps = 24, onN = 1, duration = 2, loop = true } = {}) {
    this.render = render;
    this.fps = fps;
    this.onN = onN;
    this.duration = duration;
    this.loop = loop;
    this.time = 0;
    this.playing = false;
    this.speed = 1;
    this._last = 0;
    this._raf = 0;
    this._tick = this._tick.bind(this);
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    this._last = performance.now();
    this._raf = requestAnimationFrame(this._tick);
  }

  pause() {
    this.playing = false;
    cancelAnimationFrame(this._raf);
  }

  toggle() {
    this.playing ? this.pause() : this.play();
  }

  seek(t) {
    this.time = Math.max(0, Math.min(t, this.duration));
    this.draw();
  }

  stepFrames(n) {
    this.pause();
    this.seek(stepTime(this.time, this.fps, 1) + n / this.fps);
  }

  draw() {
    const t = stepTime(this.time, this.fps, this.onN);
    this.render(t, { frame: Math.round(t * this.fps), rawT: this.time });
  }

  _tick(now) {
    if (!this.playing) return;
    const dt = Math.min(0.1, (now - this._last) / 1000) * this.speed;
    this._last = now;
    this.time += dt;
    if (this.time > this.duration) {
      if (this.loop) this.time %= this.duration;
      else { this.time = this.duration; this.pause(); }
    }
    this.draw();
    this._raf = requestAnimationFrame(this._tick);
  }
}
