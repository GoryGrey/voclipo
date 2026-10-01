"""Animated background loops (seamless) for the visualizer.

Styles:
  nebula    - blurred episode cover, slowly breathing + hue-shifting
  aurora    - flowing purple/magenta/cyan plasma (procedural, cover-free)
  starfield - drifting twinkling stars (procedural, cover-free)
  custom    - blurred custom image with a slow zoom (no hue shift)

All loops are seamless so ffmpeg can -stream_loop them under any duration.
Rendered at 540x960 and upscaled in the final composite to keep builds fast.
"""
import hashlib
import os
import subprocess

import numpy as np

W, H, FPS, LOOP_SECS = 540, 960, 30, 16
N = FPS * LOOP_SECS

BACKGROUNDS = ['nebula', 'aurora', 'starfield', 'bokeh', 'matrix',
               'grid', 'contour', 'dunes', 'custom']


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])


def _write_mp4(frame_iter, path):
    """Stream frames (any iterable of HxWx3 uint8 arrays) into x264."""
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frame_iter:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('ffmpeg pipe failed')


def _palette(t):
    t = np.clip(t, 0, 1)
    c1 = np.array([8, 2, 24])
    c2 = np.array([90, 10, 140])
    c3 = np.array([200, 40, 180])
    c4 = np.array([60, 220, 255])
    out = np.zeros(t.shape + (3,))
    m = t < 0.4
    out[m] = c1 + (c2 - c1) * (t[m] / 0.4)[..., None]
    m = (t >= 0.4) & (t < 0.7)
    out[m] = c2 + (c3 - c2) * ((t[m] - 0.4) / 0.3)[..., None]
    m = t >= 0.7
    out[m] = c3 + (c4 - c3) * ((t[m] - 0.7) / 0.3)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _aurora(path):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn, yn = xx / W * 4, yy / H * 4

    def gen():
        for f in range(N):
            t = 2 * np.pi * f / N
            field = (np.sin(xn * 1.7 + np.cos(t * 2) * 1.3
                             + np.sin(yn * 1.1 + t))
                     + np.sin(yn * 2.3 - t * 1.7 + np.sin(xn * 0.9 - t * 0.6))
                     + np.sin((xn + yn) * 1.2 + np.cos(t * 1.3) * 2.0)) / 6 + 0.5
            v = np.clip(1 - 0.55 * np.abs(yn - 2) / 2, 0.4, 1)
            yield _palette(field * v)
    _write_mp4(gen(), path)


def _starfield(path):
    rng = np.random.default_rng(7)
    nstars = 260
    sx = rng.random(nstars) * W
    sy = rng.random(nstars) * H
    spd = rng.uniform(15, 60, nstars)   # px/sec upward
    sz = rng.uniform(0.6, 2.2, nstars)
    ph = rng.random(nstars) * 2 * np.pi
    base = np.zeros((H, W, 3), np.float32)
    base[..., 0], base[..., 1], base[..., 2] = 10, 3, 28
    grad = np.linspace(0, 25, H)[:, None, None]
    base = base + grad * np.array([0.4, 0.1, 0.9])

    def gen():
        for f in range(N):
            t = f / FPS
            # triangle-wave drift: seamless over the loop (up then back down)
            cyc = t / LOOP_SECS
            drift = spd * LOOP_SECS * (1 - abs(1 - 2 * (cyc % 1))) / 2
            img = base.copy()
            y = (sy - drift) % H
            tw = 0.55 + 0.45 * np.sin(2 * np.pi * 3 * cyc + ph)
            for k in range(nstars):
                r = int(sz[k] * 2)
                x0, y0 = int(sx[k]), int(y[k])
                b = tw[k] * 220
                tint = np.array([b * 0.75, b * 0.85, b])
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        px, py = x0 + dx, y0 + dy
                        if 0 <= px < W and 0 <= py < H:
                            fall = max(0, 1 - (dx * dx + dy * dy)
                                       / (r * r + 1))
                            if fall > 0:
                                img[py, px] = np.maximum(img[py, px],
                                                         tint * fall)
            yield np.clip(img, 0, 255).astype(np.uint8)
    _write_mp4(gen(), path)


def _bokeh(path):
    rng = np.random.default_rng(21)
    n = 36
    x0 = rng.random(n) * W
    y0 = rng.random(n) * H
    rad = rng.uniform(18, 80, n)
    ax = rng.uniform(10, 60, n)
    ay = rng.uniform(10, 60, n)
    ph = rng.random(n) * 2 * np.pi
    spd = rng.uniform(0.5, 2.0, n)
    cols = np.array([[180, 60, 220], [60, 220, 255], [255, 60, 170],
                     [120, 90, 255]], np.float32)[rng.integers(0, 4, n)]
    yg, xg = np.mgrid[0:H, 0:W].astype(np.float32)

    def gen():
        for f in range(N):
            t = 2 * np.pi * f / N
            img = np.zeros((H, W, 3), np.float32)
            img[..., 0], img[..., 1], img[..., 2] = 8, 2, 20
            for k in range(n):
                cx = x0[k] + ax[k] * np.sin(t * spd[k] + ph[k])
                cy = y0[k] + ay[k] * np.cos(t * spd[k] * 0.8 + ph[k])
                d = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2) / rad[k]
                glow = np.clip(1 - d, 0, 1) ** 2 * 0.55
                img += glow[..., None] * cols[k]
            yield np.clip(img, 0, 255).astype(np.uint8)
    _write_mp4(gen(), path)


def _matrix(path):
    rng = np.random.default_rng(33)
    col_w = 12
    cols = W // col_w
    spd = rng.integers(1, 3, cols)          # wraps per loop (seamless)
    ph = rng.random(cols)
    trail = 130

    def gen():
        for f in range(N):
            cyc = f / N
            img = np.zeros((H, W, 3), np.float32)
            img[..., 1] = 4  # faint green-black base
            for c in range(cols):
                head = ((cyc * spd[c] + ph[c]) % 1.0) * H
                x0 = c * col_w + 2
                x1 = x0 + col_w - 4
                for s_ in range(trail):
                    y = int(head - s_) % H
                    fade = max(0.0, 1 - s_ / trail) ** 1.6
                    if s_ < 3:
                        col = (0.75 * fade, 1.0 * fade, 0.85 * fade)
                    else:
                        col = (0.15 * fade, 0.85 * fade, 0.45 * fade)
                    img[y, x0:x1, 0] += col[0] * 255 * 0.5
                    img[y, x0:x1, 1] += col[1] * 255 * 0.5
                    img[y, x0:x1, 2] += col[2] * 255 * 0.5
            yield np.clip(img, 0, 255).astype(np.uint8)
    _write_mp4(gen(), path)


def _grid(path):
    horizon = int(H * 0.42)
    yg, xg = np.mgrid[0:H, 0:W].astype(np.float32)
    sky = np.zeros((H, W, 3), np.float32)
    sky[..., 0] = 6 + 10 * (1 - yg / H)
    sky[..., 1] = 2
    sky[..., 2] = 18 + 22 * (1 - yg / H)
    # sun: striped synthwave sun
    sun_cx, sun_cy, sun_r = W / 2, horizon - 40, 130
    sun_d = np.sqrt((xg - sun_cx) ** 2 + (yg - sun_cy) ** 2)
    sun = np.clip(1 - sun_d / sun_r, 0, 1)
    stripes = ((yg - (sun_cy - sun_r)) / 16 % 1) < 0.55
    sun_col = np.zeros((H, W, 3), np.float32)
    sun_col[..., 0] = 255 * sun
    sun_col[..., 1] = 90 * sun
    sun_col[..., 2] = 160 * sun
    sun_col[stripes & (yg > sun_cy)] = 0
    n_h, n_v = 14, 18

    def gen():
        for f in range(N):
            cyc = f / N
            img = sky.copy()
            img = np.maximum(img, sun_col)
            img[horizon:horizon + 2] = (255, 45, 170)
            # horizontal lines rushing toward viewer
            for k in range(n_h):
                z = (k / n_h + cyc) % 1.0
                y = int(horizon + (H - horizon) * z ** 2.4)
                wdt = 1 + int(3 * z)
                img[max(horizon, y - wdt):y + 1, :, 0] = \
                    np.maximum(img[max(horizon, y - wdt):y + 1, :, 0],
                               200 * z + 55)
                img[max(horizon, y - wdt):y + 1, :, 2] = \
                    np.maximum(img[max(horizon, y - wdt):y + 1, :, 2],
                               120 * z + 60)
            # verticals converging on the vanishing point
            for k in range(n_v):
                t = (k / (n_v - 1) - 0.5) * 2
                x_bot = int(W / 2 + t * W * 1.1)
                x_top = int(W / 2 + t * 30)
                for s_ in range(40):
                    u = s_ / 39
                    x = int(x_top + (x_bot - x_top) * u ** 1.6)
                    y = min(int(horizon + (H - horizon) * u), H - 1)
                    if 0 <= x < W:
                        img[y, max(0, x - 1):min(W, x + 2), 0] = 190
                        img[y, max(0, x - 1):min(W, x + 2), 2] = 130
            yield np.clip(img, 0, 255).astype(np.uint8)
    _write_mp4(gen(), path)


def _contour(path):
    yg, xg = np.mgrid[0:H, 0:W].astype(np.float32)
    xn, yn = xg / W * 6, yg / H * 6
    levels = np.linspace(-1.6, 1.6, 9)

    def gen():
        for f in range(N):
            t = 2 * np.pi * f / N
            field = (np.sin(xn * 1.3 + t) + np.sin(yn * 1.7 - t * 1.4)
                     + np.sin((xn + yn) * 0.8 + t * 0.6))
            img = np.zeros((H, W, 3), np.float32)
            img[..., 0], img[..., 1], img[..., 2] = 7, 3, 16
            for li, lv in enumerate(levels):
                m = np.abs(field - lv) < 0.045
                if li == len(levels) // 2:
                    img[m] = (80, 230, 255)
                else:
                    img[m] = (150, 90, 230)
            yield np.clip(img, 0, 255).astype(np.uint8)
    _write_mp4(gen(), path)


def _dunes(path):
    yg, xg = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = np.arange(W, dtype=np.float32) / W
    layers = [  # (base_y, amp, cycles_over_loop, color)
        (0.30, 0.10, 1, (24, 8, 60)),
        (0.42, 0.12, 2, (48, 12, 100)),
        (0.55, 0.10, 1, (86, 22, 140)),
        (0.68, 0.12, 3, (140, 40, 180)),
        (0.80, 0.10, 2, (60, 200, 230)),
    ]

    def gen():
        for f in range(N):
            cyc = f / N
            img = np.zeros((H, W, 3), np.float32)
            img[..., 0], img[..., 1], img[..., 2] = 8, 3, 22
            for bi, (base, amp, cycs, col) in enumerate(layers):
                curve = (base + amp * np.sin(2 * np.pi * (xn * 2
                                          + cycs * cyc + bi * 0.35))) * H
                m = yg >= curve[None, :]
                depth = np.clip((yg - curve[None, :]) / (H * 0.25), 0, 1)
                shade = (0.55 + 0.45 * depth)[..., None]
                img[m] = np.array(col, np.float32) * shade[m]
            yield np.clip(img, 0, 255).astype(np.uint8)
    _write_mp4(gen(), path)


def _image_bg(image_path, path, hue_shift):
    hue = (f"hue=h='45*sin(2*PI*t/{LOOP_SECS})'," if hue_shift else "")
    # single still input -> zoompan d=N emits exactly N frames (16s @ 30fps)
    _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
          '-i', image_path,
          '-vf', f"scale={W}:{H * 2}:force_original_aspect_ratio=increase,"
                  f"crop={W}:{H},"
                  f"gblur=sigma=40,eq=brightness=-0.12:saturation=1.4,"
                  f"{hue}"
                  f"zoompan=z='1.08+0.02*sin(2*PI*on/{N})':"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                  f"d={N}:s={W}x{H}:fps={FPS}",
          '-frames:v', str(N),
          '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path])


def _cache_key(style, image_path):
    h = hashlib.sha1(style.encode()).hexdigest()[:8]
    if image_path:
        with open(image_path, 'rb') as f:
            h += hashlib.sha1(f.read(1 << 20)).hexdigest()[:8]
    return f"{style}-{h}.mp4"


def build_background(style, out_path, cache_dir, cover_path=None,
                     custom_path=None):
    """Render (or fetch from cache) the background loop for a style."""
    if style not in BACKGROUNDS:
        raise ValueError(f'unknown background style: {style}')
    os.makedirs(cache_dir, exist_ok=True)
    img = {'nebula': cover_path, 'custom': custom_path}.get(style)
    if style in ('nebula', 'custom') and not img:
        raise ValueError(f'background style {style!r} needs an image')
    cached = os.path.join(cache_dir, _cache_key(style, img))
    if not os.path.exists(cached):
        if style == 'aurora':
            _aurora(cached)
        elif style == 'starfield':
            _starfield(cached)
        elif style == 'bokeh':
            _bokeh(cached)
        elif style == 'matrix':
            _matrix(cached)
        elif style == 'grid':
            _grid(cached)
        elif style == 'contour':
            _contour(cached)
        elif style == 'dunes':
            _dunes(cached)
        elif style == 'nebula':
            _image_bg(img, cached, hue_shift=True)
        elif style == 'custom':
            _image_bg(img, cached, hue_shift=False)
    # hardlink/copy into the job dir so the render step owns its inputs
    if os.path.abspath(cached) != os.path.abspath(out_path):
        import shutil
        shutil.copyfile(cached, out_path)
    return out_path
