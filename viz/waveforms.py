"""Waveform overlay styles.

Live ffmpeg filters (showwaves/showfreqs output transparent bg):
  neon      - dual-color centered-line waveform (cyan/magenta)
  mirror    - symmetric mirrored line waveform (purple)
  triband   - three stacked waveforms: lows / mids / highs
  spectrum  - log frequency spectrum curve (showfreqs)

Numpy pre-renders (true-alpha .mov via qtrle, overlaid directly):
  bars      - spectrum-analyzer bars with peak caps
  glow      - neon line with a soft halo
  radial    - frequency bars radiating in a circle
  particles - grid of dots pulsing with band energy (per-band AGC)
  waterfall - scrolling log-frequency spectrogram

WAVEFORMS lists the styles; wave_mode(style) says how each is consumed.
"""
import hashlib
import os
import shutil
import subprocess

import numpy as np

WAVEFORMS = ['neon', 'glow', 'bars', 'spectrum', 'waterfall',
             'mirror', 'triband', 'radial', 'particles']
VIDEO_STYLES = {'bars', 'glow', 'radial', 'particles', 'waterfall'}
W, H = 980, 300
FPS = 30

# Waveform color themes: each interpolates c1 (low energy) -> c2 (high).
WAVE_THEMES = {
    'volt': {'label': 'Volt (cyan / magenta)',
             'c1': (0.00, 0.90, 1.00), 'c2': (1.00, 0.20, 0.70)},
    'ultraviolet': {'label': 'Ultraviolet (violet / purple)',
                    'c1': (0.45, 0.30, 1.00), 'c2': (0.80, 0.20, 1.00)},
    'lime': {'label': 'Lime (green / cyan)',
             'c1': (0.55, 1.00, 0.00), 'c2': (0.00, 0.90, 1.00)},
    'sunset': {'label': 'Sunset (orange / pink)',
               'c1': (1.00, 0.55, 0.10), 'c2': (1.00, 0.15, 0.45)},
    'mono': {'label': 'Mono (white / gray)',
             'c1': (1.00, 1.00, 1.00), 'c2': (0.55, 0.55, 0.60)},
    'ice': {'label': 'Ice (light blue / deep blue)',
            'c1': (0.65, 0.90, 1.00), 'c2': (0.10, 0.35, 0.90)},
}


def _theme(name):
    return WAVE_THEMES.get(name, WAVE_THEMES['volt'])


def _rgb_hex(c):
    r, g, b = (max(0, min(1, x)) for x in c)
    return '0x%02X%02X%02X' % (int(r * 255), int(g * 255), int(b * 255))


def _theme_hexes(name):
    t = _theme(name)
    return _rgb_hex(t['c1']), _rgb_hex(t['c2'])


def _mid_hex(name):
    t = _theme(name)
    mid = tuple((a + b) / 2 for a, b in zip(t['c1'], t['c2']))
    return _rgb_hex(mid)


def wave_mode(style):
    if style in VIDEO_STYLES:
        return 'video'
    if style in WAVEFORMS:
        return 'filter'
    raise ValueError(f'unknown waveform style: {style!r}')


def waveform_filter(style, theme='volt', alabel='[1:a]'):
    """Return a filter snippet turning audio label into rgba video [wvn]."""
    c1, c2 = _theme_hexes(theme)
    if style == 'neon':
        return (f"{alabel}showwaves=s={W}x{H}:mode=cline:"
                f"colors={c1}|{c2}:scale=sqrt:rate={FPS}:draw=full,"
                f"format=rgba[wvn]")
    if style == 'mirror':
        # true mirror: upper half of the waveform + its vertical flip
        return (f"{alabel}showwaves=s={W}x{H}:mode=line:"
                f"colors={_mid_hex(theme)}:scale=sqrt:rate={FPS}:draw=full,"
                f"format=rgba,crop={W}:{H // 2}:0:0,split[tf1][tf2];"
                f"[tf2]vflip[bf];"
                f"[tf1][bf]vstack[wvn]")
    if style == 'triband':
        # lows / mids / highs as three stacked waveforms
        return (f"{alabel}asplit=3[lo][md][hi];"
                f"[lo]lowpass=f=400,showwaves=s={W}x100:mode=line:"
                f"colors={c1}:scale=sqrt:rate={FPS}:draw=full,"
                f"format=rgba[l];"
                f"[md]highpass=f=400,lowpass=f=3000,"
                f"showwaves=s={W}x100:mode=line:"
                f"colors={_mid_hex(theme)}:scale=sqrt:rate={FPS}:draw=full,"
                f"format=rgba[m];"
                f"[hi]highpass=f=3000,showwaves=s={W}x100:mode=line:"
                f"colors={c2}:scale=sqrt:rate={FPS}:draw=full,"
                f"format=rgba[h];"
                f"[l][m][h]vstack=inputs=3[wvn]")
    if style == 'spectrum':
        return (f"{alabel}showfreqs=s={W}x{H}:mode=line:fscale=log:"
                f"colors={c1}|{c2}:rate={FPS},"
                f"format=rgba[wvn]")
    raise ValueError(f'style {style!r} is not a live filter')


# ---------------- shared numpy helpers ----------------

def _decode_mono(audio_path, sr=22050):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', audio_path,
           '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-acodec', 'pcm_f32le',
           '-']
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-500:])
    return np.frombuffer(p.stdout, dtype=np.float32), sr


def _band_energies(samples, sr, n_bars=64, f_lo=80, f_hi=8000):
    win, hop = 2048, sr // FPS
    n_frames = max(1, int(len(samples) / hop))
    edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n_bars + 1)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    band_of = np.digitize(freqs, edges) - 1
    out = np.zeros((n_frames, n_bars), np.float32)
    window = np.hanning(win)
    for i in range(n_frames):
        c = i * hop + hop // 2
        s0, s1 = max(0, c - win // 2), c + win // 2
        seg = np.zeros(win, np.float32)
        a, b = max(0, s0), min(len(samples), s1)
        seg[a - s0:b - s0] = samples[a:b]
        mag = np.abs(np.fft.rfft(seg * window))
        for b_ in range(n_bars):
            m = band_of == b_
            if m.any():
                out[i, b_] = mag[m].mean()
    return out


def _smooth(v):
    """Fast attack, slow release, gamma-lifted to 0..1."""
    p99 = np.percentile(v, 99)
    v = np.clip(v / max(p99, 1e-6), 0, 1) ** 0.65
    sm = np.zeros_like(v)
    for i in range(v.shape[0]):
        prev = sm[i - 1] if i else 0
        sm[i] = np.maximum(v[i], prev * 0.88)
    return sm


def _encode_alpha_mov(frame_iter, w, h, path):
    """Pipe RGBA frames into a true-alpha .mov (qtrle)."""
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}',
           '-r', str(FPS), '-i', '-',
           '-c:v', 'qtrle', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frame_iter:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('qtrle alpha encode failed')


def _band_colors(n, theme='volt'):
    t = _theme(theme)
    u = np.linspace(0, 1, n)[:, None]
    c1 = np.array(t['c1'])[None, :]
    c2 = np.array(t['c2'])[None, :]
    return (1 - u) * c1 + u * c2  # c1 -> c2


# ---------------- bars ----------------

def _render_bars(energies, theme='volt'):
    n_frames, n_bars = energies.shape
    sm = _smooth(energies)
    peak = np.zeros(n_bars, np.float32)
    gap, bw = 3, W // n_bars
    xs = np.arange(n_bars) * bw
    cols = _band_colors(n_bars, theme)
    yy = np.arange(H)[:, None]
    for i in range(n_frames):
        img = np.zeros((H, W, 4), np.float32)
        vals = sm[i]
        peak = np.maximum(vals, peak * 0.94)
        for b_ in range(n_bars):
            x0 = xs[b_] + gap // 2
            x1 = xs[b_] + bw - gap // 2
            bh = int(vals[b_] * (H - 14))
            if bh > 0:
                y0 = H - bh
                grad = (0.35 + 0.65 * (yy[y0:H, 0:1] / H))[..., None]
                img[y0:H, x0:x1, :3] = cols[b_][None, None, :] * grad
                img[y0:H, x0:x1, 3] = 1.0
            py = H - int(peak[b_] * (H - 14)) - 2
            if 0 <= py < H:
                img[py:py + 2, x0:x1, :3] = 1.0
                img[py:py + 2, x0:x1, 3] = 1.0
        yield (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------- glow (neon line + halo) ----------------

def _render_glow(samples, sr, theme='volt'):
    hop = sr // FPS
    n_frames = max(1, int(len(samples) / hop))
    xs = np.linspace(0, hop - 1, W)
    yy = np.arange(H)[:, None]
    t = _theme(theme)
    c1, c2 = np.array(t['c1']), np.array(t['c2'])
    # strokes fade from the theme's low color (outer halo) to near-white
    white = np.array((1.0, 1.0, 1.0))
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    base = [tuple((1 - u) * c1 + u * c2) for u in stops[:-1]] + [tuple(white)]
    base[-2] = tuple(0.5 * np.array(base[-2]) + 0.5 * white)
    strokes = [  # (half-thickness, alpha, rgb)
        (9, 0.16, base[0]),
        (6, 0.26, base[1]),
        (4, 0.45, base[2]),
        (2, 0.85, base[3]),
        (1, 1.00, base[4]),
    ]
    prev = None
    for i in range(n_frames):
        c = i * hop + hop // 2
        a, b = max(0, c - hop // 2), min(len(samples), c + hop // 2)
        seg = np.zeros(hop, np.float32)
        seg[:b - a] = samples[a:b]
        line = np.interp(xs, np.arange(hop), seg)
        # per-frame normalize so quiet speech still fills the box
        peak = np.abs(line).max()
        if peak > 1e-4:
            line = line / peak * 0.92
        if prev is not None:  # slight motion smoothing
            line = 0.7 * line + 0.3 * prev
        prev = line
        yv = H / 2 - np.clip(line, -1, 1) * (H / 2 - 12)
        img = np.zeros((H, W, 4), np.float32)
        for half, alpha, col in strokes:
            m = np.abs(yy - yv[None, :]) <= half
            img[m, :3] = img[m, :3] * (1 - alpha) + np.array(col) * alpha
            img[m, 3] = np.maximum(img[m, 3], alpha)
        yield (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------- radial ----------------

RW = RH = 440


def _render_radial(energies, theme='volt'):
    n_frames, n_bars = energies.shape
    sm = _smooth(energies)
    cols = _band_colors(n_bars, theme)
    cx, cy = RW / 2, RH / 2
    r0, maxlen = 70, 120
    angles = np.linspace(0, 2 * np.pi, n_bars, endpoint=False)
    # static outer ring mask
    yyg, xxg = np.mgrid[0:RH, 0:RW]
    ring = np.abs(np.sqrt((xxg - cx) ** 2 + (yyg - cy) ** 2)
                  - (r0 + maxlen + 14)) <= 1.5
    mid = tuple((np.array(_theme(theme)['c1']) +
                 np.array(_theme(theme)['c2'])) / 2)
    for i in range(n_frames):
        img = np.zeros((RH, RW, 4), np.float32)
        img[ring, :3] = mid
        img[ring, 3] = 0.35
        vals = sm[i]
        for b_ in range(n_bars):
            ln = int(6 + vals[b_] * maxlen)
            ca, sa = np.cos(angles[b_]), np.sin(angles[b_])
            t = np.arange(ln)
            px = (cx + ca * (r0 + t)).astype(int)
            py = (cy + sa * (r0 + t)).astype(int)
            nx, ny = -sa, ca  # perpendicular
            ok = (px > 1) & (px < RW - 2) & (py > 1) & (py < RH - 2)
            px, py = px[ok], py[ok]
            for off in (-1, 0, 1):
                ox = (nx * off).astype(int)
                oy = (ny * off).astype(int)
                img[py + oy, px + ox, :3] = cols[b_]
                img[py + oy, px + ox, 3] = 1.0
        yield (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------- particles ----------------

def _render_particles(energies, theme='volt'):
    n_frames, n_cols = energies.shape
    sm = _smooth(energies)
    # per-band AGC: speech energy lives in the low bands, so normalize each
    # band against its own peak or the grid looks dead on the right side
    p99 = np.percentile(sm, 99, axis=0)
    floor = max(np.percentile(sm, 99) * 0.08, 1e-6)
    sm = np.clip(sm / np.maximum(p99, floor), 0, 1)
    cols = _band_colors(n_cols, theme)
    rows = 8
    xs = (np.arange(n_cols) + 0.5) * (W / n_cols)
    ys = (np.arange(rows) + 0.5) * (H / rows)
    # precompute dot sprites: 16 size levels per column color
    sprites = []
    for c in range(n_cols):
        lv = []
        for L in range(16):
            r = int(2 + L * 0.55)
            s = 2 * r + 1
            yg, xg = np.mgrid[0:s, 0:s]
            d = np.sqrt((xg - r) ** 2 + (yg - r) ** 2) / max(r, 1)
            a = np.clip(1 - d, 0, 1) ** 1.5 * (0.25 + 0.75 * L / 15)
            sp = np.zeros((s, s, 4), np.float32)
            sp[..., :3] = cols[c]
            sp[..., 3] = a
            lv.append(sp)
        sprites.append(lv)
    for i in range(n_frames):
        img = np.zeros((H, W, 4), np.float32)
        vals = sm[i]
        for c in range(n_cols):
            L = int(np.clip(vals[c], 0, 1) * 15.999)
            sp = sprites[c][L]
            s = sp.shape[0]
            x0, y0 = int(xs[c] - s // 2), None
            for r_ in range(rows):
                yy0 = int(ys[r_] - s // 2)
                xx0 = int(xs[c] - s // 2)
                if xx0 < 0 or yy0 < 0 or xx0 + s > W or yy0 + s > H:
                    continue
                a = sp[..., 3:4]
                img[yy0:yy0 + s, xx0:xx0 + s, :3] = \
                    img[yy0:yy0 + s, xx0:xx0 + s, :3] * (1 - a) \
                    + sp[..., :3] * a
                img[yy0:yy0 + s, xx0:xx0 + s, 3] = np.maximum(
                    img[yy0:yy0 + s, xx0:xx0 + s, 3], sp[..., 3])
        yield (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------- waterfall (scrolling spectrogram) ----------------

def _render_waterfall(energies, theme='volt'):
    sm = _smooth(energies)
    n_frames, n_bars = sm.shape
    hist = 90  # ~3s of history, newest at the right
    t = np.linspace(0, 1, 256)[:, None]
    c1 = np.array(_theme(theme)['c1'])[None, :]
    c2 = np.array(_theme(theme)['c2'])[None, :]
    lut = np.zeros((256, 4), np.float32)
    lut[:, :3] = (1 - t) * c1 + t * c2
    lut[:, 3] = np.clip(t[:, 0] * 1.6, 0, 1)
    rr = (H + n_bars - 1) // n_bars
    cr = (W + hist - 1) // hist
    for i in range(n_frames):
        seg = sm[max(0, i - hist + 1):i + 1]
        if seg.shape[0] < hist:
            pad = np.zeros((hist - seg.shape[0], n_bars), np.float32)
            seg = np.vstack([pad, seg])
        idx = (np.clip(seg, 0, 1) * 255).astype(np.uint8).T[::-1]
        img = lut[idx]  # (n_bars, hist, 4), low freq at bottom
        img = np.repeat(np.repeat(img, rr, axis=0), cr, axis=1)
        yield (np.clip(img[:H, :W], 0, 1) * 255).astype(np.uint8)


def wave_video(style, theme, audio_path, out_path, cache_dir):
    """Pre-render (or fetch from cache) the overlay video for a style."""
    if style not in VIDEO_STYLES:
        raise ValueError(f'style {style!r} is not a pre-rendered video')
    os.makedirs(cache_dir, exist_ok=True)
    with open(audio_path, 'rb') as f:
        key = hashlib.sha1(f.read(1 << 20)).hexdigest()[:10]
    cached = os.path.join(cache_dir, f'{style}-{theme}-{key}.mov')
    if not os.path.exists(cached):
        samples, sr = _decode_mono(audio_path)
        if style == 'bars':
            frames = _render_bars(_band_energies(samples, sr), theme)
            _encode_alpha_mov(frames, W, H, cached)
        elif style == 'glow':
            _encode_alpha_mov(_render_glow(samples, sr, theme), W, H, cached)
        elif style == 'radial':
            frames = _render_radial(_band_energies(samples, sr), theme)
            _encode_alpha_mov(frames, RW, RH, cached)
        elif style == 'particles':
            frames = _render_particles(
                _band_energies(samples, sr, n_bars=44), theme)
            _encode_alpha_mov(frames, W, H, cached)
        elif style == 'waterfall':
            frames = _render_waterfall(_band_energies(samples, sr), theme)
            _encode_alpha_mov(frames, W, H, cached)
    if os.path.abspath(cached) != os.path.abspath(out_path):
        shutil.copyfile(cached, out_path)
    return out_path


# backwards compat alias
def bars_video(audio_path, out_path, cache_dir):
    return wave_video('bars', 'volt', audio_path, out_path, cache_dir)
