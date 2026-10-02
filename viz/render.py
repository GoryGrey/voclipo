"""Segment-based visualizer renderer.

Long single-pass x264 encodes get killed on small VMs, so the video is
rendered in ~30s segments (each a couple of minutes of encode), then the
segments are concatenated with -c copy and the original audio is muxed.

Cuts land on caption-event boundaries, never mid-event: an event straddling
a cut would restart its karaoke sweep in the next segment and visibly lag
the voice for a few seconds after every cut.
"""
import os
import re
import subprocess

from .waveforms import waveform_filter, wave_mode

FPS = 30
SEG = 30.0
INTRO_SECS = 2.0
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

# Per-aspect composite geometry.
LAYOUTS = {
    'vertical': {'vw': 1080, 'vh': 1920, 'card': 700, 'card_y': 340,
                 'title_y': (90, 160), 'title_fs': 54, 'wave_bottom': 130,
                 'caption_margin_v': 560},
    'square': {'vw': 1080, 'vh': 1080, 'card': 380, 'card_y': 140,
               'title_y': (24, 68), 'title_fs': 40, 'wave_bottom': 40,
               'caption_margin_v': 360},
}
ASPECTS = {'vertical': 'Vertical 9:16', 'square': 'Square 1:1'}


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _ass_time(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ts(s):
    m = re.match(r'(\d+):(\d+):(\d+)\.(\d+)', s)
    h, mi, sec, cs = map(int, m.groups())
    return h * 3600 + mi * 60 + sec + cs / 100.


def parse_ass(path):
    header, events = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('Dialogue:'):
                parts = line.rstrip('\n').split(',', 9)
                events.append((parts[1], parts[2], parts[9]))
            else:
                header.append(line)
    return ''.join(header), events


def _shift_ass(header, events, start, end, out_path, time_offset=0.0):
    kept = []
    for s, e, text in events:
        s0, e0 = _ts(s) - start, _ts(e) - start
        if e0 <= 0 or s0 >= (end - start):
            continue
        kept.append((max(0, s0) + time_offset, e0 + time_offset, text))
    with open(out_path, 'w') as f:
        f.write(header)
        for s0, e0, text in kept:
            f.write(f"Dialogue: 0,{_ass_time(s0)},{_ass_time(e0)},"
                    f"Cap,,0,0,0,,{text}\n")


def _title_lines(title, max_chars=26):
    words = title.split()
    lines, cur = [], ''
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    lines = lines[:2]
    while len(lines) < 2:
        lines.append(' ')
    return lines


def _esc(t):
    return t.replace('\\', '\\\\').replace("'", "\\'").replace(':', '\\:')


def _duration(path):
    return float(_run(['ffprobe', '-v', 'error', '-show_entries',
                       'format=duration', '-of', 'csv=p=0',
                       path]).stdout.strip())


def _segment_filter(start, dur, ass_seg, title, wave_style, wave_theme,
                    use_wave_video, layout):
    vw, vh = layout['vw'], layout['vh']
    t1, t2 = _title_lines(title)
    y1, y2 = layout['title_y']
    fs = layout['title_fs']
    dt = (f"drawtext=fontfile={FONT}:text='{_esc(t1)}':fontsize={fs}:"
          f"fontcolor=white:borderw=2:bordercolor=black@0.6:"
          f"x=(w-text_w)/2:y={y1},"
          f"drawtext=fontfile={FONT}:text='{_esc(t2)}':fontsize={fs}:"
          f"fontcolor=white:borderw=2:bordercolor=black@0.6:"
          f"x=(w-text_w)/2:y={y2}")
    card = layout['card']
    wb = layout['wave_bottom']
    fc = (
        f"[0:v]scale={vw}:{vh}:force_original_aspect_ratio=increase,"
        f"crop={vw}:{vh},setsar=1[bg];"
        f"[2:v]scale={card}:{card}[card];"
        f"[bg][card]overlay=(W-w)/2:{layout['card_y']}[bgc];"
        f"[bgc]{dt}[titled];"
    )
    if use_wave_video:
        # pre-rendered overlays carry true alpha (qtrle), composite directly
        fc += ("[3:v]format=rgba[bv];"
               f"[titled][bv]overlay=(W-w)/2:H-h-{wb}[vsub];")
    else:
        fc += waveform_filter(wave_style, wave_theme, '[1:a]') + ";"
        fc += (f"[titled][wvn]overlay=(W-w)/2:H-h-{wb}[vsub];")
    fc += f"[vsub]subtitles='{_esc(ass_seg)}'[vout]"
    return fc


def _intro_card(cover, title, layout, out_path, log=print):
    """2s branded title card: dimmed blurred cover + big centered title."""
    vw, vh = layout['vw'], layout['vh']
    t1, t2 = _title_lines(title, max_chars=22)
    fs = int(vw * 0.085)
    dt = (f"drawtext=fontfile={FONT}:text='{_esc(t1)}':fontsize={fs}:"
          f"fontcolor=white:borderw=3:bordercolor=black@0.7:"
          f"x=(w-text_w)/2:y=(h-text_h)/2-70,"
          f"drawtext=fontfile={FONT}:text='{_esc(t2)}':fontsize={fs}:"
          f"fontcolor=white:borderw=3:bordercolor=black@0.7:"
          f"x=(w-text_w)/2:y=(h-text_h)/2+70")
    fc = (f"[0:v]scale={vw}:{vh}:force_original_aspect_ratio=increase,"
          f"crop={vw}:{vh},gblur=sigma=40,eq=brightness=-0.45[bg];"
          f"[bg]{dt},fade=t=in:st=0:d=0.5,fade=t=out:st={INTRO_SECS - 0.5}:d=0.5,"
          f"setsar=1,format=yuv420p[vout]")
    r = _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
              '-loop', '1', '-framerate', str(FPS), '-t', f'{INTRO_SECS:.1f}',
              '-i', cover, '-filter_complex', fc, '-map', '[vout]',
              '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
              '-r', str(FPS), '-t', f'{INTRO_SECS:.1f}', out_path])
    if r.returncode != 0:
        raise RuntimeError('intro card failed:\n' + r.stderr[-2000:])
    dur = _duration(out_path)
    if abs(dur - INTRO_SECS) > 0.15:
        raise RuntimeError(
            f'intro card came out {dur:.2f}s, expected {INTRO_SECS}s')
    log('intro card ok')


def render_episode(job_dir, audio, cover, ass_path, title, bg_video,
                   wave_style, wave_theme='volt', aspect='vertical',
                   intro=True, out_name='episode.mp4',
                   wave_video_path=None, log=print, progress=None):
    """Full pipeline: intro card -> segments -> concat -> mux audio.
    Returns output path."""
    os.makedirs(job_dir, exist_ok=True)
    layout = LAYOUTS.get(aspect, LAYOUTS['vertical'])
    # The intro card is concatenated BEFORE the content segments, so it
    # already pushes every caption +INTRO_SECS into the final video.
    # Shifting the ASS as well would double-shift captions late.
    ass_offset = 0.0
    total = _duration(audio)
    header, events = parse_ass(ass_path)
    bounds = sorted({0.0, total}
                    | {_ts(s) for s, e, _ in events}
                    | {_ts(e) for s, e, _ in events})
    cuts = [0.0]
    k = 1
    while k * SEG < total:
        nominal = k * SEG
        b = min(bounds, key=lambda x: abs(x - nominal))
        if cuts[-1] + 10 < b < total:
            cuts.append(b)
        k += 1
    cuts.append(total)

    use_wave = wave_mode(wave_style) == 'video'
    if use_wave and not wave_video_path:
        raise ValueError(f'{wave_style} style needs a pre-rendered overlay')

    segs = []
    if intro:
        intro_mp4 = os.path.join(job_dir, 'intro.mp4')
        segs.append(intro_mp4)
        if os.path.exists(intro_mp4):
            log('intro card cached')
        else:
            _intro_card(cover, title, layout, intro_mp4, log)
    n = len(cuts) - 1
    for idx, (s, e) in enumerate(zip(cuts[:-1], cuts[1:])):
        dur = e - s
        ass_seg = os.path.join(job_dir, f'seg{idx}.ass')
        out_seg = os.path.join(job_dir, f'seg{idx}.mp4')
        _shift_ass(header, events, s, e, ass_seg, time_offset=ass_offset)
        segs.append(out_seg)
        if os.path.exists(out_seg):
            log(f'segment {idx} cached')
        else:
            fc = _segment_filter(s, dur, ass_seg, title, wave_style,
                                 wave_theme, use_wave, layout)
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                   '-stream_loop', '-1', '-t', f'{dur:.2f}', '-i', bg_video,
                   '-ss', f'{s:.2f}', '-t', f'{dur:.2f}', '-i', audio,
                   '-i', cover]
            if use_wave:
                cmd += ['-ss', f'{s:.2f}', '-t', f'{dur:.2f}',
                        '-i', wave_video_path]
            cmd += ['-filter_complex', fc, '-map', '[vout]',
                    '-an', '-c:v', 'libx264', '-preset', 'veryfast',
                    '-crf', '20', '-r', str(FPS), '-t', f'{dur:.2f}',
                    out_seg]
            r = _run(cmd)
            if r.returncode != 0:
                raise RuntimeError(f'segment {idx} failed:\n'
                                   + r.stderr[-2000:])
            log(f'segment {idx} ok')
        if progress:
            progress((idx + 1) / n * 0.9)

    lst = os.path.join(job_dir, 'segs.txt')
    with open(lst, 'w') as f:
        for p in segs:
            f.write(f"file '{p}'\n")
    tmp = os.path.join(job_dir, 'video-nomux.mp4')
    r = _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
              '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', tmp])
    if r.returncode != 0:
        raise RuntimeError('concat failed:\n' + r.stderr[-2000:])
    final = os.path.join(job_dir, out_name)
    if intro:
        # the final audio starts with INTRO_SECS of silence so it lines up
        # with the episode content after the title card
        ms = int(INTRO_SECS * 1000)
        r = _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                  '-i', tmp, '-i', audio,
                  '-filter_complex',
                  f'[1:a]adelay={ms}|{ms}[aout]',
                  '-map', '0:v', '-map', '[aout]',
                  '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                  '-movflags', '+faststart', final])
    else:
        r = _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                  '-i', tmp, '-i', audio,
                  '-map', '0:v', '-map', '1:a',
                  '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                  '-movflags', '+faststart', '-shortest', final])
    if r.returncode != 0:
        raise RuntimeError('mux failed:\n' + r.stderr[-2000:])
    if progress:
        progress(1.0)
    log(f'done: {final} ({_duration(final):.1f}s)')
    return final
