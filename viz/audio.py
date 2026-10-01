import json
import os
import subprocess


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_duration(src):
    r = _run(['ffprobe', '-v', 'error', '-show_entries',
              'format=duration', '-of', 'csv=p=0', src])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def clean_voice(src, out, log=print):
    """Light voice cleanup: highpass rumble + loudness normalization."""
    r = _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
              '-i', src,
              '-af', 'highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11',
              '-c:a', 'pcm_s16le', '-ar', '44100', out])
    if r.returncode != 0:
        raise RuntimeError('voice cleanup failed:\n' + r.stderr[-2000:])
    log('voice cleanup ok')
    return out


def cut_ranges(src, ranges, out, log=print):
    """Keep only `ranges` [(start, end)] of src audio, concatenated."""
    if len(ranges) == 1 and ranges[0][0] <= 0.01:
        # single range from ~0: just trim the tail with one pass
        s, e = ranges[0]
        r = _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                  '-i', src, '-t', f'{e - s:.2f}',
                  '-c:a', 'pcm_s16le', out])
    else:
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
        for s, e in ranges:
            cmd += ['-ss', f'{s:.2f}', '-t', f'{e - s:.2f}', '-i', src]
        filt = ''.join(f'[{i}:a]' for i in range(len(ranges)))
        filt += f'concat=n={len(ranges)}:v=0:a=1[aout]'
        cmd += ['-filter_complex', filt, '-map', '[aout]',
                '-c:a', 'pcm_s16le', out]
        r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError('dead-air cut failed:\n' + r.stderr[-2000:])
    log(f'dead-air cut ok ({len(ranges)} kept ranges)')
    return out


def beep_words(src, spans, out, freq=1000, log=print):
    """Replace each (start, end) span with a sine beep."""
    if not spans:
        return src
    # mute the original under each span, then mix in delayed beeps
    vols = []
    beeps = []
    for i, (s, e) in enumerate(spans):
        vols.append(f"volume=enable='between(t,{s:.2f},{e:.2f})':volume=0")
        beeps.append(
            f"sine=frequency={freq}:duration={e - s:.2f}:"
            f"sample_rate=44100,adelay={int(s * 1000)}|{int(s * 1000)}"
            f"[b{i}]")
    filt = (f"[0:a]{','.join(vols)}[muted];"
            + ';'.join(beeps) + ';'
            + '[muted]' + ''.join(f'[b{i}]' for i in range(len(spans)))
            + f"amix=inputs={len(spans) + 1}:normalize=0[aout]")
    r = _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
              '-i', src, '-filter_complex', filt, '-map', '[aout]',
              '-c:a', 'pcm_s16le', out])
    if r.returncode != 0:
        raise RuntimeError('clean-mode beep failed:\n' + r.stderr[-2000:])
    log(f'clean mode: beeped {len(spans)} word(s)')
    return out
