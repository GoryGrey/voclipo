"""Karaoke caption builder: whisper word timestamps + script -> ASS subtitle file.

Captions follow what was ACTUALLY SPOKEN (whisper), preferring the script's
spelling wherever the two align. Script words that were never spoken are
dropped; whisper-only words are kept.

Karaoke color convention (the ASS karaoke sweep paints Primary over Secondary):
  PrimaryColour   = bright highlight -> the SUNG words
  SecondaryColour = plain white      -> the UPCOMING words
Inverting this makes captions feel 1-2s ahead of the voice.
"""
import json
import os
import re
import difflib

# ASS colors are &HAABBGGRR. sung = the word being spoken right now.
CAPTION_PRESETS = {
    'classic': {'label': 'Classic (yellow / white)',
                'font_size': 76,
                'sung': '&H0000F6FF', 'upcoming': '&H00FFFFFF'},
    'neon': {'label': 'Neon (cyan / white)',
             'font_size': 76,
             'sung': '&H00FFFF00', 'upcoming': '&H00FFFFFF'},
    'ember': {'label': 'Ember (orange / white)',
              'font_size': 76,
              'sung': '&H00007AFF', 'upcoming': '&H00FFFFFF'},
    'ghost': {'label': 'Ghost (white / gray)',
              'font_size': 68,
              'sung': '&H00FFFFFF', 'upcoming': '&H00999999'},
    'big': {'label': 'Big classic',
            'font_size': 96,
            'sung': '&H0000F6FF', 'upcoming': '&H00FFFFFF'},
}


def norm(w):
    return re.sub(r'[^a-z0-9%]', '', w.lower())


def ass_time(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def transcribe_words(audio_path, model_name='base', log=print):
    """Run faster-whisper with word timestamps. Returns [(word, start, end)]."""
    # The sandbox proxy env vars break httpx inside faster-whisper; the
    # model is cached locally so we can drop them for this call.
    for var in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'):
        os.environ.pop(var, None)
    from faster_whisper import WhisperModel
    log(f'loading whisper model {model_name}...')
    model = WhisperModel(model_name, device='cpu', compute_type='int8')
    log('transcribing...')
    segments, _ = model.transcribe(audio_path, word_timestamps=True,
                                   vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append((w.word, w.start, w.end))
    log(f'transcribed {len(words)} words')
    return words


def parse_script(script_text):
    """Pull spoken lines out of 'Name: text' script lines."""
    turns = []
    for line in script_text.splitlines():
        m = re.match(r'^[A-Za-z]+:\s*(.*)$', line.strip())
        if m and m.group(1):
            turns.append(m.group(1))
    return turns


def _final_lines(words, script_text, max_chars=30):
    """Align whisper words to the script, merge token artifacts, wrap into
    caption lines. Returns [ [(word, start, end), ...], ... ]."""
    script_words = ' '.join(parse_script(script_text)).split()

    if script_words:
        wn = [norm(w) for w, _, _ in words]
        sn = [norm(w) for w in script_words]
        sm = difflib.SequenceMatcher(None, wn, sn, autojunk=False)
        seq = []  # [display, start, end]
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for k in range(j2 - j1):
                    seq.append([script_words[j1 + k], words[i1 + k][1],
                                words[i1 + k][2]])
            elif tag in ('delete', 'replace'):
                for k in range(i2 - i1):
                    seq.append([words[i1 + k][0].strip(), words[i1 + k][1],
                                words[i1 + k][2]])
            # 'insert': script words never spoken -> dropped
    else:
        seq = [[w.strip(), s, e] for w, s, e in words]

    # merge whisper token artifacts
    merged = []
    i = 0
    while i < len(seq):
        w, s, e = seq[i]
        nxt = seq[i + 1] if i + 1 < len(seq) else [None]
        nxt2 = seq[i + 2] if i + 2 < len(seq) else [None]
        if w.lower() == 'our' and (nxt[0] or '').lower() == 'slash':
            merged.append(['r', s, e])
            i += 1
            continue
        if w == 'Self' and nxt[0] == '-Teaching' and \
                (nxt2[0] or '').startswith('AI'):
            ai = nxt2[0]
            merged.append(['SelfTeachingAI' +
                           (',' if ai.endswith(',') else ''), s, nxt2[2]])
            i += 3
            continue
        if re.fullmatch(r'\d+', w) and re.fullmatch(r'\.\d+%?\.?', nxt[0] or ''):
            merged.append([f"{w}{nxt[0]}", s, nxt[2]])
            i += 2
            continue
        if re.fullmatch(r'\d+', w) and nxt[0] == '%':
            merged.append([f"{w}%", s, nxt[2]])
            i += 2
            continue
        merged.append([w, s, e])
        i += 1

    lines, cur, cur_len = [], [], 0
    for w, s, e in merged:
        if not w:
            continue
        if cur_len + len(w) + 1 > max_chars and cur:
            lines.append(cur)
            cur, cur_len = [], 0
        cur.append((w, s, e))
        cur_len += len(w) + 1
    if cur:
        lines.append(cur)
    return lines


def _srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(words, script_text, out_path, time_offset=0.0):
    """Plain SRT from the same word alignment as the ASS captions. Each
    caption line becomes one cue. time_offset shifts cues (e.g. +2s when an
    intro card is prepended to the video)."""
    lines = _final_lines(words, script_text)
    cues = []
    for i, line in enumerate(lines, 1):
        s0, e0 = line[0][1] + time_offset, line[-1][2] + time_offset
        text = ' '.join(w for w, _, _ in line)
        cues.append(f"{i}\n{_srt_time(s0)} --> {_srt_time(e0)}\n{text}\n")
    with open(out_path, 'w') as f:
        f.write('\n'.join(cues))
    return len(lines)


def build_ass(words, script_text, out_path, preset='classic',
              play_res_x=1080, play_res_y=1920, margin_v=560):
    """words: [(word, start, end)]. script_text may be '' (use whisper words).
    preset: key of CAPTION_PRESETS."""
    cp = CAPTION_PRESETS.get(preset, CAPTION_PRESETS['classic'])
    font_size = cp['font_size']
    sung_color = cp['sung']
    upcoming_color = cp['upcoming']
    lines = _final_lines(words, script_text)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,{font_size},{sung_color},{upcoming_color},&H90000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,1,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for line in lines:
        s0, e0 = line[0][1], line[-1][2]
        # Gap-preserving karaoke: each \k covers the time since the PREVIOUS
        # word ended (including the silence gap), so the sweep highlights
        # each word exactly when it is spoken. Sum of \k = event duration.
        text, prev_end = '', s0
        for w, s, e in line:
            k = max(1, int(round((e - prev_end) * 100)))
            text += '{\\k%d}%s ' % (k, w)
            prev_end = e
        events.append(
            f"Dialogue: 0,{ass_time(s0)},{ass_time(e0)},Cap,,0,0,0,,"
            f"{text.strip()}")
    with open(out_path, 'w') as f:
        f.write(header + '\n'.join(events) + '\n')
    return sum(len(l) for l in lines), len(events)


def parse_custom_words(text):
    """Parse the brand-words textarea: one correct spelling per line."""
    words = []
    for line in (text or '').splitlines():
        w = line.strip().strip(',')
        if w:
            words.append(w)
    # de-dupe, keep order
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out


def apply_custom_words(words, custom_words):
    """Replace Whisper's spelling of brand/product names with the user's.

    words: [(word, start, end)]. Matching is case-insensitive on the word
    core (punctuation stripped); the user's spelling wins, original
    punctuation is preserved. Exact matches always win; near-misses
    (difflib ratio >= 0.85, core >= 5 chars) catch Whisper's creative
    spellings. Returns (new_words, n_replaced).
    """
    if not custom_words:
        return words, 0
    want = [(c.lower(), c) for c in custom_words]
    out, n = [], 0
    for w, s, e in words:
        m = re.match(r"^(\W*)(.*?)(\W*)$", w, re.DOTALL)
        pre, core, post = m.group(1), m.group(2), m.group(3)
        low = core.lower()
        hit = None
        for target_low, target in want:
            if low == target_low:
                hit = target
                break
        if hit is None and len(low) >= 5:
            for target_low, target in want:
                if len(target_low) >= 5 and \
                        difflib.SequenceMatcher(None, low,
                                                target_low).ratio() >= 0.85:
                    hit = target
                    break
        if hit and core != hit:
            w = pre + hit + post
            n += 1
        out.append((w, s, e))
    return out, n
