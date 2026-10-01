"""Clip intelligence for Voclipo: sentences, heuristic scorecards, quote cards.

Everything here is heuristic and local. The scorecard explains structural
signals (hook speed, questions, quotable length) - it is NOT a virality
prediction.
"""
import os
import re

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# compact profanity list for Clean Mode beeps
PROFANITY = {
    'fuck', 'fucking', 'fucked', 'fucker', 'shit', 'shitty', 'bitch',
    'bitches', 'ass', 'asses', 'asshole', 'dick', 'dicks', 'piss', 'pissed',
    'damn', 'hell', 'crap', 'bastard', 'slut', 'whore', 'douche', 'cock',
    'tits', 'titty', 'boobs', 'nigga', 'nigger', 'faggot', 'retard',
    'cunt', 'motherfucker',
}

EMOTIONAL = {
    'love', 'hate', 'crazy', 'insane', 'never', 'always', 'secret',
    'truth', 'lie', 'lies', 'scared', 'fear', 'angry', 'amazing',
    'terrible', 'worst', 'best', 'nobody', 'everyone', 'everybody',
    'destroy', 'killed', 'dead', 'alive', 'free', 'money', 'rich',
    'broke', 'fail', 'failed', 'win', 'won', 'quit',
}

CONTRAST = {'but', 'however', 'although', 'nobody', 'secret', 'actually',
            'really', 'honest', 'truth'}


def _clean_word(w):
    return re.sub(r"[^a-z0-9']", '', w.lower())


def split_sentences(words):
    """words: [(word, start, end)] -> [ [(word, start, end), ...], ... ]."""
    sents, cur = [], []
    for w in words:
        cur.append(w)
        t = w[0].strip()
        if t and t[-1] in '.!?' and len(cur) >= 3:
            sents.append(cur)
            cur = []
        elif len(cur) >= 28:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return [s for s in sents if s]


def sentence_text(sent):
    return ' '.join(w for w, _, _ in sent).strip()


def score_sentence(sent, idx, n_sents):
    """Heuristic 0-100. Rewards quotable length, questions, emotion."""
    text = sentence_text(sent)
    n = len(sent)
    score, reasons = 0, []
    if 8 <= n <= 20:
        score += 30
        reasons.append('quotable length')
    elif 5 <= n <= 30:
        score += 15
    if text.endswith('?'):
        score += 20
        reasons.append('asks a question')
    elif text.endswith('!'):
        score += 15
        reasons.append('high energy')
    emo = sum(1 for w, _, _ in sent if _clean_word(w) in EMOTIONAL)
    if emo:
        score += min(10, emo * 5)
        reasons.append('emotional language')
    con = sum(1 for w, _, _ in sent if _clean_word(w) in CONTRAST)
    if con:
        score += min(10, con * 5)
    if any(_clean_word(w) == 'you' for w, _, _ in sent):
        score += 5
        reasons.append('talks to the viewer')
    if n_sents and idx / n_sents < 0.2:
        score += 10
        reasons.append('early in the episode')
    return min(100, score), reasons


def clip_scorecard(words, total_dur):
    """Heuristic signals about the episode's clip potential."""
    sents = split_sentences(words)
    scored = []
    for i, s in enumerate(sents):
        sc, reasons = score_sentence(s, i, len(sents))
        scored.append((sc, s, reasons))
    scored.sort(key=lambda x: -x[0])
    top = scored[:3]

    hook_words = sum(1 for _, s, _ in words if s < 3.0)
    questions = sum(1 for s in sents if sentence_text(s).endswith('?'))
    gaps = [s2 - e1 for (_, _, e1), (_, s2, _)
            in zip(words, words[1:]) if s2 - e1 > 0]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    long_pauses = sum(1 for g in gaps if g > 1.5)

    signals = []
    if hook_words >= 7:
        signals.append(f'Fast hook: {hook_words} words in the first 3 '
                       f'seconds')
    elif hook_words <= 3:
        signals.append(f'Slow start: only {hook_words} words in the first '
                       f'3 seconds, consider trimming the opening')
    else:
        signals.append(f'Decent hook: {hook_words} words in the first 3 '
                       f'seconds')
    if questions:
        signals.append(f'{questions} question(s) asked, good comment bait')
    if top and top[0][0] >= 50:
        signals.append(f'Strong quotable moment: '
                       f'"{sentence_text(top[0][1])[:70]}..."')
    if long_pauses:
        signals.append(f'{long_pauses} long pause(s) over 1.5s, '
                       f'Dead-Air Killer would tighten this')
    if avg_gap > 0.45:
        signals.append(f'Leisurely pacing (avg gap {avg_gap:.2f}s between '
                       f'words)')
    if total_dur > 90:
        signals.append(f'{total_dur:.0f}s runtime gives plenty of clip '
                       f'material')
    return {
        'signals': signals,
        'hook_words_3s': hook_words,
        'questions': questions,
        'long_pauses': long_pauses,
        'top_quotes': [{'text': sentence_text(s), 'score': sc,
                        'reasons': reasons} for sc, s, reasons in top],
        'n_sentences': len(sents),
        'n_words': len(words),
    }


def dead_air_cuts(words, total_dur, max_gap=1.2, keep_gap=0.4,
                  max_lead=3.0, max_tail=2.0):
    """Compute (kept_ranges, shifted_words).

    Trims leading/trailing silence and compresses internal gaps longer than
    max_gap down to keep_gap. Word timestamps are remapped so captions stay
    honest.
    """
    if not words:
        return [(0.0, total_dur)], words
    lead = min(words[0][1], max_lead)
    tail = max(0.0, total_dur - words[-1][2])
    tail = min(tail, max_tail)

    # cut points: (cut_start, cut_end) in original time
    cuts = []
    if lead > 0.05:
        cuts.append((0.0, lead))
    prev_end = words[0][2]
    for _, s, e in words[1:]:
        if s - prev_end > max_gap:
            cuts.append((prev_end + keep_gap, s - (keep_gap / 2)))
        prev_end = max(prev_end, e)
    end_content = total_dur - tail
    if tail > 0.05:
        cuts.append((end_content, total_dur))

    kept = []
    pos = 0.0
    for cs, ce in sorted(cuts):
        if cs > pos:
            kept.append((pos, cs))
        pos = max(pos, ce)
    if pos < total_dur:
        kept.append((pos, total_dur))
    if not kept:
        return [(0.0, total_dur)], words

    def remap(t):
        removed = sum(min(t, ce) - cs for cs, ce in cuts if cs < t)
        return max(0.0, t - removed)

    shifted = [(w, remap(s), remap(e)) for w, s, e in words
               if remap(e) > remap(s)]
    return kept, shifted


def find_profanity(words):
    """Return [(start, end)] ranges to beep."""
    out = []
    for w, s, e in words:
        if _clean_word(w) in PROFANITY:
            out.append((s, e))
    return out


# ---------------- quote cards ----------------

def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def quote_card(quote, cover_path, out_path, size=1080):
    """1080x1080 still: darkened cover + big quote + VOCLIPO tag."""
    cover = Image.open(cover_path).convert('RGB')
    w, h = cover.size
    s = max(size / w, size / h)
    cover = cover.resize((int(w * s) + 1, int(h * s) + 1), Image.LANCZOS)
    x = (cover.width - size) // 2
    y = (cover.height - size) // 2
    cover = cover.crop((x, y, x + size, y + size))
    cover = cover.filter(ImageFilter.GaussianBlur(18))
    img = Image.new('RGB', (size, size), (5, 5, 8))
    img.paste(cover)
    dark = Image.new('RGB', (size, size), (5, 5, 8))
    img = Image.blend(img, dark, 0.55)

    d = ImageDraw.Draw(img)
    tag_font = _font(44)
    d.text((70, 60), 'VOCLIPO', font=tag_font, fill=(0, 229, 255))
    d.rectangle([70, 116, 220, 124], fill=(255, 47, 179))

    fs = 64
    font = _font(fs)
    lines = _wrap(d, f'\u201c{quote}\u201d', font, size - 160)
    while len(lines) > 8 and fs > 40:
        fs -= 4
        font = _font(fs)
        lines = _wrap(d, f'\u201c{quote}\u201d', font, size - 160)
    lh = int(fs * 1.35)
    total_h = lh * len(lines)
    yy = (size - total_h) // 2
    for line in lines:
        tw = d.textlength(line, font=font)
        d.text(((size - tw) / 2, yy), line, font=font, fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))
        yy += lh
    img.save(out_path)
    return out_path


def make_quote_cards(top_quotes, cover_path, out_dir, n=3):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, q in enumerate(top_quotes[:n]):
        p = os.path.join(out_dir, f'quote{i + 1}.png')
        quote_card(q['text'], cover_path, p)
        paths.append(p)
    return paths
