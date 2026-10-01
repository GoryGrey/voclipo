"""Voclipo: upload audio, get scroll-stopping captioned clips.

Audio in. Clips out. Local, free, no accounts, no cloud.
"""
import json
import os
import shutil
import traceback

from flask import (Flask, render_template, request, redirect, url_for,
                   jsonify, send_file, abort)

from viz import jobs
from viz.captions import (transcribe_words, build_ass, build_srt,
                          CAPTION_PRESETS)
from viz.backgrounds import build_background
from viz.waveforms import (wave_video, wave_mode, WAVEFORMS, WAVE_THEMES)
from viz.backgrounds import build_background, BACKGROUNDS
from viz.render import render_episode, LAYOUTS, ASPECTS, INTRO_SECS
from viz.audio import (clean_voice, cut_ranges, beep_words, probe_duration)
from viz.clips import (clip_scorecard, dead_air_cuts, find_profanity,
                       make_quote_cards)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300 MB

AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus'}
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
TXT_EXTS = {'.txt', '.md'}
PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'presets.json')

# ---- ad slots: paste ad/affiliate code into templates where marked ----
ADS = {
    'enabled': False,          # flip True when ad code is pasted in
    'index_banner': '',        # HTML pasted below the upload form
    'job_banner': '',         # HTML pasted on the finished job page
}

# ---- limited beta: one code per line in beta_codes.txt ----
BETA_CODES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'beta_codes.txt')
BETA_USAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'beta_usage.json')
MAX_AUDIO_SECONDS = 20 * 60  # beta cap: 20 minutes per episode


def load_beta_codes():
    try:
        with open(BETA_CODES_PATH) as f:
            return {ln.strip() for ln in f if ln.strip()
                    and not ln.startswith('#')}
    except OSError:
        return set()


def beta_enabled():
    return bool(load_beta_codes())


def check_beta_code(code):
    codes = load_beta_codes()
    if not codes:
        return True  # no codes file yet: gate open (dev mode)
    return (code or '').strip() in codes


def record_beta_use(code):
    code = (code or '').strip()
    if not code:
        return 0
    try:
        with open(BETA_USAGE_PATH) as f:
            usage = json.load(f)
    except (OSError, ValueError):
        usage = {}
    usage[code] = usage.get(code, 0) + 1
    try:
        with open(BETA_USAGE_PATH, 'w') as f:
            json.dump(usage, f, indent=2)
    except OSError:
        pass
    return usage[code]

# ---- built-in one-click series templates (merged with user presets) ----
SERIES_TEMPLATES = {
    'Hot Take':      {'bg_style': 'matrix',
                      'wave_style': 'triband', 'wave_theme': 'sunset',
                      'caption_style': 'ember', 'aspect': 'vertical',
                      'intro': True, 'clean_mode': False},
    'Storytime':     {'bg_style': 'nebula', 'wave_style': 'bars',
                      'wave_theme': 'volt', 'caption_style': 'classic',
                      'aspect': 'vertical', 'intro': True,
                      'clean_mode': False},
    'Interview':     {'bg_style': 'grid', 'wave_style': 'mirror',
                      'wave_theme': 'ice', 'caption_style': 'ghost',
                      'aspect': 'square', 'intro': True, 'clean_mode': False},
    'Hype':          {'bg_style': 'starfield', 'wave_style': 'particles',
                      'wave_theme': 'ultraviolet',
                      'caption_style': 'big', 'aspect': 'vertical',
                      'intro': True, 'clean_mode': False},
    'Clean Comedy':  {'bg_style': 'bokeh', 'wave_style': 'spectrum',
                      'wave_theme': 'lime', 'caption_style': 'neon',
                      'aspect': 'vertical', 'intro': False,
                      'clean_mode': True},
}


def _ext(name):
    return os.path.splitext(name.lower())[1]


def load_presets():
    try:
        with open(PRESETS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_preset(name, settings):
    presets = load_presets()
    presets[name] = settings
    with open(PRESETS_PATH, 'w') as f:
        json.dump(presets, f, indent=2)


def all_presets():
    merged = dict(SERIES_TEMPLATES)
    merged.update(load_presets())
    return merged


def pipeline(job):
    d = job['dir']
    audio = os.path.join(d, 'audio' + _ext(job['audio_name']))
    cover = os.path.join(d, 'cover' + _ext(job['cover_name']))
    script = job.get('script_path')
    bg_style = job['bg_style']
    wave_style = job['wave_style']
    wave_theme = job.get('wave_theme', 'volt')
    caption_style = job.get('caption_style', 'classic')
    aspect = job.get('aspect', 'vertical')
    intro = job.get('intro', True)
    voice_cleanup = job.get('voice_cleanup', False)
    dead_air = job.get('dead_air', False)
    clean_mode = job.get('clean_mode', False)
    title = job['title']
    bg_image = job.get('bg_image_path')
    log = lambda m: jobs.append_log(job, m)  # noqa: E731

    try:
        # ---- audio prep: cleanup -> dead-air cuts -> clean-mode beeps ----
        work_audio = audio
        if voice_cleanup:
            jobs.set_state(job, 'audio prep')
            log('voice cleanup: highpass + loudness normalize...')
            work_audio = clean_voice(audio, os.path.join(d, 'clean.wav'),
                                     log=log)
            jobs.set_progress(job, 0.06)

        jobs.set_state(job, 'transcribing')
        jobs.set_progress(job, 0.08)
        log('transcribing audio for word timestamps...')
        words = transcribe_words(work_audio,
                                 log=lambda m: jobs.append_log(job, m))
        jobs.set_progress(job, 0.30)
        log(f'{len(words)} words transcribed')

        total_dur = probe_duration(work_audio) or (
            words[-1][2] if words else 0.0)
        if dead_air and words:
            jobs.set_state(job, 'audio prep')
            log('dead-air killer: trimming silence and long pauses...')
            kept, words = dead_air_cuts(words, total_dur)
            work_audio = cut_ranges(work_audio, kept,
                                    os.path.join(d, 'tight.wav'), log=log)
            total_dur = sum(e - s for s, e in kept)
            log(f'{len(words)} words after tightening ({total_dur:.1f}s)')
        if clean_mode and words:
            jobs.set_state(job, 'audio prep')
            spans = find_profanity(words)
            if spans:
                log(f'clean mode: beeping {len(spans)} word(s)...')
                work_audio = beep_words(work_audio, spans,
                                        os.path.join(d, 'cleanbeep.wav'),
                                        log=log)
            else:
                log('clean mode: no profanity found, audio untouched')

        jobs.set_state(job, 'captions')
        script_text = ''
        if script:
            with open(script) as f:
                script_text = f.read()
            log(f'script uploaded ({len(script_text)} chars), aligning...')
        else:
            log('no script uploaded, using transcription text')
        layout = LAYOUTS[aspect]
        ass_path = os.path.join(d, 'captions.ass')
        n_words, n_events = build_ass(
            words, script_text, ass_path, preset=caption_style,
            play_res_x=layout['vw'], play_res_y=layout['vh'],
            margin_v=layout['caption_margin_v'])
        log(f'captions: {n_words} words in {n_events} events '
            f'({caption_style})')
        # standalone SRT, shifted if the intro card pushes captions later
        srt_path = os.path.join(d, 'captions.srt')
        build_srt(words, script_text, srt_path,
                  time_offset=INTRO_SECS if intro else 0.0)
        log('SRT captions written')

        # ---- clip intelligence: scorecard + quote cards ----
        jobs.set_state(job, 'clip intel')
        scorecard = clip_scorecard(words, total_dur)
        job['scorecard'] = scorecard
        log(f"clip scorecard: {len(scorecard['signals'])} signals, "
            f"{scorecard['questions']} question(s)")
        quote_paths = []
        if scorecard['top_quotes']:
            try:
                quote_paths = make_quote_cards(
                    scorecard['top_quotes'], cover,
                    os.path.join(d, 'quotes'))
                log(f'{len(quote_paths)} quote cards rendered')
            except Exception as e:  # noqa: BLE001
                log('quote cards skipped: ' + str(e))
        job['quote_paths'] = quote_paths
        jobs.set_progress(job, 0.40)

        jobs.set_state(job, 'background')
        log(f'building {bg_style} background...')
        bg_mp4 = os.path.join(d, 'bg.mp4')
        build_background(bg_style, bg_mp4, os.path.join(d, 'bgcache'),
                         cover_path=cover, custom_path=bg_image)
        jobs.set_progress(job, 0.46)

        wave_mp4 = None
        if wave_mode(wave_style) == 'video':
            jobs.set_state(job, 'waveform')
            log(f'pre-rendering {wave_style} waveform ({wave_theme})...')
            wave_mp4 = os.path.join(d, 'wave.mov')
            wave_video(wave_style, wave_theme, work_audio, wave_mp4,
                       os.path.join(d, 'wcache'))
        jobs.set_progress(job, 0.50)

        jobs.set_state(job, 'rendering')
        log(f'rendering {aspect} video segments...')
        out = render_episode(d, work_audio, cover, ass_path, title, bg_mp4,
                             wave_style, wave_theme=wave_theme,
                             aspect=aspect, intro=intro,
                             out_name='episode.mp4',
                             wave_video_path=wave_mp4,
                             log=log,
                             progress=lambda p: jobs.set_progress(
                                 job, 0.50 + 0.50 * p))
        job['srt_path'] = srt_path
        jobs.set_state(job, 'done', output=out)
        log('finished')
    except Exception as e:  # noqa: BLE001
        jobs.set_state(job, 'failed', error=str(e))
        jobs.append_log(job, 'FAILED: ' + str(e))
        jobs.append_log(job, traceback.format_exc(limit=5))


@app.route('/')
def index():
    return render_template(
        'index.html',
        presets=all_presets(),
        series_templates=SERIES_TEMPLATES,
        aspects=ASPECTS,
        caption_presets=CAPTION_PRESETS,
        wave_themes=WAVE_THEMES,
        ads=ADS,
        beta_on=beta_enabled(),
    )


@app.route('/render', methods=['POST'])
def start_render():
    title = request.form.get('title', '').strip() or 'Untitled Episode'
    audio_f = request.files.get('audio')
    cover_f = request.files.get('cover')
    script_f = request.files.get('script')
    bg_style = request.form.get('bg_style', 'nebula')
    wave_style = request.form.get('wave_style', 'neon')
    wave_theme = request.form.get('wave_theme', 'volt')
    caption_style = request.form.get('caption_style', 'classic')
    aspect = request.form.get('aspect', 'vertical')
    intro = request.form.get('intro') == 'on'
    voice_cleanup = request.form.get('voice_cleanup') == 'on'
    dead_air = request.form.get('dead_air') == 'on'
    clean_mode = request.form.get('clean_mode') == 'on'
    preset_name = request.form.get('preset_name', '').strip()
    beta_code = request.form.get('beta_code', '').strip()
    bg_image_f = request.files.get('bg_image')

    if not check_beta_code(beta_code):
        return render_template('beta_locked.html'), 403

    if not audio_f or not audio_f.filename:
        return 'Audio file is required', 400
    if _ext(audio_f.filename) not in AUDIO_EXTS:
        return 'Audio must be mp3, wav, m4a, ogg, flac or opus', 400
    if not cover_f or not cover_f.filename:
        return 'Cover art image is required', 400
    if _ext(cover_f.filename) not in IMG_EXTS:
        return 'Cover must be jpg, png or webp', 400
    if bg_style == 'custom':
        if not bg_image_f or not bg_image_f.filename:
            return 'Custom background needs an image upload', 400
        if _ext(bg_image_f.filename) not in IMG_EXTS:
            return 'Background image must be jpg, png or webp', 400
    if bg_style not in BACKGROUNDS:
        return 'Unknown background style', 400
    if wave_style not in WAVEFORMS:
        return 'Unknown waveform style', 400
    if wave_theme not in WAVE_THEMES:
        return 'Unknown waveform color theme', 400
    if caption_style not in CAPTION_PRESETS:
        return 'Unknown caption style', 400
    if aspect not in ASPECTS:
        return 'Unknown aspect ratio', 400

    settings = {'bg_style': bg_style, 'wave_style': wave_style,
                'wave_theme': wave_theme, 'caption_style': caption_style,
                'aspect': aspect, 'intro': intro,
                'voice_cleanup': voice_cleanup, 'dead_air': dead_air,
                'clean_mode': clean_mode}
    if preset_name:
        save_preset(preset_name, settings)

    job = jobs.new_job(title)
    d = job['dir']
    audio_path = os.path.join(d, 'audio' + _ext(audio_f.filename))
    cover_path = os.path.join(d, 'cover' + _ext(cover_f.filename))
    audio_f.save(audio_path)
    cover_f.save(cover_path)

    dur = probe_duration(audio_path)
    if dur and dur > MAX_AUDIO_SECONDS:
        shutil.rmtree(d, ignore_errors=True)
        return (f'Audio is {dur / 60:.1f} minutes, beta limit is '
                f'{MAX_AUDIO_SECONDS // 60} minutes', 400)

    job['audio_name'] = audio_f.filename
    job['cover_name'] = cover_f.filename
    job['title'] = title
    job.update(settings)

    script_path = None
    if script_f and script_f.filename:
        if _ext(script_f.filename) in TXT_EXTS:
            script_path = os.path.join(d, 'script.txt')
            script_f.save(script_path)
    job['script_path'] = script_path

    bg_image_path = None
    if bg_style == 'custom':
        bg_image_path = os.path.join(d, 'bgimg' + _ext(bg_image_f.filename))
        bg_image_f.save(bg_image_path)
    job['bg_image_path'] = bg_image_path

    jobs.append_log(job, f'job queued: {title}')
    record_beta_use(beta_code)
    jobs.cleanup_old_jobs()
    jobs.submit(pipeline, job)
    return redirect(url_for('job_page', jid=job['id']))


@app.route('/job/<jid>')
def job_page(jid):
    job = jobs.get_job(jid)
    if not job:
        abort(404)
    return render_template('job.html', job=job, ads=ADS)


@app.route('/job/<jid>/status')
def job_status(jid):
    job = jobs.get_job(jid)
    if not job:
        abort(404)
    return jsonify({'state': job['state'], 'progress': job['progress'],
                    'log': job['log'][-80:], 'error': job['error'],
                    'done': job['state'] == 'done',
                    'queue_pos': jobs.queue_position(jid),
                    'scorecard': job.get('scorecard'),
                    'n_quotes': len(job.get('quote_paths') or [])})


@app.route('/job/<jid>/download')
def download(jid):
    job = jobs.get_job(jid)
    if not job or job['state'] != 'done' or not job['output']:
        abort(404)
    return send_file(job['output'], as_attachment=True,
                     download_name='voclipo.mp4', mimetype='video/mp4')


@app.route('/job/<jid>/srt')
def download_srt(jid):
    job = jobs.get_job(jid)
    srt = job.get('srt_path') if job else None
    if not job or job['state'] != 'done' or not srt or \
            not os.path.exists(srt):
        abort(404)
    return send_file(srt, as_attachment=True,
                     download_name='captions.srt',
                     mimetype='text/plain')


@app.route('/job/<jid>/quote/<int:n>')
def download_quote(jid, n):
    job = jobs.get_job(jid)
    paths = job.get('quote_paths') if job else None
    if not job or job['state'] != 'done' or not paths or \
            not (1 <= n <= len(paths)) or \
            not os.path.exists(paths[n - 1]):
        abort(404)
    return send_file(paths[n - 1], as_attachment=True,
                     download_name=f'quote{n}.png',
                     mimetype='image/png')


if __name__ == '__main__':
    jobs.cleanup_old_jobs()
    app.run(host='127.0.0.1', port=5057, debug=False)
