"""Job bookkeeping for render tasks.

Single-worker FIFO queue: one render runs at a time so a small box (or a
beta) never melts under concurrent encodes. Old job files are cleaned up
automatically.
"""
import os
import queue
import shutil
import threading
import time
import uuid

JOBS_DIR = os.path.expanduser('~/workspace/podcast-viz/jobs')
os.makedirs(JOBS_DIR, exist_ok=True)

# job files older than this get deleted (hours)
JOB_RETENTION_HOURS = 24

_jobs = {}
_lock = threading.Lock()
_work_q = queue.Queue()
_worker_started = False


def new_job(title):
    jid = uuid.uuid4().hex[:12]
    d = os.path.join(JOBS_DIR, jid)
    os.makedirs(d, exist_ok=True)
    job = {'id': jid, 'dir': d, 'title': title, 'state': 'queued',
           'progress': 0.0, 'log': [], 'error': None, 'output': None,
           'started': time.time(), 'queue_pos': 0}
    with _lock:
        _jobs[jid] = job
    return job


def get_job(jid):
    with _lock:
        return _jobs.get(jid)


def queue_position(jid):
    """How many jobs are ahead of this one (0 = running or done)."""
    with _lock:
        ahead = 0
        for job in _jobs.values():
            if job['state'] == 'queued' and job['started'] < \
                    _jobs.get(jid, {}).get('started', float('inf')):
                ahead += 1
        return ahead


def append_log(job, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _lock:
        job['log'].append(line)
    with open(os.path.join(job['dir'], 'job.log'), 'a') as f:
        f.write(line + '\n')


def set_state(job, state, **kw):
    with _lock:
        job['state'] = state
        for k, v in kw.items():
            job[k] = v


def set_progress(job, p):
    with _lock:
        job['progress'] = max(0.0, min(1.0, p))


def _worker():
    while True:
        fn, job = _work_q.get()
        try:
            set_state(job, 'starting')
            fn(job)
        except Exception as e:  # noqa: BLE001
            try:
                set_state(job, 'failed', error=str(e))
                append_log(job, 'WORKER FAILED: ' + str(e))
            except Exception:
                pass
        finally:
            _work_q.task_done()


def submit(fn, job):
    """Queue a render. Jobs run one at a time, FIFO."""
    global _worker_started
    with _lock:
        if not _worker_started:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            _worker_started = True
    append_log(job, f'queued (position {queue_position(job["id"]) + 1})')
    _work_q.put((fn, job))


# kept for compatibility
def run_in_background(fn, job):
    return submit(fn, job)


def cleanup_old_jobs(max_age_hours=JOB_RETENTION_HOURS):
    """Delete job dirs (and memory entries) older than max_age_hours."""
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    with _lock:
        for jid in list(_jobs):
            job = _jobs[jid]
            if job.get('started', 0) < cutoff and \
                    job['state'] in ('done', 'failed'):
                try:
                    shutil.rmtree(job['dir'], ignore_errors=True)
                except OSError:
                    pass
                del _jobs[jid]
                removed += 1
    # orphan dirs from a previous process lifetime
    try:
        for name in os.listdir(JOBS_DIR):
            d = os.path.join(JOBS_DIR, name)
            if os.path.isdir(d) and os.path.getmtime(d) < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    except OSError:
        pass
    return removed
