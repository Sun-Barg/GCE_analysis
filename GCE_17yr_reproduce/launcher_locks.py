"""launcher_locks.py — per-model PID lock + orphan worker adoption.

Prevents launcher-restart double-spawn race:
  launcher dies → orphan workers keep running → watchdog spawns new launcher
  → new launcher sees no .dat → respawns workers for the SAME models.

Mechanism:
  - On launch_one: write .locks/{model}.pid = wrapper PID.
  - On reap: remove_lock(model).
  - On launcher startup:
      cleanup_stale_locks() removes locks whose PID is dead.
      adopt_running_workers() pgrep's live workers and adds them
      to adopted_pids so runnable filter excludes them.
  - In main loop: reap adopted workers via os.kill(pid, 0).
"""
import os
import subprocess

LOCKS_DIR = '.locks'


def _lock_path(model):
    return os.path.join(LOCKS_DIR, f'{model}.pid')


def write_lock(model, pid):
    os.makedirs(LOCKS_DIR, exist_ok=True)
    with open(_lock_path(model), 'w') as f:
        f.write(str(pid))


def read_lock(model):
    """Return (pid, alive). (None, False) if no lock or unreadable."""
    p = _lock_path(model)
    if not os.path.exists(p):
        return None, False
    try:
        with open(p) as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        return None, False
    try:
        os.kill(pid, 0)
        return pid, True
    except ProcessLookupError:
        return pid, False
    except PermissionError:
        # PID exists but owned by another user — treat as alive (safe side)
        return pid, True


def remove_lock(model):
    try:
        os.remove(_lock_path(model))
    except OSError:
        pass


def cleanup_stale_locks(models):
    """Remove locks whose PID is dead. Returns count removed."""
    n = 0
    for m in models:
        pid, alive = read_lock(m)
        if pid is not None and not alive:
            remove_lock(m)
            n += 1
    return n


def adopt_running_workers(models, adopted_pids, running, results_dir,
                          work_dir, runner_script, is_complete_fn,
                          self_pid):
    """Find live wrapper processes for our models; add to adopted_pids.

    A model is adopted only if:
      - a live `runner_script {model}` process exists,
      - its PID != self_pid,
      - model is in our models list,
      - is_complete_fn(model) is False,
      - model is not already tracked in running or adopted_pids.

    Returns the list of model names newly adopted.
    """
    new = []
    try:
        out = subprocess.check_output(
            ['pgrep', '-af', runner_script], text=True,
        )
    except subprocess.CalledProcessError:
        return new
    models_str = {str(x): x for x in models}  # int(ROI)/str(model) 양쪽 호환
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == self_pid:
            continue
        # cmdline last token = model/ROI name
        m = parts[-1].strip()
        if m not in models_str:
            continue
        m = models_str[m]  # restore original type (int for ROI)
        if m in running or m in adopted_pids:
            continue
        if is_complete_fn(m, results_dir, work_dir):
            continue
        adopted_pids[m] = pid
        write_lock(m, pid)
        new.append(m)
    return new


def adopted_alive(pid):
    """True if adopted worker PID still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
