#!/usr/bin/env python3
"""
==============================================================
  autotune_segmentation.py
==============================================================
  Self-tuning CPU runner for Segmentation_ablation_final.py.

  Instead of guessing --jobs and --threads, this script measures
  them on your machine and then adapts to the machine's load
  while it runs.

  It does three things in order:

    1. PROBE      Segment one volume with one worker and record
                  its peak resident memory. This is what decides
                  how many workers can safely run at once --
                  memory, not core count, is the binding
                  constraint for whole-volume 3D inference.

    2. CALIBRATE  Time several (jobs x threads) combinations,
                  each on a few real images, and keep the fastest.
                  Calibration images are real output, not thrown
                  away, so the only cost is that the first ~20
                  images run at sub-optimal settings.

    3. RUN        Process the rest at the winning setting, with a
                  supervisor that reduces concurrency if free RAM
                  falls or if other users' load appears on the
                  machine, and restores it when they go away.

  WHAT THIS CANNOT DO
  -------------------
  It cannot make a single image faster than PyTorch's own
  threading allows, and it cannot know in advance how your CPU's
  memory bandwidth behaves -- that is exactly why it measures
  rather than predicts. On a machine where nothing else is
  running and RAM is plentiful, expect it to land on roughly the
  same setting you would find by hand; its value is that it finds
  it without you watching, and that it backs off if someone else
  starts a job on the same node.

--------------------------------------------------------------
  HOW TO USE
  ----------
  Simplest -- use 12 cores, tune everything else automatically:

      python autotune_segmentation.py \\
          --model b_proposed_no_attention_RU_FINAL_seed1001_b_no_attention_best_metric_model.pth

  Report what it would choose, without processing the full set:

      python autotune_segmentation.py --model <file> --calibrate-only

  Skip calibration and force a setting:

      python autotune_segmentation.py --model <file> --jobs 3 --threads 4

  Leave the machine usable for others (default: on):

      python autotune_segmentation.py --model <file> --budget 12 --polite

  This script imports Segmentation_ablation_final.py for the model
  registry and the per-image functions, so both files must sit in
  the same folder and that file must be working.
==============================================================
"""

import os
import sys
import time
import glob
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

# The segmentation script is the single source of truth for architectures,
# preprocessing and output naming. Importing it (rather than duplicating any
# of it) guarantees the two cannot drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Segmentation_ablation_final as seg          # noqa: E402


# ============================================================
#  MACHINE STATE  (Linux; falls back gracefully)
# ============================================================

def mem_available_gb():
    """Free-ish memory in GB, as the kernel reports it.

    MemAvailable is the right field: it accounts for reclaimable page cache,
    unlike MemFree, which on a busy file server reads near zero and would make
    this script refuse to run.
    """
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return float('inf')          # unknown -> do not block on memory


def load_average():
    """1-minute load average, i.e. mean number of runnable processes."""
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0


def peak_rss_gb():
    """Peak resident memory of THIS process, in GB (VmHWM = high-water mark)."""
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    except Exception:
        return 0.0


# ============================================================
#  WORKER
# ============================================================

_W = {}


def _init_worker(model_path, network_key, threads):
    import torch
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_flush_denormal(True)
    except Exception:
        pass
    _W['model'] = seg.get_model(model_path, network_key,
                                torch.device('cpu'), quiet=True)


def _task(job):
    """Segment one volume. Returns (status, path, seconds, peak_rss_gb, msg)."""
    nifti_file, out_path = job
    t0 = time.time()
    try:
        image, affine, _ = seg.load_image(nifti_file)
        mask = seg.segment_image(_W['model'], image, verbose=False)
        seg.save_segmentation(mask, out_path, affine)
        return ('ok', nifti_file, time.time() - t0, peak_rss_gb(), None)
    except Exception as exc:                                    # noqa: BLE001
        return ('fail', nifti_file, time.time() - t0, peak_rss_gb(),
                f"{type(exc).__name__}: {exc}")


def _pool(model_path, network_key, jobs, threads):
    """A spawn-based pool with thread limits already in the child's env."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    return ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(model_path, network_key, threads),
    )


# ============================================================
#  PHASE 1 — PROBE
# ============================================================

def probe(model_path, network_key, tasks, threads):
    """One image, one worker: how long, and how much memory?"""
    print("\n" + "-" * 62)
    print("  PHASE 1 / 3   Probe: one image, one worker")
    print("-" * 62)
    if not tasks:
        return None, None
    t0 = time.time()
    with _pool(model_path, network_key, 1, threads) as ex:
        status, path, secs, rss, msg = ex.submit(_task, tasks[0]).result()
    if status != 'ok':
        print(f"  Probe FAILED on {os.path.basename(path)}: {msg}")
        print("  Fix this before tuning -- every image would fail the same way.")
        raise SystemExit(1)
    print(f"  {os.path.basename(path)}")
    print(f"  time      : {secs:.1f} s at {threads} thread(s)")
    print(f"  peak RSS  : {rss:.1f} GB per worker")
    print(f"  wall total: {time.time() - t0:.1f} s (includes worker startup)")
    return secs, rss


def safe_job_cap(rss_per_worker, reserve_gb):
    """How many workers fit in memory, leaving `reserve_gb` for the system."""
    avail = mem_available_gb()
    if rss_per_worker <= 0 or avail == float('inf'):
        return 99, avail
    usable = max(0.0, avail - reserve_gb)
    return max(1, int(usable // rss_per_worker)), avail


# ============================================================
#  PHASE 2 — CALIBRATE
# ============================================================

def candidates(budget, job_cap):
    """(jobs, threads) pairs whose product is <= budget, jobs within memory."""
    out = []
    for j in (1, 2, 3, 4, 6, 8, 12):
        if j > min(budget, job_cap):
            continue
        t = max(1, budget // j)
        if (j, t) not in out:
            out.append((j, t))
    return out


def calibrate(model_path, network_key, tasks, cands, per_config):
    """Time each candidate on `per_config` real images. Fastest wins."""
    print("\n" + "-" * 62)
    print("  PHASE 2 / 3   Calibrate")
    print("-" * 62)
    print(f"  {len(cands)} configuration(s) x {per_config} image(s) each")
    print("  (calibration images are kept -- nothing is wasted)\n")

    results, cursor = [], 0
    for jobs, threads in cands:
        batch = tasks[cursor:cursor + per_config * jobs]
        if not batch:
            break
        cursor += len(batch)
        t0 = time.time()
        done = fails = 0
        peak = 0.0
        with _pool(model_path, network_key, jobs, threads) as ex:
            for fut in [ex.submit(_task, b) for b in batch]:
                st, _p, _s, rss, _m = fut.result()
                done += st == 'ok'
                fails += st != 'ok'
                peak = max(peak, rss)
        el = time.time() - t0
        rate = done / el * 60 if el > 0 and done else 0.0
        results.append(dict(jobs=jobs, threads=threads, rate=rate,
                            peak=peak, fails=fails, n=len(batch)))
        print(f"  jobs={jobs:<2} threads={threads:<2}  "
              f"{rate:6.2f} img/min   peak {peak:5.1f} GB/worker"
              f"   ({done} ok, {fails} failed)")

    if not results:
        return None
    best = max(results, key=lambda r: r['rate'])
    print(f"\n  -> best: jobs={best['jobs']} threads={best['threads']} "
          f"({best['rate']:.2f} img/min)")
    if len(results) > 1:
        worst = min(results, key=lambda r: r['rate'])
        if worst['rate'] > 0:
            print(f"     {best['rate']/worst['rate']:.2f}x faster than the "
                  f"slowest configuration tested")
    return best, cursor


# ============================================================
#  PHASE 3 — ADAPTIVE RUN
# ============================================================

class Supervisor:
    """Decides how many tasks may be in flight, re-checked every interval.

    Two reasons to back off, both checked against the machine rather than
    against our own bookkeeping:

      * free memory falling towards the reserve -- the failure mode here is
        swapping, which is far slower than simply using fewer workers;
      * load from OTHER processes -- estimated as total load minus our own
        active workers, so we do not throttle ourselves for our own load.
    """

    def __init__(self, jobs, reserve_gb, polite, budget, interval=60):
        self.max_jobs = jobs
        self.cur = jobs
        self.reserve = reserve_gb
        self.polite = polite
        self.budget = budget
        self.interval = interval
        self._next = time.time() + interval
        self.events = []

    def concurrency(self, in_flight):
        now = time.time()
        if now < self._next:
            return self.cur
        self._next = now + self.interval

        avail = mem_available_gb()
        ext = max(0.0, load_average() - in_flight)
        new = self.cur

        if avail < self.reserve and self.cur > 1:
            new = self.cur - 1
            self.events.append((now, f"RAM {avail:.1f} GB < reserve "
                                     f"{self.reserve:.1f} GB -> {new} worker(s)"))
        elif self.polite and ext > self.budget * 0.5 and self.cur > 1:
            new = self.cur - 1
            self.events.append((now, f"external load {ext:.1f} -> {new} worker(s)"))
        elif (self.cur < self.max_jobs
              and avail > self.reserve * 2
              and (not self.polite or ext < self.budget * 0.25)):
            new = self.cur + 1
            self.events.append((now, f"machine quiet again -> {new} worker(s)"))

        if new != self.cur:
            print(f"\n  [supervisor] {self.events[-1][1]}\n", flush=True)
            self.cur = new
        return self.cur


def run(model_path, network_key, tasks, jobs, threads, reserve_gb,
        polite, budget, already):
    print("\n" + "-" * 62)
    print("  PHASE 3 / 3   Run")
    print("-" * 62)
    print(f"  {len(tasks)} image(s) remaining at jobs={jobs} threads={threads}\n")
    if not tasks:
        return 0, 0

    sup = Supervisor(jobs, reserve_gb, polite, budget)
    done = fails = 0
    t0 = time.time()
    it = iter(tasks)
    pending = set()
    total = len(tasks)

    with _pool(model_path, network_key, jobs, threads) as ex:
        while True:
            # Top up to the concurrency the supervisor currently allows.
            allow = sup.concurrency(len(pending))
            while len(pending) < allow:
                try:
                    pending.add(ex.submit(_task, next(it)))
                except StopIteration:
                    break
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                st, path, secs, _rss, msg = fut.result()
                done += st == 'ok'
                fails += st != 'ok'
                i = done + fails
                el = time.time() - t0
                rate = i / el * 60 if el else 0
                eta = (total - i) / (rate / 60) if rate else 0
                tag = "OK  " if st == 'ok' else "FAIL"
                print(f"  [{i}/{total}] {tag} "
                      f"{os.path.basename(path)[:48]:<48} "
                      f"{rate:5.1f} img/min  ETA {int(eta//3600)}h"
                      f"{int(eta % 3600 // 60):02d}m", flush=True)
                if msg:
                    print(f"        {msg}", flush=True)

    el = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"  Done.  segmented: {done}   failed: {fails}   "
          f"already existed: {already}")
    print(f"  Wall clock: {int(el//3600)}h {int(el % 3600 // 60)}m "
          f"{int(el % 60)}s" + (f"   ({done/el*60:.1f} img/min)" if done else ""))
    if sup.events:
        print(f"\n  Supervisor adjusted concurrency {len(sup.events)} time(s):")
        for ts, what in sup.events:
            print(f"    {time.strftime('%H:%M:%S', time.localtime(ts))}  {what}")
    print(f"{'=' * 62}\n")
    return done, fails


# ============================================================
#  MAIN
# ============================================================

def main(a):
    model_path = os.path.join(a.model_dir, a.model_name)
    if not os.path.isfile(model_path):
        raise SystemExit(f"Model file not found: {model_path}")
    if not os.path.isdir(a.images):
        raise SystemExit(f"Images path not found: {a.images}")

    network_key = a.network or seg.detect_network_key(a.model_name)
    out_tail = f'_PDw_inference_{a.suffix or network_key}.nii.gz'
    tasks, skipped = seg._collect_tasks(a.images, out_tail, a.overwrite)

    n_cpu = os.cpu_count() or 1
    budget = min(a.budget, n_cpu)

    print("\n" + "=" * 62)
    print("  Self-tuning CPU segmentation")
    print("=" * 62)
    print(f"  Network      : {network_key}")
    print(f"  Output suffix: {out_tail}")
    print(f"  Images       : {len(tasks) + skipped} found, "
          f"{len(tasks)} to do, {skipped} already done")
    print(f"  Cores        : {budget} of {n_cpu} (budget)")
    print(f"  RAM available: {mem_available_gb():.1f} GB "
          f"(reserving {a.reserve_ram} GB)")
    print(f"  Load average : {load_average():.2f}")
    print(f"  Polite mode  : {'on' if a.polite else 'off'}")
    print("=" * 62)

    if not tasks:
        print("\n  Nothing to do -- every image already has an output.\n")
        return

    # Manual override: skip probing and calibration entirely.
    if a.jobs and a.threads:
        run(model_path, network_key, tasks, a.jobs, a.threads,
            a.reserve_ram, a.polite, budget, skipped)
        return

    secs, rss = probe(model_path, network_key, tasks, budget)
    cap, avail = safe_job_cap(rss, a.reserve_ram)
    print(f"\n  Memory allows at most {cap} concurrent worker(s) "
          f"({avail:.1f} GB available, {rss:.1f} GB each, "
          f"{a.reserve_ram} GB reserved).")
    if cap == 1:
        print("  -> single worker; more processes would swap and be slower.")

    cands = candidates(budget, cap)
    remaining = tasks[1:]                       # image 0 consumed by the probe

    if len(cands) == 1 or a.no_calibrate:
        jobs, threads = cands[0]
        print(f"\n  Only one viable configuration -- skipping calibration."
              if len(cands) == 1 else "\n  Calibration disabled.")
        used = 0
    else:
        best, used = calibrate(model_path, network_key, remaining,
                               cands, a.calib_images)
        jobs, threads = best['jobs'], best['threads']

    if a.calibrate_only:
        print(f"\n  --calibrate-only: stopping here.")
        print(f"  Recommended:  --jobs {jobs} --threads {threads}\n")
        return

    run(model_path, network_key, remaining[used:], jobs, threads,
        a.reserve_ram, a.polite, budget, skipped)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "--model-name", dest="model_name", required=True)
    p.add_argument("--model-dir", dest="model_dir",
                   default=seg.DEFAULT_MODEL_DIR)
    p.add_argument("--images", default=seg.DEFAULT_IMAGES_PATH)
    p.add_argument("--network", default=None,
                   choices=[k for k, _ in seg.NETWORKS])
    p.add_argument("--suffix", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--budget", type=int, default=12,
                   help="Total cores this job may use.")
    p.add_argument("--reserve-ram", dest="reserve_ram", type=float, default=8.0,
                   help="GB of RAM to leave free for the rest of the system.")
    p.add_argument("--calib-images", dest="calib_images", type=int, default=3,
                   help="Images per worker per configuration during calibration.")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Probe for memory, then use the largest safe setting "
                        "without timing alternatives.")
    p.add_argument("--calibrate-only", action="store_true",
                   help="Probe and calibrate, print the recommendation, stop.")
    p.add_argument("--jobs", type=int, default=None,
                   help="Force worker count (skips probe and calibration; "
                        "requires --threads).")
    p.add_argument("--threads", type=int, default=None,
                   help="Force threads per worker.")
    p.add_argument("--polite", dest="polite", action="store_true", default=True,
                   help="Back off when other users load the machine (default).")
    p.add_argument("--no-polite", dest="polite", action="store_false",
                   help="Ignore other load; take the full budget.")
    main(p.parse_args())
