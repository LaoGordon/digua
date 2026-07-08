#!/usr/bin/env python3
"""System-level monitor for FAST-LIVO2 process death (upgraded 2026-06-24).

Runs alongside run_navigation_real.sh. Samples memory + FAST-LIVO2 RSS/CPU
plus NEW dimensions every 1s:
  - RSS (resident set, MB)
  - VIRT (virtual memory, MB) — glibc heap damage may inflate VM before abort
  - threads count — data races often correlate with thread count anomalies
  - [heap] segment size (from /proc/$PID/maps) — glibc brk heap growth
  - anonymous mmap segment count — per-thread arenas / large allocs
The moment fastlivo_mapping disappears, captures dmesg OOM + final snapshot
+ (best-effort) last /proc/$PID/maps and status.

All reads are BYPASS (/proc, ps) — does NOT touch the process (no gdb/strace),
so it is safe on PREEMPT_RT (will not break realtime like gdb/ASan/MALLOC_CHECK_).

Usage: python3 sysmon.py <output_log_path>
"""
import sys, os, time, subprocess, signal

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sysmon.log"
INTERVAL = 1.0  # seconds (was 2.0; 1.0 to catch pre-crash spikes)
PROCPAT = "fastlivo_mapping"

logf = open(LOG_PATH, "w", buffering=1)  # line-buffered

def log(msg):
    logf.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception as e:
        return f"(err: {e})"

def mem_info():
    out = run("free -m")
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            return f"mem total={parts[1]}M used={parts[2]}M free={parts[3]}M avail={parts[6]}M"
    return out.replace("\n", " | ")

def swap_info():
    out = run("free -m")
    for line in out.splitlines():
        if line.startswith("Swap:"):
            parts = line.split()
            if len(parts) >= 3:
                return f"swap total={parts[1]}M used={parts[2]}M"
    return "swap unknown"

def proc_info(name):
    """Return dict with pid, rss_mb, virt_mb, cpu, stat, OR None.
    Uses ps aux; rss is parts[5] (KB), vsz is parts[4] (KB)."""
    out = run(f"ps aux | grep '[{name[0]}]{name[1:]}' | head -1")
    if not out:
        return None
    parts = out.split()
    if len(parts) >= 8:
        return {
            "pid": parts[1],
            "rss_mb": int(parts[5]) // 1024,
            "virt_mb": int(parts[4]) // 1024,
            "cpu": parts[2],
            "stat": parts[7],
        }
    return None

def thread_count(pid):
    """Number of threads via /proc/$PID/task."""
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except Exception:
        return -1

def maps_stats(pid):
    """Parse /proc/$PID/maps: count anon mmap segments + [heap] size.
    Returns (anon_count, heap_kb_str). Bypass read, no ptrace."""
    anon = 0
    heap_kb = 0
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                # [heap] line: parse address range for size
                if "[heap]" in line:
                    addrs = line.split()[0]
                    start, end = addrs.split("-")
                    heap_kb = (int(end, 16) - int(start, 16)) // 1024
                # anonymous mmap: no file path at end AND readable+writable
                elif "rw" in line[:5] and (line.rstrip().endswith(" ") or line.split()[-1] in ("", "[anon:libc_malloc]")):
                    # anonymous mappings typically have no pathname or [anon:*]
                    last = line.rstrip().split(" ")[-1] if line.strip() else ""
                    if last == "" or last.startswith("[anon"):
                        anon += 1
    except Exception:
        pass
    return anon, heap_kb

def all_nav_procs():
    return run("ps aux --sort=-%mem | head -8")

def dmesg_oom():
    out = run("dmesg 2>/dev/null | grep -iE 'oom|kill|out of memory|killed process|memory cgroup' | tail -20")
    return out if out else "(no OOM records in dmesg)"

def dmesg_recent(seconds=60):
    return run(f"sudo dmesg --since '@{int(time.time())-seconds}' 2>/dev/null | tail -30 || dmesg 2>/dev/null | tail -30")

def dump_proc_snapshot(pid):
    """Best-effort full snapshot at death moment: maps + status + smaps rollup."""
    out = []
    # maps
    try:
        with open(f"/proc/{pid}/maps") as f:
            out.append("=== /proc/$PID/maps (death moment) ===")
            out.append(f.read())
    except Exception as e:
        out.append(f"(maps read failed: {e} — process already reaped)")
    # status
    try:
        with open(f"/proc/{pid}/status") as f:
            out.append("=== /proc/$PID/status ===")
            out.append(f.read())
    except Exception as e:
        out.append(f"(status read failed: {e})")
    return "\n".join(out)

# --- Main loop ---
running = True
def handler(sig, frame):
    global running
    running = False
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

log("=" * 60)
log(f"System monitor started (upgraded 2026-06-24). LOG={LOG_PATH}")
log(f"Monitoring process: {PROCPAT}")
log(f"Sample interval: {INTERVAL}s (RSS+VIRT+threads+heap+anon, bypass reads)")
log(f"Initial: {mem_info()} | {swap_info()}")
log("=" * 60)

max_rss = 0
max_rss_time = ""
max_virt = 0
sample_count = 0
died_reported = False
died_ts = None
died_time = None
last_pid = None

while running:
    p = proc_info(PROCPAT)
    m = mem_info()
    s = swap_info()

    if p:
        sample_count += 1
        pid = p["pid"]
        last_pid = pid
        rss = p["rss_mb"]
        virt = p["virt_mb"]
        thr = thread_count(pid)
        anon, heap_kb = maps_stats(pid)
        if rss > max_rss:
            max_rss = rss
            max_rss_time = time.strftime("%H:%M:%S")
        if virt > max_virt:
            max_virt = virt
        # Log EVERY sample (1s) — we want full pre-crash trace. Compact format.
        log(f"ALIVE pid={pid} rss={rss}MB virt={virt}MB cpu={p['cpu']}% thr={thr} heap={heap_kb}KB anon={anon} | {m}")
    else:
        # Process gone!
        if not died_reported:
            died_reported = True
            died_ts = time.time()
            died_time = time.strftime("%H:%M:%S")
            log("!" * 60)
            log(f"!!! {PROCPAT} DISAPPEARED at {died_time} !!!")
            log(f"!!! Sample #{sample_count}, peak RSS={max_rss}MB @ {max_rss_time}, peak VIRT={max_virt}MB")
            log("!" * 60)
            log(f"[MEM at death] {m} | {s}")
            log(f"[TOP PROCS at death]")
            logf.write(all_nav_procs() + "\n")
            # best-effort proc snapshot (usually fails — process already dead/reaped,
            # but if we catch it in zombie state, maps still readable)
            if last_pid:
                log(f"[PROC SNAPSHOT for pid={last_pid} (best-effort)]")
                logf.write(dump_proc_snapshot(last_pid) + "\n")
            log(f"[DMESG OOM records]")
            logf.write(dmesg_oom() + "\n")
            log(f"[DMESG recent 60s]")
            logf.write(dmesg_recent(60) + "\n")
            log("!" * 60)
            log(f"Will keep sampling memory for 30s post-death, then exit.")
            logf.flush()
        else:
            if sample_count % 5 == 0:
                log(f"post-death {m} | {s}")
            if time.time() - died_ts > 30:
                log(f"30s post-death elapsed, exiting monitor.")
                break

    time.sleep(INTERVAL)

# Cleanup on exit
log("=" * 60)
log(f"Monitor stopped. Total samples: {sample_count}")
log(f"Peak FAST-LIVO2 RSS: {max_rss}MB @ {max_rss_time} | Peak VIRT: {max_virt}MB")
log(f"Final: {mem_info()} | {swap_info()}")
log("=" * 60)
logf.close()
