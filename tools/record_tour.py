#!/usr/bin/env python3
"""Unattended recording tour for the pipeline warm-up (issue #11): loads checkpoints and lets the patched DXVK record every pipeline.

One short game process per chunk of at most --chunk checkpoints of one map (long sessions lost the GPU device three times, so the
game is restarted often). Every chunk starts from a fresh copy of the profile with a TdInput.ini key chain made by
gen_checkpoint_bindings.py, runs the game with ME_PC_AUTO_INPUT, waits until the last key press and the level after it have had time
to draw, then stops the game. All chunks append to one pipelines.warm (+ .shaders). The launch-time rebuild is switched off
(DXVK_PIPELINE_WARM_BOOT=0) so the tour records what the game draws, not what the rebuild already knows.

usage: record_tour.py --exe me_pc --binaries <pc/Binaries> --deps <pc-deps> --profile <profile dir> --dxvk <libdxvk_d3d9.0.dylib>
                      --out <dir> [--checkpoints pc/tools/checkpoints.txt] [--chunk 10] [--maps Edge_p,Escape_p] [--spacing 1500]
                      [--first-frame 2600] [--settle 30] [--chunk-timeout 2400] [--compare <pipelines.warm.shaders of a reference recording>]

Outputs under --out: pipelines.warm(.shaders), dxvk-cache/ (DXVK_SHADER_CACHE_PATH), chunks/NN-<map>/{run.log,profile}, progress.csv.
The recording is derived from game data: keep --out out of the repository.
"""
import argparse, csv, glob, os, re, shutil, struct, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def read_shaders(path):
    out = {}
    if not os.path.exists(path):
        return out
    d = open(path, "rb").read(); p = 0
    while p < len(d):
        nl, _px, sz = struct.unpack_from("<HBI", d, p); p += 7
        name = d[p:p + nl].decode(); p += nl
        out[name] = d[p:p + sz]; p += sz
    return out


def count_records(path):
    if not os.path.exists(path):
        return 0
    d = open(path, "rb").read()
    if d[:4] != b"MEPW":
        return 0
    _v, sz = struct.unpack("<II", d[4:12]); p = 12; n = 0
    while p < len(d):
        a, b, m = struct.unpack("<HHH", d[p:p + 6]); p += 6 + a + b + m + sz; n += 1
    return n


def count_metal_manifest(path):
    if not os.path.exists(path):
        return (0, 0)
    with open(path, "rb") as f:
        hdr = f.read(16)
        if len(hdr) < 16:
            return (0, 0)
        magic, ver, rec_sz, count = struct.unpack("<4sIII", hdr)
        if magic != b"1SPM" or ver != 1 or rec_sz < 8:
            return (0, 0)
        maps = set()
        for _ in range(count):
            rec = f.read(rec_sz)
            if len(rec) < 8:
                break
            maps.add(struct.unpack_from("<Q", rec, 0)[0])
    return (count, len(maps))


def set_ini_resolution(prof_dir, res_str):
    m = re.match(r"^(\d+)x(\d+)$", res_str.strip().lower())
    if not m:
        return
    w_str, h_str = m.group(1), m.group(2)
    for root, _, files in os.walk(prof_dir):
        for fn in files:
            if fn.lower() == "tdengine.ini":
                p = os.path.join(root, fn)
                raw = open(p, "r", encoding="utf-8", errors="replace", newline="").read()
                raw = re.sub(r"(?m)^ResX=\d+", f"ResX={w_str}", raw)
                raw = re.sub(r"(?m)^ResY=\d+", f"ResY={h_str}", raw)
                open(p, "w", encoding="utf-8", newline="").write(raw)


def count_metal_shaders(cache_dir):
    return len(glob.glob(os.path.join(cache_dir, "*.msl"))) + len(glob.glob(os.path.join(cache_dir, "*.ir8*")))


def main():
    ap = argparse.ArgumentParser()
    for k in ("exe", "binaries", "deps", "profile", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--dxvk", default="")
    ap.add_argument("--metal", action="store_true", help="Record native Metal pso_manifest_v10*.bin + ShaderCache instead of DXVK")
    ap.add_argument("--checkpoints", default=os.path.join(HERE, "checkpoints.txt"))
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--maps", default="")
    ap.add_argument("--spacing", type=int, default=None)
    ap.add_argument("--first-frame", type=int, default=None)
    ap.add_argument("--settle", type=int, default=None, help="seconds to keep running after the last action")
    ap.add_argument("--chunk-timeout", type=int, default=2400)
    ap.add_argument("--compare", default="")
    ap.add_argument("--new-game-keys", default=None, help="ME_PC_AUTO_KEYS: seconds at which Return is tapped to leave the menu and start the game")
    ap.add_argument("--res", default=None, help="Override ResX x ResY in temporary chunk TdEngine.ini (e.g. 1280x720)")
    ap.add_argument("--no-spin", action="store_true", help="Disable camera spin after each checkpoint in --metal mode")
    ap.add_argument("--no-audio", action="store_true", help="Mute game audio via ME_PC_NO_AUDIO=1")
    ap.add_argument("--start-chunk", type=int, default=1, help="1-based chunk index to resume from")
    a = ap.parse_args()
    if not a.metal and not a.dxvk:
        sys.exit("either --metal or --dxvk is required")
    if a.new_game_keys is None:
        a.new_game_keys = "4,7,10,13,15,17,19,21" if a.metal else "32,36,40,44,48"
    if a.first_frame is None:
        a.first_frame = 2000 if a.metal else 2600
    if a.spacing is None:
        a.spacing = 700 if a.metal else 1500
    if a.settle is None:
        a.settle = 10 if a.metal else 30
    if a.res is None:
        a.res = "1280x720" if a.metal else ""

    cps = [l.strip() for l in open(a.checkpoints, encoding="utf-8") if l.startswith("start ")]
    if a.maps:
        keep = set(a.maps.split(","))
        cps = [c for c in cps if re.match(r"start (\w+)", c).group(1) in keep]
    chunks, order = [], []
    for c in cps:
        m = re.match(r"start (\w+)", c).group(1)
        if not chunks or chunks[-1][0] != m or len(chunks[-1][1]) >= a.chunk:
            chunks.append((m, []))
        chunks[-1][1].append(c)

    out = os.path.abspath(a.out); os.makedirs(out + "/chunks", exist_ok=True)
    warm = out + "/pipelines.warm"
    cache = os.path.join(out, "ShaderCache") if a.metal else (out + "/dxvk-cache")
    manifest = os.path.join(cache, "pso_manifest_v10.bin")
    os.makedirs(cache, exist_ok=True)
    ref = read_shaders(a.compare) if a.compare else {}
    refcodes = set(ref.values())
    prog = open(out + "/progress.csv", "a", newline=""); w = csv.writer(prog)
    if prog.tell() == 0:
        w.writerow(["chunk", "map", "checkpoints", "status", "seconds", "records", "shaders", "ref_covered", "ref_total"]); prog.flush()

    for i, (m, lst) in enumerate(chunks, 1):
        if i < a.start_chunk:
            continue
        cdir = "%s/chunks/%02d-%s" % (out, i, m); shutil.rmtree(cdir, ignore_errors=True); os.makedirs(cdir)
        prof = cdir + "/profile"
        shutil.copytree(a.profile, prof, symlinks=True,
                        ignore=shutil.ignore_patterns("ShaderCache", "cache", "dxvk-cache", "*.log"))
        if a.res:
            set_ini_resolution(prof, a.res)
        lst_file = cdir + "/list.txt"; open(lst_file, "w").write("\n".join(lst) + "\n")
        ini = subprocess.check_output(["find", prof, "-name", "TdInput.ini"], text=True).split("\n")[0]
        bind_cmd = [sys.executable, "-I", os.path.join(HERE, "gen_checkpoint_bindings.py"), ini, lst_file,
                    "--first-frame", str(a.first_frame), "--spacing", str(a.spacing)]
        if a.metal and not a.no_spin:
            bind_cmd.append("--spin")
        auto = subprocess.check_output(bind_cmd, text=True, stderr=subprocess.DEVNULL).strip()
        last_action_frame = max(int(tok.split(":")[0]) for tok in auto.split(";") if ":" in tok)
        rt_log = os.path.join(cdir, "rt.log")
        if a.metal:
            env = dict(os.environ, ME_PC_SHADER_CACHE=cache, ME_PC_AUTO_INPUT=auto, ME_PC_ASSUME_FOCUS="1",
                       ME_PC_MOUSE_CAPTURE="0", ME_PC_AUTO_KEYS=a.new_game_keys, ME_PC_USER=prof,
                       ME_PC_NO_INTRO="1", ME_PC_FULLSCREEN="0", ME_PC_VSYNC="0",
                       ME_PC_LOG_STALLS="1", ME_PC_LOG_PATH=rt_log)
            env.pop("ME_PC_DXVK", None)
        else:
            env = dict(os.environ, ME_PC_DXVK=a.dxvk, ME_PC_AUTO_INPUT=auto, ME_PC_ASSUME_FOCUS="1", ME_PC_MOUSE_CAPTURE="0", ME_PC_AUTO_KEYS=a.new_game_keys,
                       ME_PC_USER=prof, ME_PC_LOG_PATH=rt_log, DXVK_PIPELINE_WARM_FILE=warm, DXVK_PIPELINE_WARM_BOOT="0", DXVK_SHADER_CACHE_PATH=cache,
                       DXVK_LOG_LEVEL="info", MVK_CONFIG_LOG_LEVEL="2")
        if a.no_audio:
            env["ME_PC_NO_AUDIO"] = "1"
        settle_s = max(a.settle, 8) if a.metal else a.settle
        log = open(cdir + "/run.log", "wb")
        t0 = time.time()
        proc = subprocess.Popen([a.exe, a.binaries, a.deps, prof], env=env, stdout=log, stderr=subprocess.STDOUT)
        status, done_at = "ok", None
        act_re = re.compile(rb"ME_PC_AUTO_INPUT: frame (\d+):")
        while True:
            time.sleep(5)
            if proc.poll() is not None:
                status = "exited(%s)" % proc.returncode; break
            if time.time() - t0 > a.chunk_timeout:
                status = "timeout"; break
            data = open(cdir + "/run.log", "rb").read()
            if b"DEVICE_LOST" in data or b"host signal" in data or b"FATAL" in data:
                status = "device-lost" if b"DEVICE_LOST" in data else "crash"; break
            if done_at is None:
                frames = [int(x) for x in act_re.findall(data)]
                if frames and max(frames) >= last_action_frame:
                    done_at = time.time()
            if done_at and time.time() - done_at > settle_s:
                break
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(10)
            except subprocess.TimeoutExpired: proc.kill()
        log.close()
        # keep the log small: drop the noisy lines
        raw = open(cdir + "/run.log", "rb").read().split(b"\n")
        keep = [l for l in raw if not (l.startswith(b"  hle ") or l.startswith(b"stub ") or b"not found" in l)]
        open(cdir + "/run.log", "wb").write(b"\n".join(keep))
        if a.metal:
            n_rec, n_maps = count_metal_manifest(manifest)
            n_sh = count_metal_shaders(cache)
            cov = ""
        else:
            sh = read_shaders(warm + ".shaders")
            n_rec, n_maps = count_records(warm), 0
            n_sh = len(sh)
            cov = len(refcodes & set(sh.values())) if refcodes else ""
        w.writerow([i, m, len(lst), status, int(time.time() - t0), n_rec, n_sh, cov, len(refcodes) if refcodes else ""])
        prog.flush()
        map_info = (" (%d maps)" % n_maps) if a.metal else ""
        print("chunk %d/%d %s x%d: %s in %d s | records %d%s shaders %d%s" % (
            i, len(chunks), m, len(lst), status, time.time() - t0, n_rec, map_info, n_sh,
            (" | covers %d of %d reference shaders" % (cov, len(refcodes))) if cov != "" else ""), flush=True)
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
