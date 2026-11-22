#!/usr/bin/env python3
"""Direct macOS ARM64 RunnerMetal builder using only Apple clang (zero CMake/Ninja)."""
import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIFTED_CACHE = os.path.join(
    os.path.expanduser("~"), "Library", "Application Support",
    "Mirror's Edge Native", "cache", "lifted-objs",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def lifted_env_hash(clang, rt_dir, lifted_base, physx_defs):
    h = hashlib.sha256()
    try:
        ver = subprocess.check_output([clang, "--version"], text=True).splitlines()[0]
    except Exception:
        ver = "clang"
    h.update(ver.encode("utf-8"))
    h.update(b"\0")
    for flag in lifted_base[1:] + physx_defs:
        norm = "<RT_DIR>" if flag == rt_dir else flag
        h.update(norm.encode("utf-8"))
        h.update(b"\0")
    for root, dirs, files in os.walk(rt_dir):
        dirs.sort()
        for fn in sorted(files):
            if fn.endswith((".h", ".hpp")):
                p = os.path.join(root, fn)
                h.update(os.path.relpath(p, rt_dir).encode("utf-8"))
                h.update(b"\0")
                with open(p, "rb") as fp:
                    h.update(fp.read())
                h.update(b"\0")
    return h.hexdigest()[:16]


def parse_cmake_list(text, var_name, lifted_dir):
    m = re.search(rf"set\(\s*{re.escape(var_name)}\s+(.*?)\)", text, re.S)
    if not m:
        return []
    out = []
    for raw in re.findall(r'"([^"]+)"|(\S+)', m.group(1)):
        tok = (raw[0] or raw[1]).strip()
        if not tok or tok.startswith("#"):
            continue
        tok = (tok.replace("${CMAKE_CURRENT_SOURCE_DIR}", ROOT)
                  .replace("${CMAKE_CURRENT_LIST_DIR}", lifted_dir)
                  .replace("${LIFTED_DIR}", lifted_dir))
        out.append(tok)
    return out


def compiler_for(path, clang, clangxx):
    ext = os.path.splitext(path)[1].lower()
    return clangxx if ext in (".cpp", ".cc", ".cxx", ".mm") else clang


def obj_path_for(obj_root, src):
    rel = os.path.relpath(src, ROOT).replace("..", "_ext")
    h = hashlib.sha1(rel.encode()).hexdigest()[:8]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", rel)
    return os.path.join(obj_root, f"{base}.{h}.o")


def compile_one(task):
    src, obj, cmd, use_depfile = task
    if use_depfile:
        dfile = obj + ".d"
        if os.path.exists(obj) and os.path.getmtime(obj) >= os.path.getmtime(src):
            if os.path.exists(dfile):
                return obj
        os.makedirs(os.path.dirname(obj), exist_ok=True)
        subprocess.run(cmd + ["-MMD", "-MF", dfile, "-c", src, "-o", obj], check=True)
        return obj
    if os.path.exists(obj):
        return obj
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    tmp = f"{obj}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        subprocess.run(cmd + ["-c", src, "-o", tmp], check=True)
        os.replace(tmp, obj)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return obj


def default_jobs():
    env_jobs = os.environ.get("MACMIRROR_JOBS", "").strip()
    if env_jobs:
        try:
            val = int(env_jobs)
            if val > 0:
                return val, None, False
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    try:
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        ram_gb = mem_bytes / (1024 ** 3)
        ram_cap = max(1, int(ram_gb // 3))
        jobs = min(cpus, ram_cap)
        return jobs, ram_gb, jobs < cpus
    except (OSError, ValueError, subprocess.SubprocessError):
        return cpus, None, False


def main():
    ap = argparse.ArgumentParser(description="Build RunnerMetal.app directly with clang.")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "build"))
    ap.add_argument("--sdl-framework-dir", help="Directory containing SDL3.framework")
    ap.add_argument("--sdl-include", help="Directory containing SDL3/SDL.h")
    ap.add_argument("--sdl-lib", help="Path to libSDL3.0.dylib / libSDL3.dylib")
    ap.add_argument("--lifted-cache-dir", default=DEFAULT_LIFTED_CACHE,
                    help="Persistent content-addressed object cache for lifted C files")
    ap.add_argument("--no-lifted-cache", action="store_true",
                    help="Disable persistent lifted object cache")
    ap.add_argument("--progress-file", help="Write '<done> <total> <phase> [detail]' progress updates")
    ap.add_argument("--jobs", type=int, default=0,
                    help="Parallel compile jobs (0 = auto via RAM / MACMIRROR_JOBS)")
    args = ap.parse_args()
    auto_jobs, ram_gb, capped_by_ram = default_jobs()
    args_jobs = max(1, args.jobs) if args.jobs > 0 else auto_jobs

    clang = shutil.which("clang") or subprocess.check_output(["xcrun", "-f", "clang"], text=True).strip()
    clangxx = shutil.which("clang++") or subprocess.check_output(["xcrun", "-f", "clang++"], text=True).strip()

    lifted_cmake = os.path.join(ROOT, "src", "lifted", "lifted_sources.cmake")
    if not os.path.exists(lifted_cmake):
        sys.exit(f"error: {lifted_cmake} not found; run tools/setup_lifted.py first")
    lifted_dir = os.path.join(ROOT, "src", "lifted")
    cmake_txt = open(lifted_cmake, "r", encoding="utf-8").read()
    lifted_sources = parse_cmake_list(cmake_txt, "LIFTED_SOURCES", lifted_dir)
    physxcore_set = set(os.path.abspath(p) for p in parse_cmake_list(cmake_txt, "PHYSXCORE_LIFTED", lifted_dir))

    # Resolve SDL3 (framework dir, explicit include+lib, or Homebrew fallback)
    sdl_cflags = []
    sdl_ldflags = []
    if args.sdl_framework_dir and os.path.isdir(os.path.join(args.sdl_framework_dir, "SDL3.framework")):
        fdir = os.path.abspath(args.sdl_framework_dir)
        sdl_cflags = ["-F", fdir]
        sdl_ldflags = ["-F", fdir, "-framework", "SDL3"]
    elif args.sdl_include and args.sdl_lib:
        sdl_cflags = ["-I", os.path.abspath(args.sdl_include)]
        sdl_ldflags = [os.path.abspath(args.sdl_lib)]
    else:
        for prefix in ("/opt/homebrew", "/usr/local"):
            inc = os.path.join(prefix, "opt", "sdl3", "include")
            lib = os.path.join(prefix, "opt", "sdl3", "lib", "libSDL3.0.dylib")
            if os.path.isdir(inc) and os.path.exists(lib):
                sdl_cflags = ["-I", inc]
                sdl_ldflags = [lib, "-Wl,-rpath," + os.path.dirname(lib)]
                break
        if not sdl_ldflags:
            sys.exit("error: SDL3 not found (pass --sdl-framework-dir or --sdl-include/--sdl-lib)")

    out_dir = os.path.abspath(args.out_dir)
    obj_root = os.path.join(out_dir, "objs")
    os.makedirs(obj_root, exist_ok=True)

    common = ["-arch", "arm64", "-mmacosx-version-min=14.0", "-O2", "-g0", "-DNDEBUG"]
    c_flags = [clang] + common + ["-std=gnu11"]
    cxx_flags = [clangxx] + common + ["-std=gnu++17"]

    rt_dir = os.path.join(ROOT, "src", "host_macos", "runtime")
    host_dir = os.path.join(ROOT, "src", "host_macos", "host")
    shatter_dir = os.path.join(ROOT, "src", "host_macos", "shatter")
    dxbc_dir = os.path.join(ROOT, "third_party", "dxbc-spirv")
    spvc_dir = os.path.join(ROOT, "third_party", "spirv-cross")
    metal_dir = os.path.join(ROOT, "src", "d3d9_metal")

    tasks = []
    lifted_objs = []
    lifted_cached_count = 0
    use_lifted_cache = (not args.no_lifted_cache) and bool(args.lifted_cache_dir)

    # 1. Lifted modules -> separate libGameLifted.dylib (user-built only; decouples RunnerMetal updates)
    lifted_base = c_flags + [
        "-I", rt_dir, "-fwrapv", "-ffp-contract=off", "-fno-strict-aliasing",
        "-Wno-unused-label", "-Wno-unused-variable", "-Wno-unused-but-set-variable", "-Wno-unused-function",
    ]
    physx_defs = ["-DX86_RESTRICT=restrict", "-DX86_FLOAT_PLAIN=1", "-DX86_X87_LOCAL=1", "-DX86_X87_SLOTS=1"]
    lifted_env_dir = None
    expected_cached = set()
    cached_dylib = None
    dylib_cache_hit = False
    lifted_bundle_hash = None
    if use_lifted_cache:
        env_tag = lifted_env_hash(clang, rt_dir, lifted_base, physx_defs)
        cache_root = os.path.abspath(args.lifted_cache_dir)
        lifted_env_dir = os.path.join(cache_root, env_tag)
        bh = hashlib.sha256(env_tag.encode("utf-8"))
        src_entries = []
        for src in lifted_sources:
            rel = os.path.relpath(src, lifted_dir)
            fsha = sha256_file(src)[:16]
            bh.update(rel.encode("utf-8") + b"\0" + fsha.encode("utf-8") + b"\0")
            src_entries.append((src, rel, fsha))
        lifted_bundle_hash = bh.hexdigest()[:16]
        cached_dylib = os.path.join(cache_root, "dylibs", f"libGameLifted.{lifted_bundle_hash}.dylib")
        if os.path.exists(cached_dylib):
            dylib_cache_hit = True
            lifted_cached_count = len(lifted_sources)
        else:
            for src, rel, fsha in src_entries:
                flags = lifted_base + (physx_defs if os.path.abspath(src) in physxcore_set else [])
                slug = re.sub(r"[^A-Za-z0-9._-]", "_", rel)
                c_obj = os.path.join(lifted_env_dir, f"{slug}.{fsha}.o")
                expected_cached.add(c_obj)
                lifted_objs.append(c_obj)
                if os.path.exists(c_obj):
                    lifted_cached_count += 1
                else:
                    tasks.append((src, c_obj, flags, False))
    else:
        for src in lifted_sources:
            flags = lifted_base + (physx_defs if os.path.abspath(src) in physxcore_set else [])
            obj = obj_path_for(obj_root, src)
            lifted_objs.append(obj)
            tasks.append((src, obj, flags, True))

    host_tasks_start = len(tasks)

    # 2. Shatter C++
    shatter_sources = [
        os.path.join(shatter_dir, "solver", "fluid", "fluid.cpp"),
        os.path.join(shatter_dir, "solver", "cloth", "cloth.cpp"),
        os.path.join(shatter_dir, "solver", "cloth", "cloth_attach.cpp"),
        os.path.join(shatter_dir, "solver", "cloth", "cloth_edgeshape.cpp"),
        os.path.join(shatter_dir, "solver", "cloth", "cloth_mesh.cpp"),
        os.path.join(shatter_dir, "solver", "cloth", "cloth_self.cpp"),
        os.path.join(shatter_dir, "solver", "cloth", "cloth_selftri.cpp"),
        os.path.join(shatter_dir, "solver", "cloth", "cloth_state.cpp"),
        os.path.join(shatter_dir, "shim", "fluid_ours.cpp"),
        os.path.join(shatter_dir, "shim", "cloth_ours.cpp"),
        os.path.join(shatter_dir, "shim", "scene_read.cpp"),
        os.path.join(shatter_dir, "shim", "coverage.cpp"),
        os.path.join(shatter_dir, "port", "bridge.cpp"),
        os.path.join(shatter_dir, "port", "game_hooks_port.cpp"),
    ]
    shatter_flags = cxx_flags + [
        "-I", rt_dir, "-I", os.path.join(shatter_dir, "port"),
        "-I", os.path.join(shatter_dir, "port", "neon"), "-I", os.path.join(shatter_dir, "shim"),
        "-fwrapv", "-ffp-contract=off", "-fno-strict-aliasing",
    ]
    for src in shatter_sources:
        tasks.append((src, obj_path_for(obj_root, src), shatter_flags, True))

    # 3. dxbc-spirv (MIT)
    dxbc_sources = [
        "sm3/sm3_types.cpp", "sm3/sm3_parser.cpp", "sm3/sm3_disasm.cpp",
        "sm3/sm3_converter.cpp", "sm3/sm3_io_map.cpp", "sm3/sm3_registers.cpp", "sm3/sm3_resources.cpp",
        "spirv/spirv_builder.cpp", "spirv/spirv_mapping.cpp",
        "ir/ir.cpp", "ir/ir_builder.cpp", "ir/ir_disasm.cpp", "ir/ir_divergence.cpp",
        "ir/ir_dominance.cpp", "ir/ir_legalize.cpp", "ir/ir_serialize.cpp", "ir/ir_utils.cpp",
        "ir/passes/ir_pass_arithmetic.cpp", "ir/passes/ir_pass_buffer_kind.cpp",
        "ir/passes/ir_pass_cfg_cleanup.cpp", "ir/passes/ir_pass_cfg_convert.cpp",
        "ir/passes/ir_pass_cse.cpp", "ir/passes/ir_pass_derivative.cpp",
        "ir/passes/ir_pass_descriptor_indexing.cpp", "ir/passes/ir_pass_function.cpp",
        "ir/passes/ir_pass_lower_consume.cpp", "ir/passes/ir_pass_lower_io.cpp",
        "ir/passes/ir_pass_lower_min16.cpp", "ir/passes/ir_pass_propagate_resource_types.cpp",
        "ir/passes/ir_pass_propagate_types.cpp", "ir/passes/ir_pass_remove_unused.cpp",
        "ir/passes/ir_pass_scalarize.cpp", "ir/passes/ir_pass_scratch.cpp",
        "ir/passes/ir_pass_ssa.cpp", "ir/passes/ir_pass_sync.cpp",
        "util/util_float16.cpp", "util/util_log.cpp", "util/util_md5.cpp", "util/util_swizzle.cpp",
    ]
    dxbc_flags = cxx_flags + [
        "-I", dxbc_dir, "-DDXBC_SPV_ENABLE_SM3", "-DDXBC_SPV_ENABLE_SPIRV",
        "-Wno-unused-parameter", "-Wno-missing-field-initializers", "-Wno-c++20-extensions",
    ]
    for rel in dxbc_sources:
        src = os.path.join(dxbc_dir, rel)
        tasks.append((src, obj_path_for(obj_root, src), dxbc_flags, True))

    # 4. SPIRV-Cross (Apache-2.0 / MIT)
    spvc_sources = [
        "src/spirv_cross.cpp", "src/spirv_parser.cpp", "src/spirv_cross_parsed_ir.cpp",
        "src/spirv_cfg.cpp", "src/spirv_glsl.cpp", "src/spirv_msl.cpp",
    ]
    spvc_flags = cxx_flags + [
        "-I", os.path.join(spvc_dir, "include"),
        "-I", os.path.join(spvc_dir, "include", "spirv_cross"),
        "-Wno-unused-parameter",
    ]
    for rel in spvc_sources:
        src = os.path.join(spvc_dir, rel)
        tasks.append((src, obj_path_for(obj_root, src), spvc_flags, True))

    # 5. Host Runtime + Native Metal Backend
    rt_sources = [
        os.path.join(rt_dir, "rt_core.c"), os.path.join(rt_dir, "rt_kernel32.c"),
        os.path.join(rt_dir, "rt_sync.c"), os.path.join(rt_dir, "rt_files.c"),
        os.path.join(rt_dir, "rt_user32.c"), os.path.join(rt_dir, "rt_d3d9.c"),
        os.path.join(rt_dir, "rt_d3d9_gen.c"), os.path.join(rt_dir, "rt_physx_support.c"),
        os.path.join(rt_dir, "rt_dinput.c"), os.path.join(rt_dir, "rt_ue3.c"),
        os.path.join(rt_dir, "rt_openal.c"), os.path.join(rt_dir, "rt_stubs.c"),
        os.path.join(rt_dir, "rt_main.c"), os.path.join(host_dir, "rt_host_macos.c"),
    ]
    rt_flags = c_flags + ["-I", rt_dir, "-DRT_SHATTER=1", "-fwrapv", "-ffp-contract=off", "-fno-strict-aliasing"] + sdl_cflags
    for src in rt_sources:
        flags = rt_flags + (["-Dmain=me_pc_main"] if src.endswith("rt_main.c") else [])
        tasks.append((src, obj_path_for(obj_root, src), flags, True))

    metal_sources = [
        os.path.join(metal_dir, "core", "mtl_d3d9_entry.mm"),
        os.path.join(metal_dir, "core", "mtl_d3d9_device.mm"),
        os.path.join(metal_dir, "resources", "mtl_format.mm"),
        os.path.join(metal_dir, "resources", "mtl_resources.mm"),
        os.path.join(metal_dir, "pipeline", "mtl_pipeline_cache.mm"),
        os.path.join(metal_dir, "shaders", "mtl_shader_compiler.mm"),
    ]
    metal_flags = cxx_flags + [
        "-fobjc-arc", "-DRT_SHATTER=1",
        "-I", rt_dir,
        "-I", os.path.join(metal_dir, "include"),
        "-I", os.path.join(metal_dir, "core"),
        "-I", os.path.join(metal_dir, "resources"),
        "-I", os.path.join(metal_dir, "pipeline"),
        "-I", os.path.join(metal_dir, "shaders"),
        "-I", dxbc_dir,
        "-I", os.path.join(spvc_dir, "include"),
    ] + sdl_cflags
    for src in metal_sources:
        tasks.append((src, obj_path_for(obj_root, src), metal_flags, True))

    def write_progress(done, total, phase, detail=""):
        if not args.progress_file:
            return
        tmp = f"{args.progress_file}.tmp"
        line = f"{done} {total} {phase}" + (f" {detail}" if detail else "")
        try:
            with open(tmp, "w", encoding="utf-8") as pf:
                pf.write(line + "\n")
            os.replace(tmp, args.progress_file)
        except OSError:
            pass

    total_tasks = len(tasks)
    sizes = [max(1, os.path.getsize(t[0])) if os.path.exists(t[0]) else 1 for t in tasks]
    names = [os.path.basename(t[0]) for t in tasks]
    total_bytes = sum(sizes)
    done_bytes = 0
    done_count = 0
    in_flight = set()
    prog_lock = threading.Lock()

    def active_detail():
        if not in_flight:
            return f"{done_count}/{total_tasks}"
        slow_i = max(in_flight, key=lambda idx: sizes[idx])
        mb = sizes[slow_i] / (1024 * 1024)
        tag = f"{names[slow_i]}({mb:.1f}MB)" if mb >= 0.2 else names[slow_i]
        return f"{done_count}/{total_tasks}|{tag}"

    def emit_compiling_progress():
        pct = (done_bytes * 95) // total_bytes if total_bytes > 0 else 95
        write_progress(pct, 100, "compiling", active_detail())

    def run_task_idx(idx):
        nonlocal done_bytes, done_count
        with prog_lock:
            in_flight.add(idx)
            emit_compiling_progress()
        try:
            return compile_one(tasks[idx])
        finally:
            with prog_lock:
                in_flight.discard(idx)
                done_bytes += sizes[idx]
                done_count += 1
                emit_compiling_progress()

    job_note = f"{args_jobs} jobs"
    if args.jobs <= 0 and not os.environ.get("MACMIRROR_JOBS", "").strip() and capped_by_ram and ram_gb is not None:
        job_note += f" [RAM-capped from {os.cpu_count()} for ~{ram_gb:.0f}GB]"
    print(f"Building RunnerMetal ({total_tasks} to compile, {lifted_cached_count} lifted cached, {job_note})...")
    if lifted_cached_count == 0 and any(os.path.abspath(t[0]).startswith(os.path.abspath(lifted_dir)) for t in tasks):
        print("Note: Cold build compiles large lifted C shards (can take 20-90+ min on Apple Silicon); subsequent updates are cached.")
    write_progress(0, 100, "compiling", f"0/{total_tasks}")
    schedule_order = sorted(range(total_tasks), key=lambda i: (-sizes[i], i))
    fut_by_idx = {}
    with cf.ThreadPoolExecutor(max_workers=args_jobs) as pool:
        for idx in schedule_order:
            fut_by_idx[idx] = pool.submit(run_task_idx, idx)
        for fut in cf.as_completed(fut_by_idx.values()):
            fut.result()
    compiled_objs = [fut_by_idx[i].result() for i in range(total_tasks)]
    host_objs = compiled_objs[host_tasks_start:]
    write_progress(96, 100, "linking", f"{total_tasks}/{total_tasks}")

    if lifted_env_dir and not dylib_cache_hit and os.path.isdir(os.path.dirname(lifted_env_dir)):
        cache_root = os.path.dirname(lifted_env_dir)
        keep_name = os.path.basename(lifted_env_dir)
        for entry in os.listdir(cache_root):
            if entry not in (keep_name, "dylibs"):
                shutil.rmtree(os.path.join(cache_root, entry), ignore_errors=True)
        if os.path.isdir(lifted_env_dir):
            for entry in os.listdir(lifted_env_dir):
                full = os.path.join(lifted_env_dir, entry)
                if full not in expected_cached:
                    try:
                        os.remove(full)
                    except OSError:
                        pass

    app_dir = os.path.join(out_dir, "RunnerMetal.app")
    macos_dir = os.path.join(app_dir, "Contents", "MacOS")
    fw_dir = os.path.join(app_dir, "Contents", "Frameworks")
    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(fw_dir, exist_ok=True)
    exe_out = os.path.join(macos_dir, "RunnerMetal")
    dylib_out = os.path.join(fw_dir, "libGameLifted.dylib")

    def copy_cow(src_p, dst_p):
        if os.path.exists(dst_p):
            os.remove(dst_p)
        if subprocess.run(["cp", "-c", src_p, dst_p], stderr=subprocess.DEVNULL).returncode != 0:
            shutil.copy2(src_p, dst_p)

    if dylib_cache_hit:
        copy_cow(cached_dylib, dylib_out)
    else:
        lifted_rsp = os.path.join(out_dir, "lifted_objs.rsp")
        with open(lifted_rsp, "w", encoding="utf-8") as f:
            for o in lifted_objs:
                f.write(f'"{o}"\n')
        dylib_cmd = [
            clang, "-arch", "arm64", "-mmacosx-version-min=14.0", "-dynamiclib",
            "-install_name", "@rpath/libGameLifted.dylib",
            "-Wl,-undefined,dynamic_lookup", "-Wl,-dead_strip",
            "-Wl,-exported_symbol,_rt_module_specs",
            "-Wl,-exported_symbol,_rt_module_spec_count",
            f"@{lifted_rsp}", "-o", dylib_out,
        ]
        subprocess.run(dylib_cmd, check=True)
        if cached_dylib:
            dylib_dir = os.path.dirname(cached_dylib)
            os.makedirs(dylib_dir, exist_ok=True)
            tmp_d = f"{cached_dylib}.tmp.{os.getpid()}"
            copy_cow(dylib_out, tmp_d)
            os.replace(tmp_d, cached_dylib)
            keep_d = os.path.basename(cached_dylib)
            for fn in os.listdir(dylib_dir):
                if fn != keep_d:
                    try:
                        os.remove(os.path.join(dylib_dir, fn))
                    except OSError:
                        pass

    rsp = os.path.join(out_dir, "host_objs.rsp")
    with open(rsp, "w", encoding="utf-8") as f:
        for o in host_objs:
            f.write(f'"{o}"\n')

    # Keep all host runtime symbols used by libGameLifted.dylib exported from RunnerMetal.
    lifted_host_exports = [
        "-Wl,-export_dynamic",
        "-Wl,-u,_x86_mem_base", "-Wl,-u,_x86_dispatch", "-Wl,-u,_x86_trap",
        "-Wl,-u,_x86_cpuid", "-Wl,-u,_x86_rdtsc",
        "-Wl,-u,_x86_lock_enter", "-Wl,-u,_x86_lock_leave",
        "-Wl,-u,_x86_resume_push", "-Wl,-u,_x86_resume_pop",
        "-Wl,-u,_rt_ue_script_call", "-Wl,-u,_rt_ue_font_page",
    ]
    link_cmd = [
        clangxx, "-arch", "arm64", "-mmacosx-version-min=14.0", "-dead_strip",
        *lifted_host_exports,
        f"@{rsp}", dylib_out, "-o", exe_out,
        *sdl_ldflags,
        "-framework", "Metal", "-framework", "QuartzCore",
        "-framework", "Cocoa", "-framework", "CoreAudio", "-framework", "Foundation",
        "-lm", "-lpthread",
        "-Wl,-stack_size,0x4000000",
        "-Wl,-rpath,@executable_path/../Frameworks",
    ]
    subprocess.run(link_cmd, check=True)
    shutil.copy2(os.path.join(ROOT, "src", "host_macos", "packaging", "Info.plist.in"),
                 os.path.join(app_dir, "Contents", "Info.plist"))
    seed_manifest = os.path.join(ROOT, "assets", "pso_manifest_v10.bin")
    if os.path.isfile(seed_manifest):
        res_dir = os.path.join(app_dir, "Contents", "Resources")
        os.makedirs(res_dir, exist_ok=True)
        shutil.copy2(seed_manifest, os.path.join(res_dir, "pso_manifest_v10.bin"))
    subprocess.run(["codesign", "--force", "--sign", "-", dylib_out], check=True)
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", app_dir], check=True)

    symlink = os.path.join(out_dir, "RunnerMetal")
    if os.path.lexists(symlink):
        os.unlink(symlink)
    os.symlink(os.path.join("RunnerMetal.app", "Contents", "MacOS", "RunnerMetal"), symlink)
    write_progress(100, 100, "done")
    print(f"Built {app_dir}")


if __name__ == "__main__":
    main()
