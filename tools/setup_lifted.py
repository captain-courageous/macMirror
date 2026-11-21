#!/usr/bin/env python3
"""Prepares CRT/rebased DLLs, function catalogs (work/catalogs/), and lifted C modules (src/lifted/)."""
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")

def find_llvm_lib():
    env = os.environ.get("ME_PC_LLVM_LIB")
    if env and os.path.isdir(env):
        return env
    for cand in ("/opt/homebrew/opt/llvm/lib", "/usr/local/opt/llvm/lib"):
        if os.path.isdir(cand):
            return cand
    return "/opt/homebrew/opt/llvm/lib"

LLVM_LIB = find_llvm_lib()

GAME_ROOT = os.path.join(ROOT, "work", "game_data")
GAME_BIN = os.path.join(GAME_ROOT, "Binaries")
VCREDIST = os.path.join(GAME_ROOT, "__redist", "MSVC2005", "vcredist_x86.exe")
DEPS_DIR = os.path.join(ROOT, "work", "pc-deps")
REBASED_DIR = os.path.join(DEPS_DIR, "rebased")
CATALOGS_DIR = os.path.join(ROOT, "work", "catalogs")
LIFTED_DIR = os.path.join(ROOT, "src", "lifted")

REBASE_TABLE = [
    ("PhysXLoader", "20000000"),
    ("PhysXCore", "21000000"),
    ("NxCooking", "23000000"),
    ("PhysXExtensions", "24000000"),
    ("NxCharacter", "25000000"),
    ("ogg", "26000000"),
    ("vorbis", "27000000"),
    ("vorbisfile", "28000000"),
    ("vorbisenc", "29000000"),
    ("libresample", "2a000000"),
    ("binkw32", "2b000000"),
]

MODULES = [
    ("msvcr80", os.path.join(DEPS_DIR, "msvcr80.dll"), os.path.join(CATALOGS_DIR, "msvcr80", "catalog.json"), "msvcr80.dll", "deps", 1),
    ("msvcp80", os.path.join(DEPS_DIR, "msvcp80.dll"), os.path.join(CATALOGS_DIR, "msvcp80", "catalog.json"), "msvcp80.dll", "deps", 1),
    ("physxloader", os.path.join(REBASED_DIR, "PhysXLoader.dll"), os.path.join(CATALOGS_DIR, "physxloader", "catalog.json"), "physxloader.dll", "rebased", 0),
    ("physxcore", os.path.join(REBASED_DIR, "PhysXCore.dll"), os.path.join(CATALOGS_DIR, "physxcore", "catalog.json"), "physxcore.dll", "rebased", 0),
    ("nxcooking", os.path.join(REBASED_DIR, "NxCooking.dll"), os.path.join(CATALOGS_DIR, "nxcooking", "catalog.json"), "nxcooking.dll", "rebased", 0),
    ("physxextensions", os.path.join(REBASED_DIR, "PhysXExtensions.dll"), os.path.join(CATALOGS_DIR, "physxextensions", "catalog.json"), "physxextensions.dll", "rebased", 0),
    ("nxcharacter", os.path.join(REBASED_DIR, "NxCharacter.dll"), os.path.join(CATALOGS_DIR, "nxcharacter", "catalog.json"), "nxcharacter.dll", "rebased", 0),
    ("ogg", os.path.join(REBASED_DIR, "ogg.dll"), os.path.join(CATALOGS_DIR, "ogg", "catalog.json"), "ogg.dll", "rebased", 0),
    ("vorbis", os.path.join(REBASED_DIR, "vorbis.dll"), os.path.join(CATALOGS_DIR, "vorbis", "catalog.json"), "vorbis.dll", "rebased", 0),
    ("vorbisfile", os.path.join(REBASED_DIR, "vorbisfile.dll"), os.path.join(CATALOGS_DIR, "vorbisfile", "catalog.json"), "vorbisfile.dll", "rebased", 0),
    ("vorbisenc", os.path.join(REBASED_DIR, "vorbisenc.dll"), os.path.join(CATALOGS_DIR, "vorbisenc", "catalog.json"), "vorbisenc.dll", "rebased", 0),
    ("libresample", os.path.join(REBASED_DIR, "libresample.dll"), os.path.join(CATALOGS_DIR, "libresample", "catalog.json"), "libresample.dll", "rebased", 0),
    ("binkw32", os.path.join(REBASED_DIR, "binkw32.dll"), os.path.join(CATALOGS_DIR, "binkw32", "catalog.json"), "binkw32.dll", "rebased", 0),
    ("game", os.path.join(GAME_BIN, "MirrorsEdge.exe"), os.path.join(CATALOGS_DIR, "game", "catalog.json"), "mirrorsedge.exe", "game", 0),
]


def run(cmd):
    subprocess.run(cmd, check=True)


def find_vcredist():
    if os.path.isfile(VCREDIST):
        return VCREDIST
    for root, _, files in os.walk(GAME_ROOT):
        for fn in files:
            if fn.lower() == "vcredist_x86.exe":
                return os.path.join(root, fn)
    return None


def extract_crt():
    r80 = os.path.join(DEPS_DIR, "msvcr80.dll")
    p80 = os.path.join(DEPS_DIR, "msvcp80.dll")
    if os.path.exists(r80) and os.path.exists(p80):
        print("[1/4] CRT DLLs already present.")
        return
    vc_exe = find_vcredist()
    if not vc_exe:
        raise FileNotFoundError(
            f"vcredist_x86.exe not found under {GAME_ROOT} and CRT DLLs missing in {DEPS_DIR}"
        )
    print(f"[1/4] Extracting msvcr80.dll and msvcp80.dll from {vc_exe}...")
    os.makedirs(DEPS_DIR, exist_ok=True)
    extractor_src = os.path.join(TOOLS, "extract_vcredist.c")
    extractor_bin = os.path.join(ROOT, "work", "extract_vcredist")
    if os.path.exists(extractor_src) and shutil.which("clang"):
        if not os.path.exists(extractor_bin) or os.path.getmtime(extractor_src) > os.path.getmtime(extractor_bin):
            run(["clang", "-O2", "-lz", "-o", extractor_bin, extractor_src])
        run([extractor_bin, vc_exe, DEPS_DIR])
    else:
        tmp = os.path.join(ROOT, "work", "tmp_vcredist")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(os.path.join(tmp, "u", "msi"), exist_ok=True)
        run(["cabextract", "-q", "-d", tmp, vc_exe])
        inner_exe = next(
            os.path.join(tmp, f) for f in os.listdir(tmp) if f.lower().endswith(".exe")
        )
        run(["cabextract", "-q", "-d", os.path.join(tmp, "u"), inner_exe])
        msi = next(
            os.path.join(tmp, "u", f) for f in os.listdir(os.path.join(tmp, "u")) if f.lower().endswith(".msi")
        )
        run(["msiextract", "-C", os.path.join(tmp, "u", "msi"), msi])
        for root, _, files in os.walk(os.path.join(tmp, "u", "msi")):
            for fn in files:
                low = fn.lower()
                if low == "msvcr80.dll" and not os.path.exists(r80):
                    shutil.copy2(os.path.join(root, fn), r80)
                elif low == "msvcp80.dll" and not os.path.exists(p80):
                    shutil.copy2(os.path.join(root, fn), p80)
        shutil.rmtree(tmp, ignore_errors=True)
    if not (os.path.exists(r80) and os.path.exists(p80)):
        raise FileNotFoundError("Failed to extract msvcr80.dll / msvcp80.dll from vcredist")


def find_ci_entry(directory, name_lower, want_dir=False):
    if not os.path.isdir(directory):
        return None
    for fn in os.listdir(directory):
        if fn.lower() == name_lower:
            p = os.path.join(directory, fn)
            if want_dir and os.path.isdir(p):
                return p
            if not want_dir and os.path.isfile(p):
                return p
    return None


def find_game_dll(name):
    target = f"{name}.dll".lower()
    p = find_ci_entry(GAME_BIN, target)
    if p:
        return p
    if name.lower() == "physxloader":
        plocal = find_ci_entry(GAME_BIN, "physxlocal", want_dir=True)
        if plocal:
            p = find_ci_entry(plocal, "physxloader.dll")
            if p:
                return p
    if name.lower() == "physxcore":
        p = find_ci_entry(GAME_BIN, "physxcore.bak")
        if p:
            return p
    return os.path.join(GAME_BIN, f"{name}.dll")


def rebase_dlls():
    print("[2/4] Rebasing game DLLs...")
    os.makedirs(REBASED_DIR, exist_ok=True)
    for name, addr in REBASE_TABLE:
        dst = os.path.join(REBASED_DIR, f"{name}.dll")
        if os.path.exists(dst):
            continue
        src = find_game_dll(name)
        run([sys.executable, "-I", os.path.join(TOOLS, "rebase_pe.py"), src, dst, addr])


def build_catalog_one(mod):
    tag, binary, cat_path, *_ = mod
    out_dir = os.path.dirname(cat_path)
    if os.path.exists(cat_path):
        return tag
    os.makedirs(out_dir, exist_ok=True)
    run([sys.executable, "-I", os.path.join(TOOLS, "catalog.py"), binary, "--llvm", LLVM_LIB, "--out", out_dir])
    return tag


def lift_module_one(mod, jobs=1):
    tag, binary, cat_path, *_ = mod
    out_dir = os.path.join(LIFTED_DIR, tag)
    if os.path.exists(os.path.join(out_dir, f"dispatch_{tag}.c")):
        return tag
    os.makedirs(out_dir, exist_ok=True)
    run([sys.executable, "-I", os.path.join(TOOLS, "gen_module.py"), binary, "--llvm", LLVM_LIB,
         "--catalog", cat_path, "--name", tag, "--out", out_dir, "--jobs", str(jobs)])
    return tag


def write_module_table():
    lines = ['#include "rt.h"', ""]
    for tag, *_ in MODULES:
        lines.append(f"extern const X86FnEntry x86_fns_{tag}[];")
        lines.append(f"extern const uint32_t x86_fns_{tag}_count;")
    lines.append("")
    lines.append("const RtModuleSpec rt_module_specs[] = {")
    for tag, binary, _, name, kind, init in MODULES:
        lines.append(f'    {{"{name}", "{kind}", "{os.path.basename(binary)}", x86_fns_{tag}, &x86_fns_{tag}_count, {init}}},')
    lines.append("};")
    lines.append(f"const int rt_module_spec_count = {len(MODULES)};")
    os.makedirs(LIFTED_DIR, exist_ok=True)
    with open(os.path.join(LIFTED_DIR, "rt_modules_gen.c"), "w") as fp:
        fp.write("\n".join(lines) + "\n")

    cmake_lines = ["set(LIFTED_SOURCES", '    "${CMAKE_CURRENT_LIST_DIR}/rt_modules_gen.c"']
    physx_lines = ["set(PHYSXCORE_LIFTED"]
    for tag, *_ in MODULES:
        mod_dir = os.path.join(LIFTED_DIR, tag)
        for fn in sorted(os.listdir(mod_dir)):
            if fn.endswith(".c"):
                entry = f'    "${{CMAKE_CURRENT_LIST_DIR}}/{tag}/{fn}"'
                cmake_lines.append(entry)
                if tag == "physxcore":
                    physx_lines.append(entry)
    cmake_lines.append(")")
    physx_lines.append(")")
    with open(os.path.join(LIFTED_DIR, "lifted_sources.cmake"), "w") as fp:
        fp.write("\n".join(cmake_lines + [""] + physx_lines) + "\n")


def main():
    extract_crt()
    rebase_dlls()
    cpus = os.cpu_count() or 8
    workers = min(len(MODULES), cpus)
    # Schedule largest PE modules first so single-process catalog generation doesn't tail behind DLLs.
    catalog_order = sorted(MODULES, key=lambda m: (0 if m[0] == "game" else (1 if m[0] == "physxcore" else 2)))
    print(f"[3/4] Generating function catalogs (workers={workers})...")
    with cf.ThreadPoolExecutor(workers) as ex:
        for tag in ex.map(build_catalog_one, catalog_order):
            print(f"  catalog done: {tag}", flush=True)
    small_mods = [m for m in MODULES if m[0] != "game"]
    heavy_mods = [m for m in MODULES if m[0] == "game"]
    print(f"[4/4] Lifting modules to C into src/lifted/ (workers={workers})...")
    with cf.ThreadPoolExecutor(min(len(small_mods), cpus)) as ex:
        for tag in ex.map(lambda m: lift_module_one(m, jobs=1), small_mods):
            print(f"  lifted done: {tag}", flush=True)
    for mod in heavy_mods:
        tag = lift_module_one(mod, jobs=cpus)
        print(f"  lifted done: {tag}", flush=True)
    write_module_table()
    print("All lifted modules ready.")


if __name__ == "__main__":
    main()
