<div align="center">
  <img src="assets/icon.png" width="144" alt="Mirror's Edge — Native Apple Silicon (Metal)" />
  <h1>Mirror's Edge — Native Apple Silicon (Metal)</h1>
</div>

A pure native `arm64` macOS static recompilation port of **Mirror's Edge (2008, PC)** rendering directly through Apple **Metal** — with zero Vulkan, DXVK, MoltenVK, Wine, or Rosetta 2 at runtime.

> **Requires your own legally owned, DRM-free copy of Mirror's Edge (PC)** — GOG offline installer (`setup_mirrors_edge_*.exe`), Heroic/GOG folder, CrossOver/Whisky/Porting Kit bottle, DRM-free PC folder, or an existing `Mirror's Edge.app`. **No copyrighted game files, binaries, or lifted game code are hosted in this repository**; your own copy is statically lifted and compiled locally on your Mac.
>
> *Note: Steam copies whose `MirrorsEdge.exe` uses SteamStub DRM (`.bind` section) are not supported unless replaced with a DRM-free executable — the GOG offline installer is recommended.*

---

## Quick Install & Update (Terminal)

1. Open **Terminal** (`Cmd + Space`, type `Terminal`, press `Enter`).
2. Run:

```bash
curl -fsSL https://raw.githubusercontent.com/captain-courageous/macMirror/main/quick-install.sh | bash
```

Re-running the same command later detects your existing **`Mirror's Edge.app`** (`/Applications`, `~/Applications`, or `~/Downloads`), rebuilds/updates the app in-place, and keeps your saves, settings, shader cache, and game assets intact.

### Options & Game Sources

- **Zero Homebrew dependencies required:**
  Only **Apple Xcode Command Line Tools** (`xcode-select --install`) is needed on macOS. `quick-install.sh` compiles directly with `clang` (automatically throttling concurrency based on physical RAM; override with `--jobs` or `MACMIRROR_JOBS`) and bundles standalone `innoextract`, minimal X86 `libLLVM` disassembler, a built-in `vcredist_x86*.exe` CRT extractor, vendored `SPIRV-Cross` + `dxbc-spirv`, and official `SDL3.framework` (if SDL3 isn't already installed). First-time cold builds lift and compile ~270 C shards (~20–90+ minutes depending on Mac & RAM), while subsequent updates hit the persistent object cache and finish in under a minute.
- **Standalone GOG unpacking:**
  `quick-install.sh` embeds a statically linked `arm64` build of [`innoextract`](https://constexpr.org/innoextract/) (zlib license) directly inside the script, automatically finding and unpacking `setup_mirrors_edge_*.exe` (and its `.bin` data parts) in `~/Downloads`, `~/Desktop`, or `~/Games`.
- **Custom path / DMG / Language:**

```bash
# Point directly at a GOG installer .exe or game folder and build a drag-and-drop DMG:
./quick-install.sh --game ~/Downloads/setup_mirrors_edge_1.01_jpfix_\(41238\).exe --dmg

# Non-English GOG package language (defaults to en-US when multi-language):
./quick-install.sh --lang ja-JP
```

When finished, a self-contained, ad-hoc signed **`Mirror's Edge.app`** (with bundled `libSDL3.0.dylib`, original game icon, macOS Game Mode metadata, and baked assets) is placed in `~/Downloads` (or updated in-place).

---

## Performance (Mac mini M4, 16 GB)

Measured with the native Metal backend (`RunnerMetal`) and hybrid Shatter fluid/cloth physics active:

| Resolution | Frame Rate | Notes |
| --- | --- | --- |
| **1080p** (`1920x1080`) | **120 fps** | Locked `120 Hz` |
| **1440p** (`2560x1440`) | **120 fps** | Locked `120 Hz` |
| **4K** (`3840x2160`) | **~70–80 fps** | Full native 4K (`MaxAnisotropy=16`, hardware PCF shadows) |

Key optimizations behind these numbers:
- Persistent triple-buffered `MTLResourceStorageModeShared` ring buffers for uniforms/vertices and `MTLStorageModeMemoryless` + `MTLLoadActionDontCare` TBDR tile pass management.
- Fragment shader FP16 (`half`/`half3`/`half4`) promotion, hardware `depth2d::sample_compare` bilinear shadow filtering, and `MTLBinaryArchive` + async `.ir8`/`.msl` shader pre-warming (eliminating mid-gameplay PSO compilation hitches).
- Pre-bound per-frame ARM64 NEON Shatter cloth/fluid stepping and chunked audio/disk streaming.

---

## Saves, Settings & Metal Shader Cache

User saves, screenshots, `.ini` configs, runtime logs, and compiled Metal shader/PSO caches are stored in:

```text
~/Library/Application Support/Mirror's Edge Native/
```

Saves and compiled shaders stay intact when updating or moving `Mirror's Edge.app`.

### Shortcuts & Environment Options

- **`Cmd + Return`** — Toggle borderless fullscreen / windowed
- **`Cmd + G`** — Release / recapture mouse cursor
- `ME_PC_FULLSCREEN=0` — Launch windowed instead of fullscreen
- `ME_PC_VSYNC=0` — Disable VSync presentation
- `ME_PC_NO_INTRO=1` — Skip startup logo movie
- `ME_PC_RENDER_SCALE=0.75` — Internal render resolution scale (`0.25..1.0`)

---

## Credits & Architecture

This port stands on the work of several open-source projects:

- **[`RunnerRecompiled`](https://github.com/prasys/RunnerRecompiled)** (`prasys/RunnerRecompiled` macOS port, based on [`singlemate/RunnerRecompiled`](https://github.com/singlemate/RunnerRecompiled), *MIT License*) — Original x86-to-C static recompiler pipeline (`tools/`), Win32/PE guest runtime, SDL3/OpenAL/controller integration, and Shatter integration framework.
- **[`mtld3d`](https://github.com/athei/mtld3d)** (*Alexander Theissen, zlib License*) — Direct3D 9-on-Metal design and shader/pass architecture reference.
- **[`Shatter`](https://github.com/Arecsu/shatter)** (`v1.0.7`, *MIT License*) & **[`sse2neon`](https://github.com/DLTcollab/sse2neon)** (`v1.8.0`, *MIT License*) — Native fluid/cloth physics solvers compiled to Apple Silicon NEON (see [`src/host_macos/shatter/UPSTREAM.md`](src/host_macos/shatter/UPSTREAM.md)).
- **`dxbc-spirv`** (*Philip Rebohle, MIT License*) & **`SPIRV-Cross`** (*Apache-2.0 / MIT*) — SM3 bytecode -> IR -> SPIR-V -> `MSL 2.4` shader compiler in `src/d3d9_metal/`.
- **[`SDL3`](https://libsdl.org/)** (*zlib License*) — Windowing, borderless fullscreen, input, and game controller support.
- **[`innoextract`](https://constexpr.org/innoextract/)** (`v1.9`, *Daniel Scharrer, zlib License*) + **`Boost`** (*BSL-1.0*) + **`XZ Utils / liblzma`** (*0BSD*) — Bundled statically inside `quick-install.sh` for dependency-free GOG installer unpacking.

Full license texts and third-party notices are in [LICENSE.md](LICENSE.md) and `Mirror's Edge.app/Contents/Resources/Licenses/LICENSE.md`.
