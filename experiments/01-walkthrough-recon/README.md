# Experiment 01: rebuild a room from a walkthrough video

**Question:** can an ordinary walkthrough video of a still space become a 3D scene you can step out into, with honest coverage, on one consumer GPU and with permissively licensed tools?

**Status:** done. The kitchen (one clip) and the house (three of four clips joined) are both viewable. The fourth clip didn't connect; see below.

**Answer:** yes, within limits. Played back from the recording camera, both scenes are close to the footage when rendered on the GPU: held-out PSNR is 36 dB for the kitchen and 31.6 dB for the house. Stepping off the path works for moderate moves. Views that turn toward where the camera never looked are empty, and the space right around the camera path can fill with smeared floaters.

**Rendering:** the viewer has two renderers. **GPU** mode streams frames from a local worker that renders with gsplat, the renderer the scenes were trained with, at about 55–60 fps at 1080p on an RTX 4090. **Browser** mode (Spark) needs no worker but loses about 6 dB; it is the fallback when the worker isn't running.

## Inputs

Four 4K clips of one modern house by Kindel Media on Pexels, used under the [Pexels license](https://www.pexels.com/license/) (free use and modification, no attribution required, no resale of unedited copies or redistribution on stock sites). They are downloaded by `fetch_clips.py`, never committed.

| Clip | What it shows | Frames |
|---|---|---|
| 7578540 | Kitchen island, sideways arc | 470 @ 30 fps |
| 7578552 | Dining table with glass top and chandelier | 366 |
| 7578546 | Hallway into the living room | 627 |
| 7578547 | Living room through to the stairs | 636 |

## Pipeline

| Stage | Tool | License |
|---|---|---|
| Frames | ffmpeg | LGPL/GPL (external binary, not bundled) |
| Camera solve | COLMAP 4.2.0, incremental mapper, `RADIAL` lens model, one lens per clip | BSD |
| Splat training | gsplat 1.5.3 `simple_trainer.py mcmc --antialiased --pose-opt`, 1M splat cap, 30k steps | Apache-2.0 |
| Viewer | Spark 2.2.0 + three.js 0.180 (CDN) for browser rendering | MIT |
| GPU render worker | `gpu_render_server.py`: gsplat + torchvision nvJPEG over a loopback WebSocket | Apache-2.0 / BSD |
| Objects | `scan_objects.py`: SAM 3.1 tracks from a text prompt (`track_objects.py`), placed on the splats (`lift_objects.py`) | SAM License |

## Reproduce (Windows, NVIDIA GPU)

The core setup is below. For every stage (SAM 3.1, SEVA, Qwen3.5, imajev, ComfyUI with LTX-2.3), the model downloads and the full courtyard chain, see [docs/INSTALL.md](../../docs/INSTALL.md).

```bash
# 1. Environment (Python 3.10, PyTorch 2.4.1 + CUDA 12.4, prebuilt gsplat wheel)
uv venv --python 3.10 .venv-recon
uv pip install --python .venv-recon/Scripts/python.exe torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-recon/Scripts/python.exe "numpy<2" jaxtyping ninja rich viser "imageio[ffmpeg]" scikit-learn tqdm "torchmetrics[image]" opencv-python "tyro>=0.8.8" Pillow tensorboard tensorly pyyaml matplotlib splines "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" "nerfview @ git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
uv pip install --python .venv-recon/Scripts/python.exe --no-deps gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124
# fused-ssim compiles CUDA code: run inside a Visual Studio x64 developer prompt with CUDA_HOME set
uv pip install --python .venv-recon/Scripts/python.exe --no-build-isolation "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"

# 2. Tools
#    COLMAP 4.2.0: unzip colmap-x64-windows-cuda.zip from github.com/colmap/colmap/releases into tools/colmap
git clone --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git tools/gsplat-src
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/apply_windows_patches.py

# 3. Data and run
python experiments/01-walkthrough-recon/fetch_clips.py
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/reconstruct.py --scene kitchen --clips 7578540

# 4. View
python experiments/01-walkthrough-recon/viewer/serve.py 8790
#    optional, for full-quality GPU rendering (separate terminal):
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/gpu_render_server.py
#    open http://localhost:8790/?scene=kitchen/   (or ?scene=house/)
#    the header shows "GPU · <card>" when the worker is connected; G toggles GPU/browser rendering

# 5. Objects (optional; needs .venv-sam3, see "Finding the objects" below)
python experiments/01-walkthrough-recon/scan_objects.py --scene house-filled2 --prompts chair table sofa lamp
#    or type a prompt into the viewer's Scan box; O shows objects, click one to pick it out
```

The worker binds to 127.0.0.1 and only accepts the viewer's origin. `test_gpu_worker.py` checks image quality, timing, the origin check and latest-wins behaviour.

## Import your own walkthrough (one command)

After the setup above (and `.venv-sam3` for objects and the people check):

```bash
.venv-recon/Scripts/python.exe experiments/01-walkthrough-recon/import_walkthrough.py --name loft path/to/walkthrough.mp4 --wait-for-gpu
```

It runs every stage with the settings that worked on the house, logs to `work/<name>/import.log`, and resumes where it stopped if interrupted:

1. Pre-flight check (`preflight.py`).
2. Camera solve and splat training (`reconstruct.py`, about 7.5 frames per second of video).
3. Walls (`free_space.py`).
4. Three fill passes (`complete_difix.py` + `package_filled.py`).
5. Objects (`scan_objects.py`).
6. A before/after free-roam flythrough (`roam_flythrough.py`).
7. `work/<name>/import_report.json` and the link to `?scene=<name>-roam/`.

Several videos of one place can be passed together. Clips that share no views with the rest are reported and left out.

**Pre-flight check.** From frames sampled twice a second, it measures:
- **Movement:** whether the camera moves or only turns. It matches features between neighbouring samples; a homography explains a turning camera almost exactly, while a walking camera leaves parallax.
- **Blur, exposure, plain surfaces, cuts.**
- **People,** with SAM 3.

It stops on footage that can't work (under 4 s, under 1280 px wide, too dark, or a camera that moves in under 30% of the video) unless `--force`, and warns about the rest. The house clip 7578547 passes cleanly.

**What makes footage work:**
- nothing moving in the scene;
- walking through the space rather than standing and panning;
- sharp, well-lit frames, ideally 4K;
- one steady lens with no zooming;
- clips that overlap if there are several.

## Results: kitchen (clip 7578540)

Machine: Windows 10, RTX 4090 (24 GB), 16 cores, 62 GB RAM.

| Stage | Result | Time |
|---|---|---|
| Frames | 235 (every 2nd frame, 1920×1080) | seconds |
| Features + sequential matching | about 2,800 SIFT features per frame | 35 s |
| Camera solve | 235/235 frames registered, 26,433 points, 0.75 px mean reprojection error | 5.8 min |
| Splat training (30k steps) | 306,584 splats, 69 MB `.ply`, peak GPU memory 0.97 GB | 19.3 min |
| Held-out frames (every 8th, never trained on) | PSNR 34.5 dB, SSIM 0.971, LPIPS 0.102 | |
| Export + coverage | | 9 s |

The held-out frames sit between training frames on the same path, so these scores measure near-path quality, not free-viewpoint quality. For views off the path we only have visual checks so far:

- **Sidestep 0.45 units off the path at the same height:** clean. The island, range, hood and living room hold together.
- **1.2 units back and 0.9 up:** distant surfaces are recognisable, but the space around the recording path fills with smeared floaters.
- **Turned 180° from the recording direction:** close to empty. The camera never looked that way. Holes stay holes, which is the gap Infer is meant to fill.

**Coverage.** Counting frames turned out not to be useful here: 65% of the scene was seen in 20+ frames, yet it can still fall apart from new angles. The viewer's Coverage layer instead shows the *range of angles* each point was seen from (2·acos of the mean resultant length of the directions to the cameras that saw it; visibility from exact per-splat blending weights). Opacity-weighted: 22% seen by no frame (mostly splats buried behind surfaces), 15% seen within <5°, 21% 5–15°, 22% 15–30°, 19% 30°+. Near objects the camera arced around read cyan (wide range); the far living room and the patio beyond the glass read amber (one direction).

## Quality pass: why playback looked bad, and what fixed it

Method: `compare_renderers.py` renders the exact recorded camera with gsplat (the renderer the scene was trained with) and scores it against the viewer's pixel-exact capture (`__viewer.captureFrame`, saved by `viewer/serve.py`) and against the recorded frame. That separates reconstruction problems from viewer problems. `make_playback_video.py` turns `__viewer.captureSequence` output into a side-by-side playback video.

| Finding | Evidence | Status |
|---|---|---|
| Spark caps splats at 512 px radius and culls splats centred beyond 1.4× the view edge. Near walls and columns this punched holes in close surfaces: smeared columns, streaked floors | House hallway at 18 s: gsplat renders the training frame cleanly; the viewer smeared it | **Fixed** in viewer (`maxPixelRadius 4096`, `clipXY 3`); the hallway now renders solid |
| Training settings | Kitchen held-out PSNR 34.5 → 36.1 dB, SSIM 0.971 → 0.976, LPIPS 0.102 → 0.077; unseen share 22% → 7% | **Adopted** as `reconstruct.py` defaults: MCMC (1M cap), `--antialiased`, `--pose-opt` |
| The browser renderer was the bottleneck | Same cameras: gsplat 37.0 dB, Spark 30.3 dB. The v2 training gain barely showed in Spark | **Solved by GPU mode**: viewer captures in GPU mode score 36.0 dB (43 dB against raw gsplat; the gap is JPEG) |
| ↳ Spark's packed format clamps base colour to [0, 1] | 15% of colour channels are out of range; simulating the clamp in gsplat costs 1.7 dB. `SplatMesh` ignores `extSplats` | Open for browser mode |
| ↳ Spark's other quantisation (float16 centres, 8-bit scales and rotations, 6–8-bit SH) | Simulated in gsplat: ≤ 0.1 dB each | Not a factor |
| ↳ About 5 dB remaining | Not reproduced in gsplat simulations; likely 8-bit framebuffer blending and per-splat (vs per-tile) sorting in WebGL | Open for browser mode |
| The mirror hallway broke the reconstruction itself | v1: gsplat couldn't fit its own training frames there (48 frames below 20 dB, some at 9 dB), looking past a large mirror toward the unlinked kitchen | **Largely fixed by v2 training** (pose refinement + MCMC): 2 frames below 20 dB; the mirror now renders with its reflection |
| My first harness assumed the wrong image size (2054×1104 vs 2090×1112) | Inflated the viewer gap to ~8 dB; corrected | Fixed |

### Going vertical: why views from above blur, and what was tried

Probe views (`probe_views.py`) put the camera above the recording path, inside the room (fractions of the free headroom measured by `free_space.py`), looking down. There is no ground truth there, so they are judged side by side.

**Cause, measured.** Almost every visible splat is a paper-thin card or a needle (median long/short axis ratio about 4,200). Opacity-weighted, among the flat cards:

| | House | Kitchen | Random orientation |
|---|---|---|---|
| Facing the recording camera (normal within 30° of the view direction) | 35% | 42% | 13% |
| Lying flat (normal within 30° of vertical) | 13% | 9% | 13% |
| Standing upright | 66% | 74% | 50% |

Every surface was seen from chest height within a narrow cone (median half-angle about 8°), so upright cards facing the camera explain a floor as well as a real floor does. From above, you see those cards edge-on: streaks with gaps, which blend into blur.

**Tried:**

| Attempt | Recorded-camera quality | Views from above | Verdict |
|---|---|---|---|
| Carve floaters in seen-through space (`free_space.py --carve`, exact per-splat test; a voxel test was too coarse and removed 86% of splats) | −0.9 dB | no visible change | not used |
| Fade splats viewed outside their observed cone (`observed_directions.py` + worker `fade`) | −0.4 dB | some haze removed, but can reveal what's behind walls | available in the worker, off |
| Depth loss from COLMAP points (`--depth-loss`) | 31.5 vs 31.6 dB | no visible change | not used |
| 2DGS surfels with normal and distortion losses | 28.7 vs 31.6 dB | billboards fixed (facing camera 35% → 13%, lying flat 13% → 26%), but raised views not clearly better; new artifacts | not used |

Correct orientation alone doesn't bring back what the footage never saw: what the surfaces look like from above. Remaining options are dense depth and normal priors from a pretrained model (needs a model download), footage that varies in height, or generative fill (Infer), labelled as such.

**Filling the gaps: `complete_difix.py` (Difix3D+).** Rather than stopping at the edge, this stage generates the missing views and bakes them into the splats. Over four stages it raises copies of the recorded cameras to 15%, 30%, 45% and 60% of the free headroom, pitches them down and turns them ±25°. It renders each view, repairs it with Difix (`nvidia/difix_ref`, a single-step diffusion model guided by the nearest recorded frame), then fine-tunes the splats on the recorded frames plus the repairs (repairs weighted 0.5). House, on top of v2: 720 repaired views, 6,000 steps, 8.2 min on the 4090. Held-out PSNR from the recording cameras *rose*, from 31.5 to 32.6 dB (half-resolution measure), because cleaning up floaters helps those frames too. Raised views are clearly cleaner in the dining room and pillar hallway (solid pillars, clean floors up to 70% of headroom), somewhat better in the living room, and still messy at the top of the mirror hallway. Filling amplifies whatever the base model gets wrong: run on v1, whose mirror hallway was broken, it baked that fog into a flat grey wall. The filled scene (`viewer/house-filled`) declares its fill height (`navigation.maxRise`) and marks the filled content as inferred in `scene.json`.

A second pass on top (`--rises 0.3 0.6 0.75 0.9 --yaw 45 --every 3 --steps 2000`, 1,428 repaired views, 14.8 min) held the recording-camera score (32.7 dB) and reaches 90% of headroom. `raised_flythrough.py` flies the whole tour 0.35 of floor-to-ceiling (about 0.9 m) above the recording camera, looking down 30°. Side by side (`work/captures/house_raised_before_after.mp4`), the filled scene is clean where the unfilled one smears, including the mirror hallway. The one area still soft is the very top of the narrow stairwell, where repaired views disagree and training averages them. `viewer/house-filled2` is the current best house; its artificial wall sits at 0.5 of floor-to-ceiling, about 1.3 m above the recording height.

**A third pass for free roaming (`--cameras roam`).** The first two passes only generated views near the path. A free camera in the viewer goes off the path, floor to ceiling, and looks any way, and that's where fog and streaks remain. `--cameras roam` covers those views:

- **Where the views are.** Four stages of about 340 views each, anywhere in the observed free space within 0.3, 0.6, 1.0 and 1.5 times the median surface distance of the path, pitched from 45° down to 20° up.
- **What Difix is shown.** Each view is repaired against the training frame that sees most of the same surfaces from the most similar direction.
- **How it's trained in.** MCMC relocation moves faded splats to where detail is needed, a penalty discourages needle-shaped splats, and a final 24,000 steps train on all repairs together.

1,353 views took 28 min, and every repaired view is saved (`views/`, reusable with `--reuse`). "Repair distance" is how much Difix still changes a render: the mean absolute difference over 24 fixed free-roam probe views, 0–1 RGB, lower is cleaner.

| | house-filled2 | roam, 3,000 steps per stage | roam + 24,000 consolidation steps |
|---|---|---|---|
| Probe repair distance | 0.043 | 0.040 | 0.038 |
| Held-out PSNR from the recording cameras | 32.7 dB | 32.1 dB | 31.3 dB |

The first attempt used each repaired view only about once in training. The repairs themselves are clean: Difix removes the fog and streaks convincingly. They just barely got baked in. Training is the cheap part (65 s per 3,000 steps), so the consolidation phase uses each view about nine more times.

Side by side (`roam_flythrough.py`, `work/captures/house_roam_compare.mp4`):
- Most streaks and glare fog are gone.
- Views facing never-recorded areas (towards the kitchen, whose clip didn't join) become dim, room-coloured haze instead of black with shards.
- Everything off the path is still soft rather than photographic: repairs of neighbouring views disagree in detail, and training averages them.

From the recording cameras the 1.4 dB drop isn't visible side by side (`work/captures/house_roam_playback.jpg`). `viewer/house-roam` lifts the height limit to the ceiling (`navigation.maxRise` 1.0); the walls at the edge of recorded space stay. `package_filled.py` builds such a package from a completion run.

Running Difix on every rendered frame instead would give the sharp result directly, but it takes 0.59 s per frame at 1024×576 (0.19 s at 512×288, 6.2 GB of GPU memory). That's too slow while moving; it would only work for sharpening a still view.

**Finding the objects: `detect_objects.py` (SAM 3).** It finds chairs, sofas, tables, lamps and rugs from text prompts (`.venv-sam3`, which reuses the system PyTorch 2.13 + CUDA, plus `triton-windows`). This is the first half of object-level completion; the second half, rebuilding each object whole with SAM 3D Objects, waits on gated access.

**Following objects through a clip: `track_objects.py` (SAM 3.1).** SAM 3.1's multiplex video model takes a text prompt and follows every match through a clip under one id per object, which is what gathers each object's views for rebuilding. The checkpoint is the 1038lab/sam3 mirror's fp16 copy of Meta's `sam3.1_multiplex.pt`. `prepare_sam31_mirror.py` re-saves it and fills in the one weight it lacks, the text encoder's projection, which SAM 3.1 keeps unchanged from SAM 3. On dining clip 7578552 with "chair":

- 24 ids over 92 frames, 15 of them present in at least a quarter of the frames.
- 4 ids drop out and come back.
- Median frame-to-frame box overlap is 0.86.
- 19 s (4.8 fps), 11.7 GB of GPU memory.

Four settings matter on a 24 GB card:

- **Detector batch.** SAM 3.1's default of 16 frames plus the tracker went past 24 GB, spilled into system memory and ran at 0.3 fps; 4 frames runs at 4.8 fps (`--det-batch`; 2 peaks at 10.4 GB when the GPU is shared).
- **Past-frame memory.** The tracker keeps every past frame's masks and memory features on the GPU, so memory grows with clip length and the 157-frame clips stalled near the end. SAM 3.1's own `offload_output_to_cpu_for_eval` moves them to CPU memory; it's on.
- **Object cap.** It defaults to 16, and at that cap it stopped taking new chairs at frame 78.
- **Stray specks.** Mask pieces under 5% of an object's pixels, sometimes across the frame from the object, are dropped.

`apply_windows_patches.py` also lets its attention fall back from FlashAttention, which PyTorch's Windows builds lack. Output goes to `work/<work>/objects/tracks/<clip>_<prompt>/`: per-frame id maps and `tracks.json` (`--previews` adds an overlay video and contact sheet).

**Objects in the scene: `scan_objects.py`.** One command, or the viewer's Scan box, runs the object stage for a scene:

1. **Track** (`track_objects.py`, SAM 3 environment). Each clip the scene was built from is tracked for each prompt. Results are kept per clip and prompt, so a new prompt only tracks that, and they're shared by every variant of the scene built from the same frames (house, house-filled2, ...).
2. **Place in 3D** (`lift_objects.py`, reconstruction environment). Every splat is projected into every recorded frame, through the clip's COLMAP lens model so it lines up with the original frames the masks came from. A splat that is visible (within 5% of the rendered depth) and lands inside a tracked object's mask in at least half the frames where that object is present, and at least 3 of them, belongs to it. Tracks sharing most of their splats are one object seen twice: the same chair in two clips, or tracked again after the tracker lost it. Each splat goes to the object it matched most consistently; objects get an oriented box (one axis along the recording's up) and the recorded frame that shows them best.

It writes `objects.json`, `objects.bin` (object id per splat) and `objects/<clip>/<frame>.png` (the recorded-frame masks in object ids) into the viewer package, plus an `objects` entry in `scene.json`. Re-exporting a scene means placing again.

On the dining clip, 24 chair tracks became 17 chairs; placing took 13 s. Seven tracks were the same chair picked up twice. Rendered from each chair's best frame with that chair picked out, the lit-up splats match the recorded mask: the lounge chair, the dining chairs and the living-room armchair beyond the pillar. A few far-away chairs of a few hundred splats get stretched boxes (up to 2 m), because stray matches survive at that distance.

**In the viewer.** The Objects section lists what was found, grouped by prompt.
- **Show objects (O)** draws each object's box and tag, and tints its splats in GPU mode.
- **Clicking an object** (in the list or its tag) picks it out. It jumps to the recorded frame that shows it best; in Free mode it also turns the camera to face it from there. In GPU mode that object glows in its colour while everything else dims and greys.
- **The recorded-frame inset** overlays the same objects' masks, in the same colours, so each 3D box can be checked against the footage.
- **Esc** lets go of the picked-out object.
- **Scan** tracks a new prompt through every clip and places the results. Expect about two minutes for the house's 408 frames when the GPU is free (an estimate from the dining clip's 4.8 fps); serve.py runs one at a time and only accepts it from its own page. Other GPU work (a diffusion job, say) can slow it tenfold, because SAM 3.1 then spills out of GPU memory.

Browser mode shows boxes, tags and masks but can't tint splats.

**What the viewer does at the edge: artificial walls.** `free_space.py` marks the voxels each recorded frame saw straight through (rays up to 90% of the rendered depth; grid bounds 2.5× the median surface distance around the path). The viewer keeps the camera inside that space, sliding along its edge, and shows a fading cyan grid ("Edge of recorded space") where a move is stopped. A second limit keeps the camera within 0.2 of the floor-to-ceiling extent above the recording height (about 0.5 m), where probes stay clean in most of the house; quality depends more on how close you are to objects than on height (a pillar passed at close range smears even at 20% of headroom). **B** and **H** turn the limits off. Free mode is first-person: drag to look, W/S along the view, Q/E vertical.

### GPU render mode

| Measure (RTX 4090, both scenes loaded, 5.4 GB GPU memory) | 1280×720 | 1920×1080 | 2100×1190 |
|---|---|---|---|
| Kitchen round trip | 9.5 ms (~105 fps) | 15.9 ms (~63 fps) | 18.4 ms (~54 fps) |
| House round trip | 11.5 ms (~87 fps) | 18.6 ms (~54 fps) | 20.5 ms (~49 fps) |

gsplat renders in 3–6 ms and nvJPEG encodes in under 1 ms; most of the round trip is moving 130–330 KB frames through Python and the WebSocket, so pipelining two requests is the obvious next speed-up. The first frame of a scene takes about 1 s while the worker loads it into GPU memory. The viewer keeps one request in flight and never draws a frame older than the one on screen; the worker renders only the newest request per connection (a burst of 10 returned 2 frames: the first and the last). Coverage stays in the browser renderer, since it is a diagnostic view.

Playback videos (recorded | render): `work/captures/kitchen_browser_vs_gpu.mp4`, `work/captures/house_gpu.mp4`.

## Findings

- **Lens model matters.** COLMAP's `OPENCV` model with the global mapper converged to different horizontal and vertical focal lengths (772 vs 840 px), which would stretch the room about 9% vertically. A single-focal `RADIAL` model with the incremental mapper gave f = 770 px (about 102° horizontal field of view), registered all 235 frames, and reached 0.75 px mean reprojection error.
- **Windows fixes needed.** Two small bugs in the pinned example stack, plus SAM 3.1's hard FlashAttention requirement, all patched by `apply_windows_patches.py`.
- **The viewer needs to show where the evidence ends.** When the view turns toward unrecorded space, the screen just goes dark. It should say "outside recorded coverage" instead of leaving the user to guess.
- **Browser-mode frame rate is not yet measured.** The desktop app's browser pane pauses rendering while hidden. GPU-mode speed is measured above.

## Results: whole house (4 clips)

**What happened.** The four clips (526 frames at every 4th frame; exhaustive matching 14 min, incremental mapping 39 min) came back as *one* COLMAP model with all 526 frames registered and 0.72 px mean reprojection error. It looked like a success, but it wasn't:

- The kitchen clip shared **zero** 3D points with the other three. The hallway, living-room and dining clips were linked to each other (hallway & dining 6,091 shared points, hallway & living 2,543, living & dining 247).
- With nothing tying the two groups together, bundle adjustment shrank the three-clip group to a speck: all 408 of its frames sat within 0.0003 units of each other, next to a kitchen path about 190 units long.
- Training on that produced fog. Held-out PSNR was 20.7 dB overall: kitchen frames 28.1, the other clips 17.6–19.7, with their renders a radial smear. Their *training* frames were just as bad (17.5–19 dB), which pointed at the input rather than the model.

**Fix, now part of the pipeline.** `clip_connectivity.py` counts shared 3D points between clips after mapping. It keeps the largest connected group, deletes the other clips from the model, rescales what remains (here by 8,906×) and records everything in `work/<scene>/connectivity.json`. After the fix the three clips have camera paths of 2.19, 1.88 and 0.88 units, 79,708 points and 0.70 px reprojection error.

**What it means for the product.** This is the "cuts don't always line up" case, and COLMAP's own summary doesn't reveal it. The kitchen clip ends facing the living room, but with SIFT features that wasn't enough to link it. A walkthrough importer has to check connectivity itself and tell the user which shots it couldn't place, rather than silently producing a broken scene. Stronger learned features (COLMAP 4 supports ALIKED with LightGlue) are the next thing to try for linking the kitchen.

### Three connected clips (dining, hallway into living room, living room to stairs)

| Measure | Result |
|---|---|
| Frames | 408 (every 4th frame), one lens model per clip |
| Camera solve | 79,708 points, 0.70 px mean reprojection error (after rescale) |
| Splat training (30k steps) | 741,804 splats, peak GPU memory 1.9 GB, 13.6 min |
| Held-out frames | PSNR 28.4 dB, SSIM 0.938, LPIPS 0.188 |
| Per clip (held-out PSNR mean) | dining 28.4, hallway 27.9, living 28.8; a few frames fall to 9–11 dB (sun flare and exposure swings) |
| Coverage by angle (opacity-weighted) | 31% unseen, 11% <5°, 24% 5–15°, 25% 15–30°, 10% 30°+ |

**Visual checks in the viewer:**

- **From the recording camera (dining):** the glass table, chandelier and living area all rebuild. The other clips' camera paths cross the scene where they should, which confirms the three shots sit in one space.
- **Living room, 0.3 units off the path:** the sofas, coffee table, fireplace wall and windows are clear, but softer than the kitchen, with smears at the edges and near the camera.
- **Overhead:** blocked by the top side of the ceiling, which no camera saw and which renders as streaks. A floor-plan view needs a clipping plane that cuts the ceiling away.

Quality is lower than the single kitchen clip (28.4 vs 34.5 dB) because there are fewer frames per metre of path, a larger space, and exposure and flare changes between shots.

**v2 training (MCMC, anti-aliased, pose refinement; now the default):** held-out PSNR 28.4 → 31.6 dB, SSIM 0.938 → 0.953, LPIPS 0.188 → 0.119; 1M splats, 1.7 GB peak, 17 min. Median per-frame PSNR rose to 33.0 (hallway), 35.0 (living) and 30.2 (dining), and frames the model couldn't fit fell from 48 to 2, including the mirror hallway. Viewer packages: `viewer/house` (v2), `viewer/house-v1`; likewise `viewer/kitchen` (v2) and `viewer/kitchen-v1`.

## Results: courtyard house (a harder video, imported in one command)

Pexels 10959786, "Showcase of house" by Abdullah | 4K (Pexels license): one continuous 27 s take in 4K at 30 fps. It goes from an outdoor terrace through a courtyard and a glass sliding door into the living area, up an LED-lit staircase, along an upstairs hallway and into a bedroom. It was chosen to be harder than the Kindel Media house: outdoors and indoors, two floors, bright sun then dim interiors, plain walls and fast turns. `import_walkthrough.py --name courtyard` built it (`viewer/courtyard-roam`).

**Pre-flight check.** Passed with warnings:
- 22% of the video looks at plain surfaces;
- 22% of frames are blurry;
- 6 moments are cuts or very fast turns.

**Camera solve: SIFT broke it into pieces.** At 7.5 frames/s, SIFT joined only 91 of 205 frames into one piece: 0–12 s, outside to the door. Five pieces in all; the solve broke in two places:
- **The stairwell,** where frames had 130–850 features instead of 2,000+ and the chain of matches broke.
- **A blank wall** at 21 s (68–169 features).

What was tried:

| Attempt | Largest piece | Notes |
|---|---|---|
| SIFT, 7.5 frames/s | 91 of 205 (44%) | Matching every pair against every other found no links across the gaps |
| SIFT, 15 frames/s, peak threshold 0.002, overlap 25, relaxed solver | 241 of 410 (59%) | Up the stairs; upstairs still separate |
| ALIKED + LightGlue, 15 frames/s, overlap 25, relaxed solver | **406 of 410 (99%)** | The whole video; 4 frames of blank wall left out |

ALIKED extraction takes 27 s; LightGlue matching 15 min (SIFT: about 1 min). COLMAP runs both on ONNX Runtime, whose CUDA provider needs cuDNN 9: COLMAP's Windows build lacks it, so `reconstruct.py` puts PyTorch's `torch/lib` on the path. The importer now does this on its own: SIFT first, then the ALIKED retry if under 90% of frames join.

**Numbers:**
- **Training:** held-out PSNR 25.2 dB, SSIM 0.90, 1M splats (house: 31.6 dB). Sky, sun, exposure swings and a larger two-floor space.
- **Fill passes:** held-out PSNR at half resolution 27.9 → 28.3 → 28.5 → 28.1 dB (8.9, 16.4 and 25.2 min). The fills also clean up playback from the recording cameras.
- **Free-roam probe repair distance:** 0.057 → 0.051 (house: 0.043 → 0.038).

**Side by side** (`work/captures/courtyard_roam_compare.mp4`, `courtyard_playback.jpg`):
- **Playback from the recording camera:** the terrace and courtyard are near-photographic. On the LED staircase a large smear in the unfilled scene is gone after filling. The upstairs room improves a lot but stays soft.
- **Free roam outdoors:** shards become clean paving, planters and walls.
- **Free roam indoors:** mostly soft. The stairwell and upstairs hallway are narrow, and free-roam views end up close to plain walls the video only passed.
- **Sky:** black in views facing up. Nothing that far away gets reconstructed.

**Objects:** 39 (21 plants, 6 chairs, 6 tables, 3 sofas, 2 lamps, 1 bed). Tracking took 16 min; "plant" alone took 8.5 min at 17.8 GB, because there are so many plants.

**Time on the 4090:** about 2 hours of compute (solve about 20, training 20, fills 50, objects 17 min), not counting the failed SIFT attempts and waits for the GPU.

**What would help next:**
- A sky model (a background colour or environment map fitted to the frames), so outdoor views don't show black.
- A higher splat cap for a scene this size (1M now).
- For narrow interiors, keeping free-roam views further from walls, or letting the viewer's walls stop the camera closer to the path there.

## Estimating what the recording never saw (infer pass)

The fill passes only repair and sharpen views of surfaces the camera did see. Difix is a repair model, the fine-tuning only reshapes splats that already exist, and views mostly of empty space were skipped. So empty space (sky, what's behind a wall the camera never faced, the far side of the stairwell) stayed black or smeared. The infer pass generates that content and adds it to the scene as new splats, labelled as inferred.

`infer_unseen.py --scene courtyard-roam --run run_courtyard-roam --out courtyard-infer` (also stage 8 of `import_walkthrough.py`):

1. **Plan** (`unseen_plan.py`, reconstruction env). Every 18 training frames along the walk, the anchor's camera and a neighbour's are turned sideways (±60–150°), behind, up and down, plus one raised view looking down: 15 targets. The inputs are the anchor, frames ±4 and ±8 along the walk, and the frame elsewhere that sees the most of the same surfaces.
2. **Generate** (`unseen_generate.py`, `.venv-seva`). [Stable Virtual Camera (SEVA) v1.1](https://github.com/Stability-AI/stable-virtual-camera), 1.3B parameters, generates each group's 15 views together from its 6 inputs and cameras, at 1024×576. Because a group is generated together, its guesses agree with each other.
3. **Bake** (`unseen_bake.py`, reconstruction env). For each generated view:
   - MoGe-2 estimates depth, given the camera's field of view.
   - That depth is scaled to the rendered depth in a band around the empty parts of the view, where new surfaces have to meet known ones.
   - Empty pixels become new splats. Sky, which has no depth, goes on a far dome.

   Then everything is fine-tuned, with each generated view teaching only its empty parts plus a thin border. New splats are flagged in `inferred.npy`.
4. **Package** (`package_filled.py`). Writes `inferred.bin` (one byte per splat) and an `inferred` entry in `scene.json`. The viewer's **Show what's inferred (I)** tints those splats violet (GPU mode).

**Setup.** `.venv-seva` reuses the system Python 3.11 and PyTorch 2.13, and adds only `roma`, `imageio-ffmpeg` and `utils3d_moge` (all installed with `--no-deps`); MoGe also needs `utils3d_moge` in `.venv-recon`. The SEVA weights (5.06 GB, gated on Hugging Face) come under Stability's non-commercial licence, and so do its outputs. SEVA also needs the OpenCLIP ViT-H-14 image encoder (3.94 GB) and the Stable Diffusion 2.1 VAE; the official SD 2.1 repo is no longer downloadable, so `apply_windows_patches.py` points SEVA at the sd2-community mirror (identical checksums). The same script lets SEVA's attention fall back from FlashAttention on Windows. MoGe-2 is 1.32 GB.

**What went wrong on the way:**

| Attempt | What happened |
|---|---|
| Targets = the emptiest views anywhere in the walkable space | SEVA produced shattered, crystal-like images indoors. It guesses well next to what it is shown and falls apart far from it. Fixed by planning from recorded cameras. |
| Training on whole generated views; depth scale fitted to the whole view | 39 of 285 views lifted (365k new splats). MoGe and the scene disagreed by ~40%, since the known surfaces are blurry. SEVA's softer guesses also blurred the walkway's potted trees and indoor walls. Fixed by aligning depth in a band around the empty parts and teaching only the empty parts. |

**Courtyard result** (`viewer/courtyard-infer`, `work/captures/courtyard-infer_compare.mp4`):
- **Generation:** 285 views in 19 groups, 35 min (2.1 min and 16.2 GB per group).
- **Bake:** 7.8 min. 109 views lifted into 790k inferred splats, added to the existing 1M; 103 views had empty screen to teach.
- **Held-out PSNR from the recording cameras:** 28.14 → 28.85 dB (half resolution). The recorded content improved rather than degraded.
- **Side by side:**
  - Voids that were black (sky, over the courtyard walls, parts of the terrace) now show sky, trees, foliage and buildings.
  - Recorded areas look as before.
  - The new content is rough: low detail, and glassy shard artifacts where guesses from different groups, or the depth, disagree.
  - Indoor blur is unchanged, because the pass only fills empty screen.

**What would help next:**
- Generate longer trajectories per pass (SEVA's two-pass trajectory mode), so neighbouring groups agree.
- Keep only lifted points that several generated views agree on.
- A sharper generator than SEVA's 576p.
- A separate, gentle pass that lets the generated views sharpen blurry but covered regions.

## Identifying every object (a Jev-style classifier)

`identify_objects.py --scene courtyard-infer` names, checks, tracks and describes every object, with calibrated answers instead of free text. The classifier is [imajev-4b](https://github.com/mohit67890/imajev) (Apache-2.0), an open Jev-style decision model: Qwen3.5-4B with a LoRA adapter and a shipped calibration. It answers typed questions (yes/no, choice, score, multi-label) with probabilities and an explicit "can't tell" share. It runs from `tools/imajev` as a local HTTP server on port 8792 (PyTorch backend) while it's needed.

1. **List** (`objects_list.py`). Qwen3.5-4B, the same base without the adapter, names every kind of physical object in 20 keyframes spread over the walk. Names are lower-cased and made singular, and surfaces (wall, floor, sky...) are dropped. Courtyard: 82 kinds in 2.2 min.
2. **Verify.** imajev checks every name against every keyframe with one multi-label question per 32 names ("which of these can be seen?") and keeps kinds it confirms at 0.7 or more somewhere. Courtyard: 76 of 82 kept in 6.9 min. It rejected car, fence, ottoman, chimney, shutter and stair railing.
3. **Consolidate.** Verification is honest about presence, but tracking every variant separately is slow and makes duplicates. Variants are grouped under a general kind: olive tree and palm tree under "tree", framed artwork and wall art under "artwork", coffee table, side table, desk and nightstand under "table". Surfaces and building structure are dropped: floors, decking, paving, ceilings, beams, panels, doorways, the building. Courtyard: 76 names become 33 kinds; 16 dropped.
4. **Track and place.** SAM 3.1 follows each kind through the clips and `lift_objects.py` places the tracks on the splats (see "Finding the objects"). Courtyard: 27 new kinds in 48 min (6 were already tracked), 87 objects.
5. **Classify.** For each object, a crop of the recorded frame that shows it best (from its own mask) gets up to seven questions:
   - which of its kind's variants it is;
   - whether it really is that kind;
   - material (10 classes);
   - movable by hand;
   - rigid;
   - mass (4 classes);
   - whether the whole object is in view.

   Answers go into `objects.json` as `attributes`, each with its probabilities and "can't tell" share. Courtyard: 87 objects in 1.7 min. The viewer shows them on each object's tag and tooltip.

**How good the answers are.**
- **The kind is right:** olive tree vs palm tree, sliding door, side table vs coffee table, snake plant vs potted plant.
- **The "is it really a ..." check catches mislabels:**
  - "artwork 1" is more likely a television (is-artwork 0.56);
  - "window 2" is a mirror (0.08);
  - "table 2" is a bench (0.23).
- **Physical answers are weak on their own:** a sofa is 30% movable, a chair "over 50 kg". Real to sim (below) therefore weighs them against a prior for the kind rather than using them raw.

## Trust layer: where every pixel came from

A world model that fills in the unseen should say so, pixel by pixel. `trust_map.py` gives every splat a class:

| Class | Meaning | Courtyard (share of splats) |
|---|---|---|
| recorded | seen by 4+ recorded frames spanning 6°+ | 31% |
| recorded once | seen, but by few frames or from one direction: real colour, weaker depth | 8% |
| filled | never seen: placed by the fill passes (Difix-repaired novel views) | 16% |
| inferred | the infer pass (SEVA views of what the recording never saw) | 44% |
| rebuilt | an object's unseen sides, grown by its rebuild (LTX orbit) | – |
| completed | drawn by geometry-guided completion (below) | – |

"Seen by a frame" is exact rather than a projection guess. It is the splat's blending weight summed over the frame's pixels: the gradient of the rendered frame with respect to the splat's colour. A first version depth-tested projected centres; it called the back layers of surfaces "never seen" even though they showed in recorded views. The whole scene takes 6 s on the 4090.

The GPU worker renders the classes per pixel: class colour shaded by the splat's brightness, plus one channel per class, so every frame also reports how much of the screen each class covers. The viewer's **Show where each pixel came from (T)** shows this with a live legend.
- **From the recording's own cameras:** 92–99% recorded, as it should be.
- **Turned 150° away at the start of the walk:** 6% recorded, 8% recorded once, 26% filled, 60% inferred. That is exactly the view that looks blurry, and now the viewer says why.

## Metric scale

A camera solve has no scale. `metric_scale.py` compares MoGe-2's metric depth with the scene's rendered depth on 24 recorded frames (well-covered pixels, middle 60% of the frame). The per-frame median log ratio, taken as a median over frames, gives metres per unit. Courtyard: **1 unit = 7.78 m**, frame-to-frame spread ±17%.

Check against objects of known kind (box height along up):
- chairs 0.85–1.04 m;
- bed 0.72 m;
- tables 0.45–0.48 m.

The ones out of range are fragments, such as a chair piece 9 cm tall. Written to `work/<name>/metric.json`.

## Real to sim: MuJoCo and OpenUSD export

`export_sim.py --scene courtyard-infer` writes `sim/<scene>/`: `scene.xml` (MJCF), `scene.usda` (UsdPhysics), `splats/` and `sim.json`.
- **Frame:** metres, z up, z = 0 at the floor where the walk starts. The courtyard walk climbs about 3.8 m, so there is no single floor: every level comes from the static world, with a safety ground plane under the lowest.
- **Static world:** every surface splat that isn't a movable object, near the walk. Splats are voxelised at 6 cm, holes are closed, and the voxels are merged into boxes. Courtyard: 30 × 19 × 9 m in 7,066 boxes.
- **Bodies:** each movable object becomes a free rigid body.
  - Its collision shape is its own splats voxelised (20 voxels along its longest side) and merged into boxes, so a chair keeps its legs and a table the space under it. A first version used convex hulls; a hull fills that space and swallows the chairs tucked under a table or the cushions on a sofa, which would explode on the first step.
  - Splats that sit inside the static world (feet in the floor) are left out, so bodies don't start interpenetrating.
- **Physical properties:** imajev's answers weighed against a prior for the kind (naive Bayes), with both recorded in `sim.json`.
  - Movable: kind prior 80% × imajev 30% gives 63% for the sofa.
  - Mass classes are combined the same way: sofa 25 kg, chair 6.6 kg, bed 57 kg, side table 6.3 kg, vase 0.7 kg, coffee maker 3.9 kg.
  - Material sets friction and restitution: wood 0.5, fabric 0.8, glass 0.3...
- **Each body's splats** are written in its own frame and in metres (`splats/<body>.ply`), so a splat renderer can draw the simulated scene.
- **Courtyard:** 11 bodies (sofa, chair, bed, two tables, four potted plants, coffee maker, vase). 76 objects stay in the static world: built in (doors, windows, stairs, cabinets, planters, trees), fixed, fragments, or too big to be one movable thing.

`sim_run.py` settles the bodies (or shoves one) in MuJoCo, reports whether they settle and stay in the world, and writes the motion as rigid transforms in the scene frame, for replaying on the real splats. It needs the `mujoco` package, which isn't installed yet, so the export hasn't been stepped in a simulator.

## Rebuilding objects whole (the scroll-studio technique)

The recording sees most objects from one side. `rebuild_objects.py --scene courtyard-infer --out courtyard-objects` rebuilds each free-standing object from every side, using the technique from scroll-studio (a video model guided by rendered depth):

1. **Orbit** (`object_orbit.py`). The object's splats alone, less floaters away from its main body, along a 97-frame orbit that starts from the direction the recording saw it best and frames it at 55% of the height. Depth is encoded as scroll-studio's Blender pass does it (far-clamped log depth, near = white), and the first frame is the object on plain grey.
2. **Generate** (`object_generate.py`). LTX-2.3 22B distilled (fp8) with the union-control IC-LoRA, in the local ComfyUI, guided by that depth (blurred, strength 0.6), with the first frame as the start. The prompt comes from the object's kind and imajev's material.
3. **Fit** (`object_fit.py`).
   - The object's reconstructed splats stay fixed except their opacity, and 40k seeds through its box learn the rest.
   - The fit uses the recorded frames it appears in, masked by its SAM masks, plus the generated orbit, colour-matched to the recording and trusted only inside its projected box. Each orbit frame gets a small camera correction.
   - The check: PSNR against the recording inside the object's real masks, reconstructed object alone vs rebuilt object alone.
4. **Swap** (`object_replace.py`). The rebuilt objects replace their old splats in a new package; splats grown for unseen sides are flagged 2 ("rebuilt").
5. **Integrate.** 3,000 steps on every recorded training frame, whole frames. Only the rebuilt objects' splats train, plus the opacity of other splats inside their boxes: pieces of the object that SAM's masks missed, which may fade. Then the whole scene's held-out PSNR is compared before and after the swap.

**Which objects.** The largest free-standing objects (not built in, not plants) that imajev confirms are what their label says (at least 50%). imajev's "movable" is too cautious to choose by. On the courtyard, the confirmation gate left out "table 2", which imajev thinks is a bench (23%). Sofa 2 was left out by hand: its reconstruction is only the wooden frame.

**Results** (`viewer/courtyard-objects`, about 280 s per object on the 4090):

| Object | vs recording, reconstructed → rebuilt | Covers its real mask |
|---|---|---|
| sofa 1 | 19.6 → 31.3 dB | 92% → 100% |
| table 1 (side table) | 20.2 → 30.1 dB | 89% → 100% |
| vase 1 | 23.1 → 29.6 dB | 97% → 100% |
| bed 1 | 20.8 → 29.4 dB | 87% → 99% |
| chair 1 (two chairs) | 28.4 → 28.5 dB | 98% → 100% |
| coffee maker 1 | 18.5 → 26.9 dB | 90% → 100% |

The scene goes from 1.79M to 1.86M splats; 5% are now "rebuilt" in the trust map. **Whole scene, held out: 28.85 → 28.70 dB** (half resolution). The PSNR only checks that the rebuilt object still matches what was recorded (and fills its holes there). Nothing can score the sides nobody filmed, so `work/captures/courtyard_rebuild_isolated.jpg` shows each object alone from the front, side, back, other side and above, before and after:
- **Sofa 1:** from the side, back and above the reconstruction is broken fragments. Rebuilt, it is a complete sofa from every side: arms, legs, back frame, cushions.
- **Coffee maker:** side views go from smears to a solid body.
- **Vase:** solid from every side, but the wrong shape: LTX gave it horns and handles it doesn't have.
- **Chairs:** already well recorded; the rebuild is more complete but blobbier at the base.
- **Side table:** still messy.

Moving a rebuilt object in the viewer leaves no ghost behind, unlike the reconstructed sofa.

**What went wrong on the way:**

| Problem | Fix |
|---|---|
| "Studio turntable shot" in the prompt: LTX invented a turntable platform under the chair and the bed | Prompt asks for an orbit with the object resting on the floor; platforms, pedestals and turntables in the negative prompt; generated pixels only count inside the object's projected box |
| A stray shard in the first frame became a floating stick in every generated frame | Drop splats away from the object's main body (voxel components) before the orbit |
| The orbit kept the recording camera's distance: a vase was a few pixels tall | Frame the object, since it is rendered alone |
| An old fit's folder was keyed by an object id that changed when objects were placed again | Rebuild folders are matched to objects by kind, orbit centre and size |
| The per-object check (19.6 → 31.2 dB for the sofa) hid a whole-scene loss: held-out PSNR fell 28.85 → 25.82 dB. A hallway frame went 22.3 → 14.8 dB under a smear of chair splats: seeds had grown or drifted where no orbit frame looks (`work/captures/courtyard_rebuild_haze.jpg`) | Seeds stay inside the object's box and at most a quarter of its size. In recorded frames the object is penalised wherever it would stand in front of the rest of the scene outside its mask. The swap now reports the whole scene's held-out PSNR. That brought it back to 27.12 dB |
| Still 1.7 dB down: pieces of the sofa the masks missed doubled up with the rebuilt sofa, and the hallway frame (the chair isn't in it, so the object fit never saw it) kept its smear | The integration pass on whole recorded frames: 28.70 dB |

**Lesson:** a generated object has to be checked in the whole scene, not just inside its own mask.

**The limit:** a rebuild is only as good as the object it starts from.
- "Bed 1" is only its striped cover plus part of a lamp, so LTX drew a crumpled cover, not a bed.
- "Chair 1" is two chairs merged into one object.
- Sofa 2's reconstruction is only its frame, and LTX grew it into a boat-like shape.

Better segmentation (separate instances, whole objects) would help more than a better generator.

## Editable world

Pick an object in the viewer (Objects list or its tag) to get **Remove / Turn ⟲ ⟳ / Away / Toward / Left / Right / Reset** (Delete, `[` and `]` work too). The GPU worker applies the rigid transform (or hides the object) on its splats, and its box follows. Nothing is saved.

Two limits show straight away:
- Splats of an object that SAM's masks missed stay behind as a ghost (the sofa's armrest). Rebuilt objects (above) are complete, so they move cleanly.
- What was behind an object was never recorded. Moving it uncovers a hole, or the infer pass's estimate, and the trust view says which.

## Geometry-guided completion: a video model on the scene's own geometry

The infer pass (SEVA) filled what the recording never saw, but its guesses are soft shards. `complete_scene.py --scene courtyard-objects --out courtyard-complete` runs scroll-studio's technique at scene scale, and keeps the result honest:

1. **Paths** (`scene_paths.py`). At an anchor every 6 recorded frames, 11 turns (±30°...180°) are rendered small in the trust view. The chosen turns are the ones showing the most never-recorded pixels, and the anchors are at least 18 frames apart. Turns facing something too close (median depth under half the median surface distance) are rejected, and a near plane at 0.2 × that distance keeps floaters out of the guide. Each path holds on the recorded view for 12 frames, turns smoothly over 73, drifts a little toward the turn's end where free space allows, then holds.
   - Courtyard: only the outdoor start of the walk has large unseen areas (60–89% of the best turn); indoors, every anchor's best turn shows under 25% unrecorded. 6 paths (anchors at frames 2, 22, 50, 70, 91, 111).
   - Along each path: the scene's depth (scroll-studio encoding), its own render, and per pixel the share that was recorded.
2. **Caption** (`scene_caption.py`). Qwen3.5-4B describes the place in the recorded first frame, e.g. "An outdoor modern patio with minimalist wooden furniture, light gray cushions... under a dark slatted ceiling".
3. **Generate** (`scene_generate.py`). LTX-2.3 with the union-control IC-LoRA:
   - the first frame is the recorded frame;
   - the depth guide is blurred (σ 6) at strength 0.6;
   - the scene's own render at frame 96 is a soft keyframe (0.35);
   - the prompt is the caption plus the camera move.

   97 frames at 1024×576 take about 70 s per path (229 s when models load).
4. **Fit** (`scene_bake.py`). The scene trains on its 355 recorded frames and the 582 generated frames, with two rules:
   - a generated frame only teaches pixels the recording never saw (its target elsewhere is the render itself);
   - splats the trust map calls recorded, or recorded once, are frozen (664k of 1.86M).

   Generated frames are colour-matched to the scene where they overlap recorded pixels, and each gets a small camera correction. 12,000 steps, 8.5 min. Splats whose colour moved are flagged 3 ("completed" in the trust view).

**What went wrong on the way:**

| Attempt | What happened |
|---|---|
| Guide blurred at σ 10, strength 0.45; prompt = camera move only | LTX turned the terrace's unseen side into an indoor furniture showroom. The rough geometry behind the start of the walk isn't a strong enough guide on its own. Fixed with the caption and the end keyframe on the scene's own render, which keeps its layout (sky above, garden wall, planters) while LTX sharpens it. |
| Turns planned without a near plane | One turn walked the camera into splats a metre away and the depth guide was near-field noise. |

**Courtyard result** (`viewer/courtyard-complete`):
- **Held-out PSNR from the recording cameras: 28.70 → 29.04 dB.** Nothing recorded got worse; the unrecorded splats now fit around what was recorded.
- **Never-recorded pixels of the generated frames:** the scene matches them at 25.7 dB (was 18.3).
- **Trust map:** 29% recorded, 8% recorded once, 12% filled, 32% inferred, 3% rebuilt, 15% completed (281k splats changed).
- **Side by side** (`work/captures/courtyard_completion_v1.jpg`: before, after, trust view, from the first version of this fit): on the paths, brown and green smears become a garden with trees, planters, a wall and sky. Off the paths it partly carries over: two of four test turns go from smear to trees and sky, one stays a smear, one barely changes. The trust view marks 35–78% of these views "completed".
- **What it isn't:** photoreal. The completed garden is soft and dreamlike. LTX turns white smears into white patio furniture that probably isn't there, and one path (p03) drifts toward armchairs. A sharpness measure (variance of the Laplacian in never-recorded pixels) dropped, because the old shards counted as detail. It can't tell noise from structure, so no sharpness claim is made here.

**What would help next:**
- More, longer paths with overlap, so neighbouring paths agree.
- Letting completion add splats where the scene has none (it only re-colours and reshapes what the infer pass placed).
- A multi-view check that drops generated content paths disagree on.

The courtyard package now uses paths p00, p01, p02 and p05 only (`--skip-paths p03 p04`): p04 left a blue cast on the paving and p03 drifted toward armchairs. Held-out 28.98 dB.

## Geometry refinement: surfaces that hold up from any angle

A free-roam wander through the completed courtyard (`roam_flythrough.py`, 16 waypoints) showed that the main problem off the path isn't what the recording never saw. Surfaces it did see break up: glassy shards on the outdoor paving, long streaks on hallway walls. The splats are thin cards and needles that only look right from the recorded angles (see "Going vertical"). A COLMAP-point depth loss and 2DGS self-consistency didn't fix that on the house. The missing piece was a dense prior on what the surfaces are, and MoGe-2 (already downloaded for the infer pass) gives one per image.

`geometry_refine.py --scene courtyard-complete --out courtyard-refined --views run_courtyard-roam --generated --skip-paths p03 p04` fine-tunes the scene, same splats in the same order:
- **Recorded frames (355):** RGB as trained, plus MoGe-2 depth (scale-aligned to the scene per frame, so it only shapes) and MoGe-2 normals. They're compared with the rendered depth and with a rendered normal map, each splat's shortest axis turned toward the camera.
- **Repaired free-roam views (700 of the roam pass's 1,297 Difix repairs):** RGB and MoGe-2 normals, only where the trust map says the view shows recorded surfaces.
- **Generated completion paths (176 frames of p00, p01, p02, p05):** RGB, depth and normals only in pixels no frame recorded. Recorded splats are frozen on those steps.
- **A disc penalty:** each splat's thinnest axis relative to its middle one, with the middle axis held fixed in the penalty. No splat may grow past 1.5× its starting size.

15,000 steps, 9.4 min on the 4090 (MoGe-2 on 1,231 images first: 2 min).

| | completed | refined |
|---|---|---|
| Held-out PSNR from the recording cameras | 28.98 dB | 28.08 dB |
| Opacity share of splats lying flat (normal within 30° of up) | 17.5% | 28.6% |
| Splat size, median / 99th percentile (scene units) | 0.016 / 0.171 | 0.016 / 0.166 |
| Render time, 1909×1064 | 4.8 ms | 4.0 ms |

**Side by side** (`work/captures/courtyard_roam_refined.mp4`, `courtyard_refine_before_after.jpg`):
- Outdoors, the paving goes from glassy shards to flat stone with tile detail. Streaky glass and wall views become clean walls, and potted plants, the tree in its pot and the glass door come out cleanly.
- The view facing the unfilmed garden shows some of the completed structure but is still rough.
- Indoors (hallway, stairwell), views are about as soft as before.

It costs 0.9 dB from the recording cameras.

**What went wrong on the way:** the first version penalised thinnest / middle axis, and splats satisfied it by growing their middle axis. The 99th-percentile splat grew 15× and a generated view took 36× the tile work (1.2M → 43.5M intersections). A second round then filled the GPU and thrashed. Its roam sheet looked smoother, from blur rather than geometry, and its held-out loss was smaller (0.45 dB) for the same reason. With the middle axis held fixed and the size cap, splat sizes don't move.

**A colour polish wins back the recording cameras.** After the priors, 3,000 steps on the recorded frames train only colour (`--polish 3000`, `--polish-params sh0 shN`), with the new geometry and opacity held. 1 min.

| | completed | refined | refined + colour polish (`viewer/courtyard-final`) |
|---|---|---|---|
| Held-out PSNR from the recording cameras | 28.98 dB | 28.08 dB | 28.75 dB |
| Lying flat | 17.5% | 28.6% | 28.6% |

Off the path, `courtyard-final` looks like the refined scene (`work/captures/courtyard_roam_final.mp4`). Polishing opacity as well reached 29.30 dB, but brought back streaks on walls and a smear on a planter off the path: opacity revives the cards that only suit the recorded angles.

## Sharpening still views (Difix in the render worker)

Everything baked into the splats stays a little soft off the path, because repairs of neighbouring views disagree in detail and training averages them (see the roam pass). A single Difix repair doesn't have that problem: it is sharp. The GPU worker therefore repairs the view you stop on.

- **When:** the camera hasn't moved for 450 ms. The viewer asks for one more frame with `sharpen`, and moving again returns straight to live frames. **Sharpen still views (X)** turns it off.
- **How:** the worker picks the recorded frame that sees most of the view's surfaces from the most similar direction (the roam pass's choice). It runs Difix (`difix_ref`, half precision, at most 1280 px wide, resized back to the frame) with that frame as the reference. Difix loads in the background when the worker starts (about 1 min); frames render normally meanwhile.
- **Only where it's true:** views that are less than 50% recorded (recorded + recorded once in the trust map) are left alone. There, with nothing true to guide it, Difix invents: the unfilmed garden became an indoor wall with steps. Views with the trust view or object edits showing are left alone too.
- **Labelled:** the badge reads "Sharpened by Difix · repaired, not recorded", and its tooltip names the reference frame. Skipped views say why ("Not sharpened: only 26% of this view was recorded").
- **Speed:** about 1–1.5 s per view on the 4090 at 1280×720; 1.8 s round trip at 2.5 MP.

On the courtyard (`work/captures/courtyard_sharpen_test.jpg`), free-roam views of the olive tree and planters go from soft to photographic, and the living room gets clean edges. Difix still invents small things behind glass.

## Walking off the path: where it still turns to glass

A screen recording from the viewer (free camera, W from the start of the walk) found the next problem:
- **What happens:** a straight move forward from the terrace stays clean right up to the sofa. Past the sofa, turning toward the olive tree and back, the view turns to glass and fog.
- **What's in that view:** from there, half of it was never recorded (trust view). The sofa's rebuilt back is at 0.8–1.6 m (about 9,000 rebuilt splats fitted to an orbit at about 4 m), and completed ground surrounds it.
- **Why the paths didn't help:** the completion paths had only turned on the spot at recorded positions, so no generated frame came from there.

**Ruled out** (`work/review/userclip/`):
- Turning off view-dependent colour (spherical harmonics beyond degree 0): no change.
- Removing splats with almost no weight in any recorded frame: no change. The dark needles in front of walls are real, weighted splats of the wall and planter, sub-pixel at recording distance.
- Mip-Splatting's 3D smoothing filter applied after training: darkens the scene. Most splats are paper-thin, and its opacity compensation empties them; it only works when trained with it.
- A near plane or near fade: the fog isn't at the lens.

**Walk paths** (`scene_paths.py --mode walk`). From a recorded frame the camera walks along its heading, as far as free space allows (up to 1.4× the median surface distance, about 2.9 m), turning 90–180° over the second part to look back. Five paths, all on the terrace (anchors at frames 2, 22, 50, 70, 91), turning 120–150°. LTX's frames for them are coherent: past the sofa, looking back at the pergola, the garden and the boundary wall.

**Fitted all at once, they don't hold** (`viewer/courtyard-final2`, `work/captures/walk_p13.mp4`):
- Held-out PSNR from the recording cameras is 28.87 dB and never-recorded pixels match the generated frames at 24.3 dB (was 21.2).
- Along the walk itself, though, the look-back views go from glassy to foggy rather than to what LTX drew.
- The five paths were generated independently from the same scene and disagree where they overlap, and fitting them all averages the disagreement into blur. The fit could also only re-colour and reshape splats the infer pass had put at rough depths.

`complete_walk.py` completes the paths one at a time instead:
- each path's guides are rendered again from the scene as it now is;
- LTX is anchored to that render at frames 32, 64 and 96;
- the path's never-recorded pixels are lifted into new splats at MoGe-2 depth (aligned to the recorded surfaces in the same frame) before the fit.

**First run: worse, and the failure fed itself** (`viewer/courtyard-walk`, `work/captures/walk_p13_v2.mp4`):
- Recorded views that were clean before went hazy, and past the sofa the walk became a green blur.
- The lifted splats landed in space the recording shows as empty. Straight after lifting p10, before any fitting, held-out recorded views fell from 28.75 to 23.09 dB.
- The damage compounded. The next path's guides rendered that fog as near geometry and counted it as never recorded (p13: 82% of its views; even at its recorded start frame the never-recorded mask was mostly white). LTX then drew close-up foliage into it (`work/review/p13_gen_sheet.jpg`).
- It ended at 3.59M splats, 58% of them completed, with held-out 27.91 dB.

**Free space fixes it** (`scene_bake.py`, on by default for `--lift`). The lifted points are checked against every recorded training frame at quarter resolution. A point is dropped if that frame shows a surface more than 5% farther along the same ray, or shows nothing there (alpha < 0.5): the camera saw through it. Each splat's extent is tested as well as its centre (±2 sizes along each axis), since a centre behind a surface can still reach past it.

| p10, lifted and fitted | Held-out recorded, before fit | After 4000 steps | Never-recorded vs LTX |
|---|---|---|---|
| no lift | 28.75 | | |
| lift, no free-space check | 23.09 | 27.91 | 25.42 |
| lift, centre only | 27.71 | | |
| lift, centre and extent | 28.15 | 28.75 | 25.60 |

Between a third and two thirds of each path's lifted points go (p10 49%, p11 63%, p12 59%, p13 49%, p14 31%).

`complete_walk.py` also stops if a path's fit drops held-out recorded views more than 0.5 dB below the first path's, since each path is guided by what the one before left.

**Second run** (`viewer/courtyard-walk2`; p10 as above, then p11–p14 one at a time; log in `work/courtyard/complete_walk2.log`):

| Path | Never recorded along it (first run → now) | Held-out after fit | Never-recorded vs LTX |
|---|---|---|---|
| p11 | 41% → 31% | 28.74 | 25.13 |
| p12 | 64% → 39% | 28.76 | 24.65 |
| p13 | 82% → 46% | 28.73 | 24.60 |
| p14 | 60% → 39% | 28.71 | 25.00 |

After geometry refinement and colour polish, held-out PSNR is 28.91 dB (courtyard-final 28.75, courtyard-final2 28.87). The scene has 2.60M splats: 17% recorded, 4% recorded once, 13% filled, 22% inferred, 2% rebuilt, 42% completed.

**What changed, and what didn't:**
- **Past the sofa (p13, `work/captures/walk_p13_v3.mp4`), better.** The first half matches the recording as before. Behind the sofa you see the terrace, palm, planter and glass doors; in the first run the same view was fog. Looking back at the end you get sky, the far buildings in evening light, the boundary wall and planting where there was glass: soft, but a place.
- **End of p11 (`work/captures/walk_p11_v3.mp4`), still murky.** The camera finishes almost against the planting next to the sofa. LTX's frames there are clean (trees, wall, pergola; `work/review/p11_gen_sheet_v2.jpg`), but they don't agree with the scene's guide at those frames. The fit holds its middle frames and not these last ones.
- **Free roam at 1.4× reach (`work/captures/courtyard_roam_walk2.mp4`), little change.** This test spends most of its time pressed against walls and the pergola ceiling. It is only better in the garden stretch.

## V2: point tracking (TAPNext++), to steer and check generation

[TAPNext++](https://github.com/google-deepmind/tapnet) (PyTorch, 512 px checkpoint) follows any point through a clip and catches it again after it is hidden. It runs on cached 512×512 fp16 frames, forward and back from query frames.

**Points from tracks** (`track_triangulate.py`). From every 8th frame, a grid of textured points is tracked ±90 frames, and each track is triangulated through the recorded cameras on the GPU (least-squares meeting of the rays; views more than 2 px off are dropped and it is solved again). A point is kept with at least 5 views spanning at least 2°. Courtyard, 96×54 grid: 26,824 points from 1.13M observations, median 37 views, 15° of baseline, 1.29 px error, 11 minutes.

**What they say about the fog** (`track_depth_check.py`):
- Recorded views: the scene's rendered depth is 14% nearer than the tracked points (median). Counting only splats with opacity above 0.9 it is 0.5%: a veil of half-transparent splats in front of the surfaces.
- Removing splats that recorded frames saw through to a tracked point (`track_carve.py`, 56,510 splats, 2.2%) cost 1.2 dB held out (28.91 → 27.74) and barely changed the look-back fog. That fog isn't in the recorded parts.
- Generated walk paths: tracking LTX's own frames (`track_triangulate.py --path pNN`, 64×36 grid, about 1 minute a path) keeps 67–75% of tracks at 0.8–1.1 px. Measured in those paths' cameras, where nothing was recorded, courtyard-walk2 sits in front of the tracked surfaces: median error 25% (p10), 18% (p11), 15% (p13); 56–82% of points more than 10% too near.

**Re-fitting with them** (`scene_bake.py --tracks`): the tracked points set MoGe-2's depth scale for the lift; textured pixels with no track nearby (content that wobbled between frames) teach at a quarter weight; and a depth loss holds the render to the tracked points. Same nine paths from courtyard-final, 8000 steps, with and without:

| | Held-out recorded | Never-recorded vs LTX | Sharpness in completed areas | Depth error vs path tracks (median, p10–p14) |
|---|---|---|---|---|
| without (`courtyard-tapA`) | 28.65 | 24.45 | 2.51 | 24%, 16%, 13%, 11%, 15% |
| with (`courtyard-tapB`) | 28.02 | 24.42 | 2.78 | 17%, 10%, 6%, 7%, 6% |

The depth now agrees with what the generated frames show (partly what it was trained on), and completed areas are 11% sharper, but the look-back views (`work/captures/tracks_p13.mp4`, `tracks_p10.mp4`) look only slightly better. LTX drew that fog into its frames, because its guides were rendered from the foggy scene. Carving with the path tracks as well (`courtyard-clearB`) removed recorded content: held out 28.02 → 26.20 dB. Not used.

**Filming a move** (`motion_edit.py`, `motion_check.py`; the viewer's **Film this move**). LTX-2.3's motion-track IC-LoRA moves things along point tracks, drawn as coloured trails (`LTXVDrawTracks`), from a first frame. Here every track is computed:
- the background's are triangulated points, tracked through the whole stretch, projected through the recorded cameras interpolated to 24 fps; so the camera moves as it really did;
- the object's are 16 of its surface splats in view in the first frame, moved by the edit (slide and turn, eased in and out); a background track stops where the moved object would cover it;
- the stretch of the recording is picked where the object stays in view where it is and where it goes.

`motion_check.py` tracks the first point of every track through the result with TAPNext++. On the real recording as a baseline, the background tracks match to 3.4 px and the sofa counts as not moved (0%).

| Courtyard sofa | Object error | Camera error | Object points at the new place | Time |
|---|---|---|---|---|
| slide 0.9 m toward the camera | 3.8 px | 11.3 px | 90% | 9.1 min |
| turn 40° | 11.0 px | 8.4 px | 62% | 9 min |
| 0.6 m and 15°, from the viewer | 6.6 px | 6.4 px | 100% | 6.6 min |

What doesn't work yet:
- Where the camera pans onto what the first frame doesn't show, LTX invents it: a pool, a palm, an extra armchair. Keyframes from the recording would fix the background, but they also show the object where it was.
- The 40° turn shrank the sofa to a two-seater.
- With the viewer's GPU worker holding Difix (about 5 GB), LTX's text encoder didn't fit beside it. Windows spilled VRAM into system memory, and the encode step crawled for 10 minutes before it was stopped. Now the worker moves Difix to system memory while a viewer job runs (`work/gpu_busy.json`).

## V2: replacing an object from a prompt

`replace_object.py --scene courtyard-walk2 --object 63 --prompt "a deep green velvet chesterfield sofa ..."`, step by step:

1. **Edit** (`object_edit.py`). Qwen-Image-Edit-2511 in ComfyUI redraws the sofa in its best recorded frame (a crop 2.2× its mask), told to keep the place, size, angle and light. SAM 3 finds the new sofa in the edit (overlap with the old one: best instance kept) and cuts it out. About 2 minutes. On a 24 GB card ComfyUI needs `--disable-dynamic-vram --disable-smart-memory`; dynamic VRAM was about 5× slower.
2. **3D** (`object_asset.py`, TRELLIS.2-4B, 1024 cascade). 126 s to generate; loading took 285 s the first time. The result is a 7.3M-face mesh with a voxel volume of PBR attributes, sampled into 400k flat splats on the surface. Two blockers had to be worked around:
   - **DINOv3 is gated.** TRELLIS.2's image encoder needs Meta's approval. timm's ungated copy of the same weights is converted to transformers' layout (`dinov3_from_timm.py`), and the converted model matches timm to a relative 5.7e-7 on all 1,029 tokens. transformers 5 also moved the blocks (`model.model.layer`), which TRELLIS.2's extractor didn't expect.
   - **Windows builds.** FlexGEMM and o-voxel needed small fixes (`patches/`). CuMesh and nvdiffrast are stubbed, since image-to-3D doesn't need them.
3. **Place** (`object_place.py`). The asset is posed by its silhouette in the edited frame, starting from 16 yaws and four sizes, with MoGe-2 depth to scale: IoU 0.85.
   - **Light.** TRELLIS.2's colours are albedo, unlit, and looked like a flat cut-out. A directional light plus ambient is fitted to the edited frame. Where that frame saw the surface (6% of splats, facing it, in front of the asset's own depth), its colours are taken directly, fading out at grazing angles. L1 in that frame: 0.080 → 0.032.
   - **The old sofa wasn't just its label.** Removing object 63's 27,945 splats left grey cushions poking through the new seat. Inside the old box were 186,601 completed splats (the walk completion drew the sofa whole, unlabelled) and 6,176 recorded ones the labels had missed. Everything unlabelled inside the box goes now, except a layer at the floor.
4. **Trust and review.** The new splats are flagged 4, "replaced", in the trust view, and the object's old attributes move into its replacement record. In the review from the recorded cameras (`work/courtyard/objects/replace/63/review.jpg`), the chesterfield sits where the sofa was, tufted and lit like the scene. Its back is plain green; the edit never showed it.

Not yet:
- The replaced object has 400k splats against the old one's 28k: it costs frame rate.
- The floor under the old object was never recorded. Where the new object is smaller, the gap shows the completion's guess.
- Replacing from the viewer: the pipeline runs from the command line only.
