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

## Editable world

Pick an object in the viewer (Objects list or its tag) to get **Remove / Turn ⟲ ⟳ / Away / Toward / Left / Right / Reset** (Delete, `[` and `]` work too). The GPU worker applies the rigid transform (or hides the object) on its splats, and its box follows. Nothing is saved.

Two limits show straight away:
- Splats of an object that SAM's masks missed stay behind as a ghost (the sofa's armrest). Rebuilt objects (below) are complete, so they move cleanly.
- What was behind an object was never recorded. Moving it uncovers a hole, or the infer pass's estimate, and the trust view says which.
