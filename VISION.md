# Vision (working name: Braindance Studio)

Open a recorded moment, freeze time, step out of the camera and walk around inside it. Change one thing and watch a different outcome. The app always shows what is recorded, what is reconstructed, what is simulated and what is guessed.

This document is the product north star. The feasibility review and build spec work from it.

## The braindance walkthrough

This is the experience the project exists to deliver. Each step notes where its content comes from, so the walkthrough stays honest.

**Scene: "The Pull."** A hand pulls a stuffed sloth across a table for a few seconds. Three depth cameras filmed it. The recording is a published PhysTwin example, which comes with a physics model already fitted to the sloth.

### v0.1: the tabletop

1. **Open.** Launch the app and open the sample scene. You don't need an account or a GPU.
2. **Replay.** The original footage plays in a small inset while the 3D scene plays in the main view, both on one timeline. The camera starts locked to one of the three original cameras. Scrub back and forth.
   *Source: recorded footage, plus the 3D scene built from it.*
3. **Freeze.** Pause. The sloth and the hand stop mid-pull.
4. **Step out.** Detach the camera and orbit the frozen moment. You can see the sloth from angles none of the three cameras had, within a marked viewing area.
   *Source: the sloth is drawn from its fitted model. The hand is drawn from the recorded depth and can only be replayed.*
5. **Scrub from outside.** Move through time while the camera stays where you left it. Watch the pull from behind.
6. **Switch layers.**
   - Color.
   - Depth: measured by the cameras, not estimated.
   - Motion: the sloth's path.
   - Coverage: which cameras saw each spot. Unseen areas stay as holes.
   Every object has a badge showing where it came from.
7. **Inspect.** Select the sloth by clicking it or picking it from the object list. The inspector shows its path, how closely the fitted model matches the footage, and the branch points available.
8. **Branch.** Jump to a branch point. The recorded pull is replaced by a handle you control: set direction and strength by dragging or typing numbers. Click Simulate, and your GPU computes a few seconds of new motion. The hand turns into a ghost, because it's recorded and doesn't react to the new pull.
   *Source: physical simulation, resumed from a saved state.*
9. **Compare.** Switch or split between Recorded, Fitted baseline and Your branch, all at the same time and from the same camera. The fitted baseline is the model's reproduction of the event, so it's close to the recording but never identical.
10. **Keep it.** Save the project with its bookmarked cameras and branches. Export a short clip along a camera path, captioned as a simulation.

### v0.2: the room

11. **Leave the table.** The tabletop now sits inside a scanned room. Fly away from the table and look around. Because the table and the room are two separate recordings, the scene is labeled "staged".
    *Source: 3D reconstruction of a room scan.*
12. **Find the edge of what was seen.** Fly into a corner the scan never filmed and it appears as an outline. Click "Infer this area": a background job fills it in on your GPU, tagged as Inferred and shown in its own color in the Coverage layer. The guess is saved, so it looks the same every time you come back.
    *Source: generated content, stored as its own layer.*

## Rules that keep the vision true

- **Time and camera are independent.** Scrubbing never moves your camera, and moving the camera never changes the time.
- **Recorded footage is never altered.** Reconstructions, simulations and guesses are separate layers on top.
- **Everything has a label.** Recorded, Reconstructed, Simulated, Inferred or Staged.
- **Guesses are made once and saved.** Inferred areas are saved into the scene, not re-imagined every frame, and never count as evidence in Investigate mode.
- **Holes are allowed.** An honest gap beats a convincing invention you didn't ask for.
- **The viewer works without a GPU.** Everything already computed plays on any machine, rendered in the browser. A local GPU adds full-quality rendering (the scene drawn by the same renderer it was trained with) and computes new branches and inferred areas. The view always says which renderer drew it.
- **Branches change one thing clearly.** A branch replaces the recorded pull. It never adds a second pull on top of the recorded one.

## Decisions so far

| Topic | Decision |
|---|---|
| Core experience | Investigate and What-if together, on one timeline |
| Goal | Open-source project that others can reproduce and contribute to |
| Footage | Existing published datasets only |
| First scene | A PhysTwin sample (three depth cameras; code is MIT-licensed, dataset license still to confirm) |
| Hardware | RTX 4090 runs simulation and inference locally |
| Unseen areas | Fill once, then render in real time. No per-frame generation |
| Infer feature | Optional plugin, not in the core. HY-World 2.0 is a test backend only because of its license; the default should be permissively licensed |
| Platform | Windows first. macOS and Linux marked untested until verified |
| Rendering | Two renderers on one timeline: GPU mode streams frames from a local gsplat worker (reference quality, about 55–60 fps at 1080p on the 4090); browser mode (Spark) is the fallback with no worker. Chosen 2026-09-24 after measuring Spark about 6 dB below the reference renderer |
| Walkthrough video | Experiment 01 shows a walkthrough of a still space can be rebuilt and explored (Pexels clips). Importing a walkthrough video is a feature (chosen 2026-09-27): `import_walkthrough.py` runs every stage in one command, a pre-flight check reports footage that won't work before hours of compute, and shots that don't connect are reported and left out, never silently merged |
| Name | Choose an original name before the repo goes public |

## Not in scope

- Generating every frame live as the camera moves.
- Filming your own footage.
- Simulating people or their reactions. Recorded people stay replay-only.
- Thermal views, recovered audio, VR, multiplayer, accounts.
- Turning any phone video into a full interactive world.

## Milestones

1. **Engine proof.** PhysTwin runs on the 4090, restores a saved state and produces two different continuations from it: the recorded pull and a changed pull.
2. **Walkthrough v0.1.** Steps 1–10 on the tabletop scene, working end to end, then the first public release.
3. **Room v0.2.** Steps 11–12: a scanned room plus the Infer plugin.
4. **Later.** A larger scene with people (replay only), and spatial audio from prepared recordings.

## Open questions

- Is the 4090 in the Windows machine used for development? PhysTwin has a Windows setup branch, and WSL2 is the fallback.
- The PhysTwin dataset license needs confirming on the official download, so we know whether the sample can be bundled or must be fetched by a script.
- How best to draw the recorded hand: fused depth points from the three cameras is the first candidate.
- Which public room scan to use, and which permissively licensed Infer backend is good enough.
- The project's final name.
