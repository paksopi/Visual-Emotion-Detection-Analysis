# Track B image set — provenance and limitation

The evaluation plan (`reports/evaluation_plan.md` §3) calls for ~20-30 staged
scene images with emotional-context ground truth "written by the tester
beforehand." That requires either a physical camera to stage shots or curated
stock photography — neither was available in this environment.

**Substitute used:** 20 real photographs of people in context (sports,
domestic scenes, social events) sourced from the `chitradrishti/Emotic`
Hugging Face mirror of the EMOTIC dataset's underlying MS-COCO images
(images only, no EMOTIC annotations were used). Images were hand-selected for
single-primary-subject framing and cases where the face alone under- or
mis-represents the emotional read (motion blur, exertion faces, obscured
faces, sun squinting) so scene context is actually load-bearing — matching
the plan's stated intent for this track.

**Ground truth in `ground_truth.csv` was authored by the AI assistant running
this evaluation, not a human tester**, by viewing each image directly. Treat
Track B's rubric scores as illustrative of methodology and relative
model behavior, not as a validated benchmark — a human-authored ground truth
set (ideally with real staged/self-collected photos, per the original plan)
would be needed before treating Track B's numbers as decision-grade.
