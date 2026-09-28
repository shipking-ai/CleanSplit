# Third-party code

CleanSplit never downloads model weights on its own and never redistributes them. This directory holds *code* only.

## `apollo/` — vendored, tracked in git
Extracted from [JusperLee/Apollo](https://github.com/JusperLee/Apollo), pinned at commit `e84bcacc`, licence
**CC BY-SA 4.0** (see `apollo/LICENSE`). Only the modules CleanSplit needs are vendored; deviations from upstream are
documented in the file headers. Kept in-tree because `cleansplit/restoration/apollo.py` imports it directly and the
measurements in `docs/04_results.md` §9 are not reproducible without exactly this code.

**Resolved 2026-09-28.** Kept, with the licence stated plainly in the repository's top-level `LICENSE` rather than
removed. Reasoning: a repository may carry third-party directories under their own licences as long as that is
explicit and the licence text travels with them, which is the case here. CleanSplit's code *imports* Apollo; it is not
a derivative of it. The two modified files remain CC BY-SA 4.0 themselves, which is what that licence requires.

Deleting it was the other option and was rejected: it would remove the code behind a published measurement
(`docs/04_results.md` §9, Apollo worse than doing nothing in 24/24) and the two documented edits mean a plain
re-clone would not reproduce it. Reproducibility of a negative result is worth more than licence tidiness. Anyone who
cannot accept CC BY-SA in their tree can delete this directory: the restorer is off by default and nothing of measured
value is lost.

## `diffusion-audio-restoration/` — cloned, NOT tracked in git
[NVIDIA/diffusion-audio-restoration](https://github.com/NVIDIA/diffusion-audio-restoration) (A2SB), pinned at commit
`02ddff01c4dbbe20839e13e09da1988db7dd4be9` ("add docker"). It is a complete git repository of its own, so it is
git-ignored rather than vendored — committing it would create a broken gitlink that clones cannot resolve.

Restore it with:

    git clone https://github.com/NVIDIA/diffusion-audio-restoration.git third_party/diffusion-audio-restoration
    git -C third_party/diffusion-audio-restoration checkout 02ddff01c4dbbe20839e13e09da1988db7dd4be9

Its **weights are CC BY-NC** (non-commercial). A2SB lost its evaluation too (`docs/04_results.md` §4).
