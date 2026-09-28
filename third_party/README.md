# Third-party code

CleanSplit never downloads model weights on its own and never redistributes them. This directory holds *code* only.

## `apollo/` — vendored, tracked in git
Extracted from [JusperLee/Apollo](https://github.com/JusperLee/Apollo), pinned at commit `e84bcacc`, licence
**CC BY-SA 4.0** (see `apollo/LICENSE`). Only the modules CleanSplit needs are vendored; deviations from upstream are
documented in the file headers. Kept in-tree because `cleansplit/restoration/apollo.py` imports it directly and the
measurements in `docs/04_results.md` §9 are not reproducible without exactly this code.

**Open question before any public release:** CC BY-SA 4.0 is a share-alike licence. Verbatim inclusion in its own
directory with its licence intact is ordinary redistribution, but if CleanSplit's own licence is chosen to be something
other than CC BY-SA, confirm the boundary between "collective work" and "adaptation" before publishing. Apollo is
disabled by default and lost every measurement (§9), so removing it entirely is a live option.

## `diffusion-audio-restoration/` — cloned, NOT tracked in git
[NVIDIA/diffusion-audio-restoration](https://github.com/NVIDIA/diffusion-audio-restoration) (A2SB), pinned at commit
`02ddff01c4dbbe20839e13e09da1988db7dd4be9` ("add docker"). It is a complete git repository of its own, so it is
git-ignored rather than vendored — committing it would create a broken gitlink that clones cannot resolve.

Restore it with:

    git clone https://github.com/NVIDIA/diffusion-audio-restoration.git third_party/diffusion-audio-restoration
    git -C third_party/diffusion-audio-restoration checkout 02ddff01c4dbbe20839e13e09da1988db7dd4be9

Its **weights are CC BY-NC** (non-commercial). A2SB lost its evaluation too (`docs/04_results.md` §4).
