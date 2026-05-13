# Static assets

Media files for the project page. The paths in `app.jsx` are wired
to the names below, so just drop files into the matching slots without
renaming.

- `static/videos/`: 14 mp4 slots.
- `static/images/`: the pipeline figure slot.
- `static/meta/`: favicon and social preview assets.

## Asset map

### Videos

| File | Section | Notes |
|---|---|---|
| `static/videos/pf-woman_000.mp4` | Qualitative Results | MovieGenVideoBench line 1 (Tokyo woman) |
| `static/videos/pf-paper_021.mp4` | Qualitative Results | MovieGenVideoBench line 22 (paper airplanes) |
| `static/videos/pf-robot_028.mp4` | Qualitative Results | MovieGenVideoBench line 29 (cyberpunk robot) |

> Qualitative-results filenames follow `pf-<tag>_<NNN>.mp4`, where the
> trailing 3-digit index identifies the MovieGenVideoBench prompt
> (`NNN + 1` is the original line number in that benchmark's prompt
> file). The prompt strings themselves are embedded inline in
> `app.jsx` so the page has no external prompt source to lose.
| `static/videos/pf-girl.mp4` | Long video showcase (featured) | Pyramid Forcing, 60 s @ 832×480 |
| `static/videos/sf-girl.mp4` | Long video showcase (bottom-left) | Self Forcing, same prompt and horizon |
| `static/videos/df-girl.mp4` | Long video showcase (bottom-right) | Deep Forcing, same prompt and horizon |
| `static/videos/sf-chef.mp4` | Drop-in effect (top-left) | Self Forcing baseline, 30 s @ 832×480 |
| `static/videos/sf-pf-chef.mp4` | Drop-in effect (top-right) | Self Forcing augmented with Pyramid Forcing |
| `static/videos/cf-chef.mp4` | Drop-in effect (bottom-left) | Causal Forcing baseline |
| `static/videos/cf-pf-chef.mp4` | Drop-in effect (bottom-right) | Causal Forcing augmented with Pyramid Forcing |

### Images

| File | Section | Source | Notes |
|---|---|---|---|
| `static/images/main_results.webp` | Hero / Figure 1 | paper `_fig/Fig_6.pdf` | Frame-extraction comparison across methods at long-horizon timesteps |
| `static/images/method_overview.webp` | Method overview / Figure 2 | paper `_fig/Fig_5.pdf` | Three-stage pipeline diagram |
| `static/images/head_attention.webp` | Head-Level Analysis / Figure 3 | paper `_fig/Fig_3.pdf` | Head-type attention patterns (anchor / wave / veil), wide aspect |

> Source PNGs were resized to 2200px-max width and re-encoded as WebP at
> q=85, dropping the three-image total from ~13 MB to ~1 MB while keeping
> figures visually clean. WebP is supported by every modern browser
> (Chrome/Firefox/Safari/Edge), no PNG fallback needed.

### Meta

| File | Use |
|---|---|
| `static/meta/favicon.svg` or `favicon.ico` | Browser tab icon |
| `static/meta/og-cover.png` | Social preview / Open Graph card |
| `paper.pdf` | Optional local PDF link for the paper (root, not `static/`) |

## Run locally

Any static server from the `demo/` folder. The page does not need a build step.

```bash
cd demo && python3 -m http.server 8000
# then open http://localhost:8000/
```

## Layout guidance

- Keep the teaser video at 16:9.
- Keep the qualitative grid at 3 columns and about 2:1 per cell.
- Keep the baseline comparison at 4 columns.
- Keep the long-video section wide enough for a 16:9 player.
- Keep the pipeline image in a wide figure band.

## Notes

- The JSX file is prewired to the exact filenames above; the deploy
  workflow stages the whole `static/` tree alongside `index.html` and
  `app.jsx`.
- Files missing from `static/` will render as broken `<video>` /
  `<img>` elements at runtime; the browser's poster background hides
  most of the noise, but the cleanest preview comes from populating
  every slot.
- If you want to swap in other media, update the `media` block at the
  top of `app.jsx` only.

## GitHub size limits

GitHub blocks individual files larger than 100 MB and warns past
50 MB. If a clip is too big, host it in external storage and put the
absolute URL into the `media` block in the
JSX instead of a `static/` path.
