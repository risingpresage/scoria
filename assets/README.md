# scoria brand assets

The mark is **a faceted volcanic shard**: three planes of a shattered block, with one
molten face. *Scoria* is the dark, vesicular rock left when lava cools — the same
after-the-heat idea as a weight tensor once the GPU has streamed it.

## Files

| file | use |
|---|---|
| `logo.svg` | horizontal lockup, dark ink — light backgrounds (README default) |
| `logo-on-dark.svg` | horizontal lockup, light ink — dark backgrounds / `<picture>` |
| `logo-mono.svg` | lockup with no magma face, single ink, for one-colour reproduction |
| `logo-mark.svg` | mark alone, dark ink |
| `logo-mark-on-dark.svg` | mark alone, light ink |
| `favicon.svg` | simplified mark (opacity planes removed) for 16–32 px |
| `social-card.svg` | 1280×640 source for the GitHub social preview |
| `social-preview.png` | raster of the above; upload at *Settings → Social preview* |

Every file is hand-checkable: `viewBox` only (no fixed `width`/`height`), `role="img"`
with `<title>`/`<desc>`, no embedded raster, no external references, no metadata, and
under 3 KB each except the social card.

## Palette

| token | hex | use |
|---|---|---|
| Obsidian | `#0b0e13` | dark backgrounds |
| Ink | `#14161a` | mark and wordmark on light |
| Ink inverse | `#f2f5f9` | mark and wordmark on dark |
| Magma | `#ff5a1f` | the one accent: the molten face |
| Ash | `#8a94a6` | secondary text |

One accent only. The magma face is the single point of colour; everything else is
ink or stone.

## Usage

- Keep clear space of at least one shard-width around the lockup.
- Minimum sizes: lockup 120 px wide, mark 16 px, favicon 16 px.
- Do not recolour the magma face, add gradients, rotate the mark further, or set the
  wordmark in another typeface.
- On dark backgrounds use the `-on-dark` files, or `<picture>` as the README does —
  some renderers ignore `prefers-color-scheme` inside an SVG loaded as an image, so the
  explicit variant is the safe path.

## Type

The wordmark is set in **Cantarell Bold**, converted to outlines so no font is required
to render it. Cantarell is licensed under the SIL Open Font License; using it to produce
logo artwork is permitted, and no font software is redistributed in this repository.
