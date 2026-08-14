# NeuroViz Design System — "NeuroViz Identity"

Source of truth: the claude.ai/design project **NeuroViz Identity**
(`NeuroViz Identity.dc.html`). This file mirrors its tokens for tools that
read DESIGN.md. The system's one job: read as a precise analytical
instrument. Monochrome shell; **hue is reserved for data, and data only.**

## Color — monochrome shell

| Token | Hex | Job |
|---|---|---|
| Ink | `#1C1C1C` | headlines · mark · dark surfaces |
| Graphite | `#55554E` | body copy · secondary |
| Slate | `#7A7A74` | labels · axis · meta |
| Paper | `#F4F4F1` | canvas |
| Card | `#FFFFFF` | surfaces |
| Rule | `#DDDDD7` | borders (light rule: `#ECECE7`) |

No brand hue, no tinted UI states.

## Color — data only

Categorical (6 steps, matched lightness; pass as an explicit Altair range):
`#3F5D8C` · `#4A7C63` · `#A8663F` · `#7A5A8C` · `#8C8340` · `#3F7C8C`

Sequential/heatmap: greyscale ink→paper ramp
`#1C1C1C` `#454540` `#6E6E67` `#9A9A93` `#C6C6BF` `#EAEAE5`

State hues (parser/pipeline status only, never series):
parsed `#4A7C63` · ambiguous `#8C8340` · failed `#A8663F`

## Typography — one face

**JetBrains Mono** everywhere (fallback `ui-monospace`). Weights 400/500/600.

| Role | Size / weight / tracking |
|---|---|
| Display | 68 / 500 / −0.055em |
| Heading | 26 / 600 / −0.035em |
| Label | 11 / 600 / +0.16em, uppercase |
| Body | 16 / 400 / line-height 1.7 |
| Data | 13 / 400 |

## Shape & layout

- **Radius: 0.** Square everything — never a pill, circle container, or rounded tile.
- Hierarchy from size, weight, and rule-work (1px rules), not color.
- Numbered section headers (`01 / LABEL`) over a rule line.
- Strong dividers; cards are white on paper with 1px rule borders.
- No glow, gradient, bevel, or sparkle — ever ("nothing to signal 'AI'").

## The mark

Four nodes on a rising path (three open, last solid) in a square ink
container. SVG at `app/assets/logo.svg`. Clear space = one node diameter.
Min 28px with nodes; below that, path-only variant.

## Voice

- **Show the reasoning.** State what was parsed/assumed ("Read as: monthly sum of sales").
- **No hype.** Never "AI-powered", never "intelligent insights". It's a parser and a compiler.
- **Offline is the promise.** "Runs on your machine. No key required." Facts, not badges.
- Wordmark: `NeuroViz` — one word, two capitals.

## Applied surfaces

- Streamlit app theme: `.streamlit/config.toml` (colors, zero radius, chart palette)
- Altair chart theme: registered in `app/compilers/altair_compile.py`
- Demo/landing page: `docs/index.html`
