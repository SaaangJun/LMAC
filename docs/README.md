# LMAC Project Page

This directory hosts the GitHub Pages site for the paper:

> **LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning**
> Sangjun Bae, Yisak Park, Sanghyeon Lee, Seungyul Han — ICML 2026

## Local Preview

```bash
cd docs
python -m http.server 8000
# open http://localhost:8000
```

## Deploying to GitHub Pages

1. Push this `docs/` folder to the `main` branch of <https://github.com/SaaangJun/LMAC>.
2. In the repo **Settings → Pages**, set:
   - **Source**: `Deploy from a branch`
   - **Branch**: `main` and **Folder**: `/docs`
3. Save. The page will be served at `https://saaangjun.github.io/LMAC/`.

## Updating Figures

Figures are converted from `ICML2026_Final/Figure/*.pdf` to PNG and placed in `static/images/`.
To regenerate (macOS):

```bash
cd ICML2026_Final/Figure
sips -s format png -s formatOptions best concept_v7.pdf --out ../../docs/static/images/concept.png
# ... repeat for other figures
```

## Files

- `index.html` — main project page
- `static/css/style.css` — styles
- `static/images/` — figures (converted from paper PDFs)
- `.nojekyll` — disables Jekyll processing so paths beginning with `_` are served verbatim
