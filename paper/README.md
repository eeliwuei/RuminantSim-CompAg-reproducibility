# Paper materials

`figures/` holds the four main-text figures exactly as submitted (PDF and PNG) together with
`figure_bindings.json`, which lists every plotted value and the SHA-256 of the aggregate table it was read
from. Regenerate them with `python extensions/figures/make_main_figures.py --data data/aggregate_results --out paper/figures`.
The manuscript sources are added at acceptance.
