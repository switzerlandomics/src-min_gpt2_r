# Model inspection and visualisation

**Status:** Preliminary design. Figure selection may evolve as the model is tested.

Keep model computation independent of plotting. During training, `--plot`
refreshes only a small `training.png` after each validation measurement.
`--plot-detailed` retains live monitoring and creates publication figures from
the best-validation checkpoint at the end (or from an existing completed run).

Each attention head in each block is exported separately at 4.5 × 4.5 inches,
with a common 0–1 scale. Masked future positions are neutral grey rather than
the colour used for valid zero attention. The detailed learning history uses
5.5 × 4.5 inches and the next-token comparison uses 5 × 4.5 inches. Dimensions
may change to accommodate measured data, but avoid generating an oversized
plot and shrinking the entire SVG in the article.

The shared company-style theme in `R/plots.R` uses at least 12 pt for every
text element, 16 pt for titles, and a standard `sans` font family. A base theme
of 12 pt is insufficient if individual captions or axis labels override it
with smaller sizes. SVG text appears smaller if the entire figure is subsequently
scaled down in Inkscape or on the website. Do not distribute font files.

Live monitoring writes PNG only. Detailed plots export PNG and, when the optional
`svglite` package is installed, SVG from the same measured ggplot object.
Labels describe byte tokens rather than assuming each token is a complete
character. A schematic diagram must be labelled as such and must not be
presented as measured model output. Plotting failures must not stop training.

Additional tensor-flow diagrams, tokenisation histories or residual-stream
figures may be added later if they clarify a specific result. Their code should
remain separate from the model's forward and backward implementations.
