# CIELAB versus Oklab brick classification

Run from the repository root. Both setups use the existing white balance,
robust box measurement, table-relative hue/lightness/chroma features, LDA,
and bounded scene correction. CIELAB preserves the current 8-bit OpenCV
baseline; Oklab uses floating-point conversion from corrected sRGB. This
comparison therefore includes the difference in coordinate precision.
Oklab's polar hue/chroma representation is commonly called OKLCH.

Train and evaluate on the same dataset and validation splits:

```bash
bash colour_setups/cielab/train.sh
bash colour_setups/oklab/train.sh
```

Each writes its own `model.json`, including validation metrics and color-space
metadata. Training requires the existing sam3 environment with scikit-learn.
Set `PY=/path/to/python` to use another environment.

Capture and run the full geometry/classification pipeline:

```bash
bash colour_setups/cielab/run.sh
bash colour_setups/oklab/run.sh
```

Compare the same existing recording/frame (replace the recording path):

```bash
bash colour_setups/cielab/run.sh --recording /path/to/recording --frame 0
bash colour_setups/oklab/run.sh --recording /path/to/recording --frame 0
```

Results go to `data/pipeline_cielab_results` and `data/pipeline_oklab_results`.
To reuse exactly the same detector geometry, classify an existing normal result:

```bash
$HOME/miniconda3/envs/sam3/bin/python pipeline_classifier/reclassify.py /path/to/geometry_folder --colour-model colour_setups/cielab/model.json --out data/compare_cielab
$HOME/miniconda3/envs/sam3/bin/python pipeline_classifier/reclassify.py /path/to/geometry_folder --colour-model colour_setups/oklab/model.json --out data/compare_oklab
```

The original pipeline still defaults to `classifier/colour_model.json`.
Older model files without color-space metadata retain CIELAB behavior.
For output-schema compatibility, `lab_raw`, `lab_corrected`, and `lab_q*`
contain encoded Oklab coordinates in Oklab runs: `[255*L, 255*a+128, 255*b+128]`.
The classification JSON identifies the space and encoding explicitly. Do not
interpret those coordinates as CIELAB or compare numerical distances across models.

Random-fold scores can be optimistic due to similar frames; lighting-held-out
scores can be confounded when color labels and lighting groups are correlated.
Use labeled stack captures under unseen lighting to choose a production setup.
