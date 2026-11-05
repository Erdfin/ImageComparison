# comparison-demo

Compare two images with classical (non-ML) similarity / difference metrics.
The metrics are described, with citations, in the LaTeX document in `../research`.

## Run (Windows, existing `.venv`)

```powershell
cd comparison-demo
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\streamlit run app.py
.venv\Scripts\python -m pytest          # optional: run the tests
```

Requires Python 3.13.

## Layout

```
app.py                 Streamlit UI (Compare / Sweep / About tabs)
runner.py              UI-independent helpers: compare_pair(), run_sweep()
metrics/
  base.py              ComparisonMetric (ABC), MetricResult, Direction, helpers
  registry.py          all_metrics(), get_metric()
  pixelwise.py         MAE, MSE, RMSE, PSNR, ZNCC
  structural.py        SSIM, MS-SSIM, GMSD
  color.py             ΔE76, ΔE00
  histogram.py         intersection, Bhattacharyya, chi-square, EMD, JS, KL
  information.py       MI, NMI
  frequency.py         phase correlation
  hashing.py           aHash, dHash, pHash, wHash
sources/
  base.py              ImagePairSource (ABC), ParamSpec, ImagePair  <- the interface
  disk.py              two images from files / uploads
  synthetic.py         random reference + distorted copy (swap this file to change the generator)
tests/
```

## Adding a metric

Create a class in any module under `metrics/` (or a new module there):

```python
class MyMetric(ComparisonMetric):
    key = "my_metric"; name = "My metric"; family = "Pixel-wise"
    direction = Direction.LOWER_IS_BETTER
    value_range = (0.0, 1.0)

    def _compute(self, ref, test):            # float32, (H, W, 3), RGB in [0, 1]
        return MetricResult(value=...)        # optionally map=..., warnings=[...]

    def _to_similarity(self, value):          # optional 0-1 display score
        return 1.0 - value
```

It registers itself; the UI, sweeps and tests pick it up automatically.

## Replacing the image source

`sources/base.py` defines the contract. A source declares its parameters with
`ParamSpec` (the UI builds its controls from them) and implements
`generate(**params) -> ImagePair`. Add a module under `sources/` with a new
`ImagePairSource` subclass and it appears in the sidebar; the sweep tab can
vary any numeric parameter it declares.

## Notes

* The "similarity" column is a heuristic 0-1 score for side-by-side display,
  not a calibrated quantity. Raw values are always shown too.
* Degenerate inputs (flat images, identical images, images too small) produce
  warnings in the Notes column instead of errors, because those are exactly
  the cases the random generator is meant to explore.
* Phase 2 (not yet implemented): UQI, FSIM, HaarPSI, VIFp, CW-SSIM, FLIP and
  shape metrics on edge maps.
