"""Streamlit UI: compare two images with many classical similarity metrics.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from metrics import all_metrics, get_metric
from runner import compare_pair, instantiate_metrics, run_sweep
from sources import ImagePairSource, ParamSpec, all_sources

st.set_page_config(page_title="Image comparison demo", layout="wide")

IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"]


# ------------------------------------------------------------ parameter UI


def render_param(spec: ParamSpec, prefix: str) -> Any:
    """Draw the widget for one ``ParamSpec`` and return its current value."""
    key = f"{prefix}.{spec.name}"
    match spec.kind:
        case "int":
            return st.number_input(
                spec.title, min_value=int(spec.min), max_value=int(spec.max),
                value=int(spec.default), step=int(spec.step or 1), help=spec.help or None, key=key,
            )
        case "float":
            return st.slider(
                spec.title, float(spec.min), float(spec.max), float(spec.default),
                float(spec.step or 0.01), help=spec.help or None, key=key,
            )
        case "choice":
            return st.selectbox(
                spec.title, spec.choices, index=spec.choices.index(spec.default),
                help=spec.help or None, key=key,
            )
        case "bool":
            return st.checkbox(spec.title, value=bool(spec.default), help=spec.help or None, key=key)
        case "image":
            upload = st.file_uploader(spec.title, type=IMAGE_TYPES, key=key)
            path = st.text_input(
                f"…or path of {spec.title.lower()}", key=f"{key}.path",
                help="Path of a file on the computer running this app.",
            )
            return upload if upload is not None else (path.strip() or None)
        case _:
            return st.text_input(spec.title, value=spec.default or "", help=spec.help or None, key=key)


def difference_map_rgb(map_: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Colourise a [0, 1] difference map at the image resolution."""
    m = np.clip(map_, 0.0, 1.0)
    m = cv2.resize((m * 255).astype(np.uint8), (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.applyColorMap(m, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Image source")
    source_cls: type[ImagePairSource] = st.selectbox(
        "Source", all_sources(), format_func=lambda c: c.name, label_visibility="collapsed"
    )
    st.caption(source_cls.description)
    params = {
        spec.name: render_param(spec, f"src.{source_cls.key}") for spec in source_cls.params()
    }

    st.header("Metrics")
    metric_keys: list[str] = st.multiselect(
        "Metrics",
        [c.key for c in all_metrics()],
        default=[c.key for c in all_metrics()],
        format_func=lambda k: f"{get_metric(k).family} · {get_metric(k).name}",
        label_visibility="collapsed",
    )

source = source_cls()
metrics = instantiate_metrics(metric_keys)

st.title("Image comparison demo")
tab_compare, tab_sweep, tab_about = st.tabs(["Compare", "Sweep a parameter", "About the metrics"])

# ------------------------------------------------------------ compare tab

with tab_compare:
    pair, error = None, None
    try:
        pair = source.generate(**params)
    except ValueError as exc:
        error = str(exc)

    if error:
        st.info(error)
    elif not metrics:
        st.info("Select at least one metric in the sidebar.")
    else:
        results = compare_pair(pair, metrics)
        map_keys = [m.key for m in metrics if results[m.key].map is not None]

        col_ref, col_test, col_map = st.columns(3)
        col_ref.image(pair.reference, caption="Reference", clamp=True, width="stretch")
        col_test.image(pair.test, caption="Test", clamp=True, width="stretch")
        with col_map:
            if map_keys:
                shown = st.selectbox(
                    "Difference map", map_keys,
                    format_func=lambda k: get_metric(k).name, label_visibility="collapsed",
                )
                st.image(
                    difference_map_rgb(results[shown].map, pair.reference.shape[:2]),
                    caption=f"{get_metric(shown).name}: {results[shown].map_label} (bright = different)",
                    width="stretch",
                )
            else:
                st.caption("No metric with a difference map selected.")

        if pair.meta:
            st.caption("  ·  ".join(f"{k}: {v}" for k, v in pair.meta.items()))

        table = pd.DataFrame(
            [
                {
                    "Metric": m.name,
                    "Family": m.family,
                    "Value": results[m.key].value,
                    "Unit": m.unit,
                    "Better": m.direction.arrow,
                    "Similarity": results[m.key].similarity,
                    "Time (ms)": results[m.key].elapsed_ms,
                    "Notes": "; ".join(results[m.key].warnings),
                }
                for m in metrics
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Value": st.column_config.NumberColumn(format="%.4g"),
                "Similarity": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.3f",
                    help="Heuristic 0–1 score for side-by-side display; not calibrated.",
                ),
                "Time (ms)": st.column_config.NumberColumn(format="%.1f"),
                "Better": st.column_config.TextColumn(help="↑ higher raw value = more similar"),
            },
        )

# -------------------------------------------------------------- sweep tab

with tab_sweep:
    numeric = [s for s in source_cls.params() if s.is_numeric]
    if not numeric:
        st.info("This source has no numeric parameter to sweep.")
    elif not metrics:
        st.info("Select at least one metric in the sidebar.")
    else:
        st.write(
            "Vary one numeric parameter while keeping the others as set in the sidebar, "
            "and watch how each metric responds."
        )
        names = [s.name for s in numeric]
        sweep_name = st.selectbox(
            "Parameter to sweep", names,
            index=names.index("strength") if "strength" in names else 0,
            format_func=lambda n: next(s.title for s in numeric if s.name == n),
        )
        spec = next(s for s in numeric if s.name == sweep_name)
        is_int = spec.kind == "int"
        cast = int if is_int else float
        lo, hi = st.slider(
            "Range", cast(spec.min), cast(spec.max), (cast(spec.min), cast(spec.max)),
            cast(spec.step or (1 if is_int else 0.01)),
        )
        steps = st.number_input("Number of steps", min_value=2, max_value=201, value=21)

        if st.button("Run sweep", type="primary"):
            values = np.linspace(lo, hi, int(steps))
            if is_int:
                values = np.unique(np.round(values).astype(int))
            bar = st.progress(0.0, text="Running sweep…")
            try:
                df = run_sweep(source, params, sweep_name, values, metrics, progress=bar.progress)
                st.session_state["sweep"] = (source_cls.key, spec.title, df)
            except ValueError as exc:
                st.session_state.pop("sweep", None)
                st.info(str(exc))
            bar.empty()

        saved = st.session_state.get("sweep")
        if saved and saved[0] == source_cls.key:
            _, x_title, df = saved
            fig = px.line(
                df, x="x", y="similarity", color="metric", markers=True,
                labels={"x": x_title, "similarity": "Heuristic similarity (1 = identical)", "metric": ""},
            )
            fig.update_yaxes(range=[-0.02, 1.02])
            st.plotly_chart(fig, width="stretch")

            with st.expander("Raw metric values (one panel per metric)"):
                raw = df.assign(value=df["value"].replace([np.inf, -np.inf], np.nan))
                n_rows = -(-raw["metric"].nunique() // 4)
                fig_raw = px.line(
                    raw, x="x", y="value", facet_col="metric", facet_col_wrap=4,
                    markers=True, height=max(300, 230 * n_rows), labels={"x": x_title, "value": ""},
                )
                fig_raw.update_yaxes(matches=None, showticklabels=True)
                fig_raw.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(fig_raw, width="stretch")
                st.caption("Infinite values (e.g. PSNR of identical images) are left out of the panels.")

            warned = df[df["warnings"] != ""]
            if not warned.empty:
                with st.expander("Warnings raised during the sweep"):
                    summary = (
                        warned.groupby(["metric", "warnings"])["x"]
                        .agg(first="min", last="max", count="count")
                        .reset_index()
                    )
                    st.dataframe(summary, hide_index=True, width="stretch")

# -------------------------------------------------------------- about tab

with tab_about:
    st.write(
        "All metrics are full-reference and classical (no machine learning). "
        "Definitions and citations are in the LaTeX document in `research/`; "
        "the *citation* column gives the key of the entry in `references.bib`."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Metric": c.name,
                    "Family": c.family,
                    "Better": c.direction.arrow,
                    "Range": f"{c.value_range[0]:g} … {c.value_range[1]:g}",
                    "Needs alignment": "yes" if c.requires_alignment else "no",
                    "Description": c.description,
                    "Citation": c.citation,
                }
                for c in all_metrics()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
