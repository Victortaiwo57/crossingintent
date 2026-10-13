"""Pedestrian Crossing Intention: Streamlit app.

    streamlit run app.py

Self-contained: runs on CPU with the packages in requirements.txt.
The Live demo is an illustrative simulation of the research hypothesis; the
Research plan tab describes the full JAAD / DINOv2 experiment.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

# ---------------------------------------------------------------- simulation used by the Live demo
# Simulated pedestrians wait at the kerb. Crossers step out at time t_cross; before that, many show an
# appearance cue (head turn towards traffic, leaning forward) that a visual model could see but a
# box-trajectory model cannot. Two logistic-regression models are compared on observation windows.


FPS = 30
DURATION = 10.0


def simulate(n_peds: int, cue_hit: float, cue_false: float, box_noise: float, seed: int = 0) -> list[dict]:
    """Return one dict per pedestrian with per-frame lateral position y (m), cue flags and label timeline."""
    rng = np.random.default_rng(seed)
    n = int(DURATION * FPS)
    t = np.arange(n) / FPS
    peds = []
    for i in range(n_peds):
        crosses = rng.random() < 0.5
        t_cross = rng.uniform(3.0, 8.0) if crosses else np.inf
        speed = rng.normal(1.35, 0.2)
        # lateral position: small fidgeting at the kerb, then walking into the road with a 0.5 s ramp
        moving = np.clip((t - t_cross) / 0.5, 0, 1)
        # crossers edge towards the kerb in the last ~1 s before stepping out: a weak, late motion cue
        approach = 0.25 * np.clip((t - (t_cross - 1.0)) / 1.0, 0, 1) ** 2
        y = np.cumsum(moving * speed / FPS) + approach + 0.05 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 6))
        y_obs = y + rng.normal(0, box_noise, n)
        # appearance cue: shown in the 1.5 s before stepping out (with prob cue_hit), or randomly by non-crossers
        cue = np.zeros(n, bool)
        if crosses and rng.random() < cue_hit:
            cue |= (t > t_cross - rng.uniform(0.8, 2.0)) & (t <= t_cross + 0.5)
        if rng.random() < cue_false:
            s = rng.uniform(0, DURATION - 2)
            cue |= (t > s) & (t < s + rng.uniform(0.5, 2.0))
        cue_obs = cue ^ (rng.random(n) < 0.05)  # imperfect detector
        peds.append({"id": i, "crosses": crosses, "t_cross": t_cross, "y": y_obs, "cue": cue_obs, "t": t})
    return peds


def windows(peds: list[dict], obs_s: float, horizon_s: float, stride_s: float = 0.25) -> pd.DataFrame:
    obs, hor, stride = int(obs_s * FPS), int(horizon_s * FPS), int(stride_s * FPS)
    rows = []
    for p in peds:
        y, cue, t = p["y"], p["cue"], p["t"]
        for end in range(obs, len(t) - hor, stride):
            if t[end - 1] >= p["t_cross"]:  # already crossing: trivial, skip
                break
            seg = y[end - obs:end]
            v = np.diff(seg) * FPS
            rows.append({
                "ped": p["id"],
                "pos": seg[-5:].mean(),
                "vel": v[-10:].mean(),
                "acc": (v[-5:].mean() - v[:5].mean()) / obs_s,
                "cue_share": cue[end - obs:end].mean(),
                "cue_recent": cue[end - 10:end].mean(),
                "label": int(t[end - 1 + hor] >= p["t_cross"]),
            })
    return pd.DataFrame(rows)


MOTION = ["pos", "vel", "acc"]
FULL = MOTION + ["cue_share", "cue_recent"]


def grouped_auc(df: pd.DataFrame, cols: list[str], folds: int = 4) -> float:
    if df.label.nunique() < 2:
        return float("nan")
    preds = np.zeros(len(df))
    for tr, te in GroupKFold(folds).split(df, df.label, df.ped):
        m = LogisticRegression(max_iter=500, class_weight="balanced")
        m.fit(df.iloc[tr][cols], df.iloc[tr].label)
        preds[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
    return float(roc_auc_score(df.label, preds))


def auc_vs_horizon(peds, obs_s: float, horizons: list[float]) -> pd.DataFrame:
    rows = []
    for h in horizons:
        w = windows(peds, obs_s, h)
        rows.append({"horizon_s": h, "motion only": grouped_auc(w, MOTION), "motion + appearance cue": grouped_auc(w, FULL),
                     "windows": len(w), "positive_rate": w.label.mean() if len(w) else np.nan})
    return pd.DataFrame(rows)


GITHUB = "https://github.com/Victortaiwo57/crossing-intent-fm"
BLUE, RED, AMBER, GREY = "#2a6f97", "#d1495b", "#edae49", "#8d99ae"

st.set_page_config(page_title="Pedestrian Crossing Intention", page_icon="🚶", layout="wide")


def tidy(fig, ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()


st.caption("Vision foundation models · road user behaviour understanding")
st.title("🚶 Will this pedestrian cross?")
st.markdown(
    "Predicting, from a car's dashcam, whether a pedestrian will step into the road **about one second before it "
    "happens**, and testing whether a general-purpose vision foundation model (**DINOv2**) can see intent cues "
    "that the pedestrian's motion alone does not reveal."
)

tab_over, tab_demo, tab_res, tab_method = st.tabs(["Overview", "Live demo", "Research plan", "Method"])

# ---------------------------------------------------------------- overview
with tab_over:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Why it matters")
        st.markdown(
            "- About **1.19 million people** die on the world's roads every year, and more than half are "
            "pedestrians, cyclists and motorcyclists (WHO, 2023).\n"
            "- Many pedestrian crashes happen at the moment someone steps out. A warning one second earlier gives "
            "a car (or driver) time to brake smoothly instead of not at all.\n"
            "- Systems that only track motion react *after* the person starts moving. People usually signal intent "
            "earlier: a head turn towards traffic, a lean, a step to the kerb edge.\n"
            "- If a frozen, general-purpose visual model already captures these cues, crossing prediction becomes "
            "cheaper to build and easier to move to new cities without collecting a new labelled dataset for each."
        )
    with c2:
        st.subheader("Where it could be used")
        st.markdown(
            "- **Driver-assistance systems:** earlier pedestrian warnings and automatic emergency braking.\n"
            "- **Automated vehicles:** planning that anticipates a crossing instead of reacting to it.\n"
            "- **Ride-hailing and fleet safety:** flagging near-miss hotspots from dashcam footage.\n"
            "- **Road design and city planning:** finding where people cross outside marked crossings.\n"
            "- **Driving simulators:** giving simulated pedestrians realistic 'about-to-cross' behaviour."
        )
    st.subheader("Research questions")
    st.markdown(
        "1. Do frozen DINOv2 features of a pedestrian crop predict crossing better than box motion alone?\n"
        "2. How does the advantage change as we predict further ahead (0.5 → 2 s)?\n"
        "3. Does a self-supervised foundation model beat a supervised ImageNet encoder (ResNet-50) here?\n"
        "4. What fails, and does a model trained on North American/European video transfer to Lagos traffic?"
    )
    st.info("Start with the **Live demo** tab for an interactive illustration of the idea, then **Research plan** "
            "for the full JAAD experiment.")

# ---------------------------------------------------------------- demo
with tab_demo:
    st.markdown(
        "**Illustrative simulation, not results.** Simulated pedestrians wait at the kerb. Crossers step out at a "
        "random moment. Before that, some show an **appearance cue** (looking at traffic, leaning forward) that a "
        "box-trajectory model can't see but a visual model could. Two models are trained live on noisy observations, "
        "and cross-validation is grouped by pedestrian so no one appears in both train and test."
    )
    c1, c2, c3, c4 = st.columns(4)
    cue_hit = c1.slider("Crossers who show a cue", 0.0, 1.0, 0.7, 0.05)
    cue_false = c2.slider("Non-crossers who show it anyway", 0.0, 0.6, 0.15, 0.05)
    noise = c3.slider("Box noise (m)", 0.0, 0.15, 0.04, 0.01)
    obs = c4.select_slider("Observation window (s)", [0.25, 0.5, 1.0], 0.5)

    @st.cache_data(show_spinner="Simulating pedestrians and training models…")
    def run(cue_hit, cue_false, noise, obs):
        peds = simulate(300, cue_hit, cue_false, noise, seed=0)
        return peds, auc_vs_horizon(peds, obs, [0.25, 0.5, 1.0, 1.5, 2.0])

    peds, res = run(cue_hit, cue_false, noise, obs)
    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.plot(res.horizon_s, res["motion only"], "o-", color=GREY, lw=2, label="motion only (box trajectory)")
        ax.plot(res.horizon_s, res["motion + appearance cue"], "o-", color=BLUE, lw=2, label="motion + appearance cue")
        ax.axhline(0.5, color="#bbb", ls="--", lw=1)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("how far ahead we predict (s)")
        ax.set_ylabel("ROC-AUC (0.5 = guessing)")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        tidy(fig, ax)
        st.pyplot(fig)
    with right:
        g = res.set_index("horizon_s")
        st.metric("AUC gain from the cue, 1 s ahead", f"{g.loc[1.0, 'motion + appearance cue'] - g.loc[1.0, 'motion only']:+.3f}")
        st.dataframe(res[["horizon_s", "motion only", "motion + appearance cue"]].round(3)
                     .rename(columns={"horizon_s": "horizon (s)", "motion + appearance cue": "+ cue"}),
                     hide_index=True, width="stretch")
    st.markdown(
        "**What to try:** set *Crossers who show a cue* to 0 and the two curves meet, because appearance only helps "
        "if the cue exists. Raise *Non-crossers who show it anyway* and the advantage shrinks. Short horizons are easy "
        "for both models; the interesting part is 1–2 s ahead. The real experiment asks whether DINOv2 features of "
        "real pedestrians contain such a cue."
    )
    with st.expander("See a few simulated pedestrians"):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        shown = [p for p in peds if p["crosses"]][:3] + [p for p in peds if not p["crosses"]][:2]
        for p in shown:
            ax.plot(p["t"], p["y"], color=RED if p["crosses"] else BLUE, lw=1)
            on = p["cue"]
            ax.scatter(p["t"][on][::6], p["y"][on][::6], s=6, color=AMBER, zorder=3)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("distance into road (m)")
        ax.set_ylim(-0.3, 4)
        ax.set_title("red = crosses, blue = stays · amber dots = cue detected", fontsize=9)
        tidy(fig, ax)
        st.pyplot(fig)

# ---------------------------------------------------------------- results
with tab_res:
    st.markdown(
        "**Status:** the live demo above illustrates the hypothesis. The full experiment on the **JAAD** dataset "
        "(346 dashcam clips with per-frame pedestrian boxes and crossing labels) is the next step. These are the "
        "comparisons it will report. No numbers are shown until they exist."
    )
    st.dataframe(pd.DataFrame({
        "model": ["motion only (box GRU)", "appearance only (frozen DINOv2)", "motion + DINOv2", "motion + ResNet-50"],
        "role": ["baseline to beat", "what the foundation model sees", "main model", "supervised-encoder comparison"],
        "AUC @ 0.5 s": ["–"] * 4, "AUC @ 1 s": ["–"] * 4, "AUC @ 2 s": ["–"] * 4,
    }), hide_index=True, width="stretch")
    st.markdown(
        "**Protocol:** train/test split **by video** (no pedestrian in both); horizons 0.5–2 s; AUC and AP "
        "(threshold-free) plus balanced accuracy and F1; at least 3 seeds; a gallery of the most confident mistakes; "
        "and a separate out-of-distribution test on hand-labelled Lagos dashcam footage."
    )

# ---------------------------------------------------------------- method
with tab_method:
    st.code(
        "JAAD clip ─► pedestrian box track ─┬─► box geometry (cx, cy, w, h, Δ) ─────┐\n"
        "                                   │                                       ├─► GRU ─► p(crossing at t + horizon)\n"
        "                                   └─► crop ─► frozen DINOv2 ─► embedding ─┘",
        language=None,
    )
    st.markdown(
        "- **Data:** JAAD, 346 dashcam clips with per-frame pedestrian boxes and crossing labels (30 fps).\n"
        "- **Models:** `bbox` (motion baseline), `visual` (frozen DINOv2 only), `fusion` (both); frozen ResNet-50 "
        "as a supervised comparison. Every model uses the same small GRU head, so only the input changes.\n"
        "- **Protocol:** splits by video, so no pedestrian is in both train and test; horizons of 0.5–2 s; a weighted "
        "loss for class imbalance; reported metrics are AUC, AP, balanced accuracy and F1.\n"
        "- **Analysis:** a gallery of the most confident mistakes, and running the trained model on my own Lagos "
        "dashcam footage (YOLO tracking + the same frozen encoder) to test domain shift.\n"
        "- **Limitations:** the backbone is frozen and sees one crop per frame; JAAD labels are per frame; Lagos "
        "test data is small and hand-labelled by one person."
    )
    st.markdown(f"Code and README: [{GITHUB.replace('https://', '')}]({GITHUB})")

st.divider()
st.caption("Victor Taiwo · Lagos, Nigeria · research prototype, not for driving decisions or safety-critical use.")
