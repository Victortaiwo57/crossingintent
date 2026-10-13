# Will This Pedestrian Cross? Crossing Intention from Vision Foundation-Model Features

**Live app:** https://crossing-intent-fm.streamlit.app

This project asks whether a frozen, general-purpose vision foundation model (**DINOv2**) can tell from a car's dashcam that a pedestrian is about to step into the road, **before** they start moving. It also asks whether that adds anything beyond the pedestrian's motion alone.

The repository is a Streamlit app that anyone can open in a browser to explore the idea, with a README that sets out the full JAAD experiment it leads to.

---

## Why this project is important

- Road traffic kills about **1.19 million people a year**, and more than half are pedestrians, cyclists and motorcyclists, mostly in low- and middle-income countries ([WHO, 2023](https://www.who.int/publications/i/item/9789240086517)).
- Many pedestrian crashes happen at the moment of stepping out. A warning **one second earlier** can be the difference between smooth braking and a collision.
- Most crossing predictors rely on the pedestrian's trajectory, so they only react **after** movement starts. People signal intent earlier through head turns towards traffic, posture, and a step to the kerb edge.
- Foundation models such as DINOv2 are trained on huge amounts of unlabelled images. If their frozen features already capture these cues, crossing prediction becomes **cheaper to build** (only a small head is trained) and **easier to transfer** to new cities without collecting a large labelled dataset for each one.
- Public datasets come mostly from North America and Europe. Testing on Lagos traffic, with mid-block crossings, okadas and danfos, checks whether the approach generalises to the places where pedestrian deaths are highest.

## Use cases

| Who | How this helps |
|---|---|
| Driver-assistance (ADAS) developers | Earlier pedestrian warnings and better-timed automatic emergency braking |
| Automated-vehicle planning teams | Anticipating a crossing instead of reacting to it, giving smoother and safer manoeuvres |
| Ride-hailing and fleet operators | Flagging near-miss hotspots and risky routes from dashcam footage |
| City planners and road-safety agencies | Finding where people cross outside marked crossings, to prioritise infrastructure |
| Simulation and testing teams | Realistic "about to cross" behaviour for simulated pedestrians |
| Researchers | A clean, reproducible baseline for asking what foundation models know about human intent |

## Research questions

1. Do frozen DINOv2 features of a pedestrian crop predict crossing better than box motion alone?
2. How does any advantage change as the prediction horizon grows from 0.5 s to 2 s?
3. Does a self-supervised foundation model (DINOv2) beat a supervised ImageNet encoder (ResNet-50)?
4. What kinds of cases fail, and does a model trained on JAAD transfer to Lagos dashcam footage?

## What the Streamlit app shows

| Tab | Contents |
|---|---|
| **Overview** | Why the problem matters, where it applies, and the research questions |
| **Live demo** | An interactive simulation: change how often pedestrians show an intent cue, how noisy the tracking is, and the observation window. A motion-only model and a motion + appearance model are trained live (cross-validation grouped by pedestrian), and the app plots how well each predicts crossing 0.25–2 s ahead. It is clearly labelled as an illustration, not a result. |
| **Research plan** | The comparisons the full JAAD experiment will report, and its protocol |
| **Method** | Pipeline diagram, models, protocol and limitations |

The app runs on CPU with light dependencies, so it fits Streamlit Community Cloud's free tier.

## Method

```
JAAD clip ──► pedestrian box track ──┬─► box geometry (cx, cy, w, h, Δ) ─────┐
                                     │                                       ├─► GRU ─► p(crossing at t + horizon)
                                     └─► crop ─► frozen DINOv2 ─► embedding ─┘
```

| Model | Input | Role |
|---|---|---|
| `bbox` | box position, size and frame-to-frame change | motion-only baseline |
| `visual` | frozen backbone embedding of the pedestrian crop (1.5× context) | appearance only |
| `fusion` | both | does appearance add to motion? |

- **Backbones:** `dinov2_vits14`, `dinov2_vitb14` (self-supervised) and `resnet50` (ImageNet-supervised).
- **Setup:** 16 observed frames (~0.5 s at 30 fps); horizons of 15/30/45/60 frames (0.5–2 s). Windows must be contiguous and have a known future label.
- **No leakage:** train/val/test are split **by video**, so no pedestrian or scene appears in two splits.
- **Imbalance:** weighted loss. Reported metrics are AUC and AP (threshold-free), plus balanced accuracy and F1.
- **Failure analysis:** the most confident false positives and false negatives are saved and described.
- **Domain shift:** the trained model is run on my own Lagos dashcam video (YOLO tracking + the same frozen encoder) and a sample of windows is hand-labelled.

## Results

> The live demo illustrates the hypothesis with simulated pedestrians. The full JAAD experiment is the next step, and this table will be filled when it is run. No numbers here are invented.

| model | backbone | horizon (s) | AUC | AP | balanced acc | F1 |
|---|---|---|---|---|---|---|
| bbox (motion baseline) | – | 1.0 | | | | |
| visual | dinov2_vits14 | 1.0 | | | | |
| fusion | dinov2_vits14 | 1.0 | | | | |
| fusion | resnet50 | 1.0 | | | | |

| Test data | AUC | Notes |
|---|---|---|
| JAAD test split | | |
| Lagos dashcam (own footage, n = __ labelled windows) | | |

## Environment

| Component | Version / requirement |
|---|---|
| Python | 3.10 or 3.11 |
| OS | Linux, macOS or Windows |
| **Streamlit app** | CPU only. `requirements.txt`: streamlit ≥ 1.50, numpy, pandas, scikit-learn, matplotlib |
| Network | Only needed to install packages; the app downloads nothing at runtime |
| **Full experiment (planned)** | PyTorch ≥ 2.1 and torchvision, OpenCV; DINOv2 weights from `torch.hub` (`facebookresearch/dinov2`), ResNet-50 from torchvision; Ultralytics YOLO for tracking pedestrians in my own footage |
| Hardware for the experiment | A GPU for feature extraction (a free Google Colab T4 is enough); the small GRU heads train on CPU |
| Dataset | [JAAD](https://github.com/ykotseruba/JAAD): 346 dashcam clips with pedestrian boxes and crossing labels (check its licence terms) |

## Run the app locally

```bash
git clone https://github.com/Victortaiwo57/crossing-intent-fm.git
cd crossing-intent-fm
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `README.md` to a public GitHub repository named `crossing-intent-fm`.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → choose the repo, branch `main`, main file **`app.py`**.
3. Under **Advanced settings**, choose Python 3.11 and set the app URL to `crossing-intent-fm`.
4. Deploy. The app needs no GPU and no PyTorch.

## How the full experiment will be run

1. **Data:** parse JAAD's per-clip XML annotations into pedestrian tracks with per-frame crossing labels (30 fps).
2. **Features:** crop every pedestrian box with 1.5× context and embed it with frozen DINOv2 (and frozen ResNet-50 for comparison). The backbones are never trained.
3. **Windows:** 16 observed frames (~0.5 s); label = crossing or not at 0.5 / 1 / 1.5 / 2 s ahead. Contiguous windows with known future labels only.
4. **Models:** the same small GRU head on three inputs (box motion only, DINOv2 only, both), trained with a class-weighted loss.
5. **Splits:** train/validation/test by **video** (official JAAD split), so no pedestrian appears in two splits.
6. **Analysis:** AUC/AP per horizon over at least 3 seeds; a gallery of the most confident mistakes; then the trained model run on my own Lagos dashcam footage (YOLO tracking + the same frozen encoder), with a sample of windows hand-labelled before looking at predictions.

## Evaluation plan

- Use the official JAAD video split. Report every model on the same held-out test videos.
- Always compare against the motion-only `bbox` baseline. The appearance models are only interesting if they beat it.
- Report AUC and AP across horizons 0.5–2 s, over at least 3 seeds (mean ± std).
- Break results down by condition where annotations allow (e.g. occlusion, crossing at a designated crossing or not).
- For Lagos footage, hand-label a fixed sample of windows **before** looking at predictions, and report it separately as an out-of-distribution test.

## Limitations

- JAAD's crossing label is per frame. A window labelled "not crossing" may be someone who crosses just after the horizon, so the task is noisier than it looks.
- The backbone is frozen and sees one crop per frame. It cannot use scene layout (crosswalks, signals) unless that appears in the crop.
- The Lagos test set is small, recorded by one person and labelled by one annotator.
- The live demo is a simulation built to illustrate the hypothesis; it is not evidence for it.
- Research prototype: not for driving decisions or safety-critical use. Check dataset and model licence terms before any commercial use.

## What I would do next

- Fine-tune the last DINOv2 blocks, or use patch tokens instead of the CLS token.
- Add scene context (a wider crop of the road) and ego-vehicle speed.
- Evaluate on PIE as well as JAAD.
- Collect and label a larger Lagos set to study domain shift properly.

## Repository layout

```
app.py             the Streamlit app (self-contained: demo code, charts and text)
requirements.txt   Python packages for the app
README.md          this file
```

## References

- Rasouli, Kotseruba, Tsotsos. *Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior.* ICCV Workshops, 2017. (JAAD)
- Rasouli et al. *PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction.* ICCV, 2019.
- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* 2023.
- World Health Organization. *Global status report on road safety 2023.*

## Related projects

- [RoadMind Lite](https://github.com/Victortaiwo57/roadmind-lite): road-object detection baseline (YOLO + Streamlit).
- [driver-attention-fm](https://github.com/Victortaiwo57/driver-attention-fm): where do drivers look?
- [sim2real-road-users](https://github.com/Victortaiwo57/sim2real-road-users): how realistic are simulated pedestrians?

## Author

**Victor Taiwo**, Lagos, Nigeria · [LinkedIn](https://www.linkedin.com/in/victor-taiwo-076094167/) · [GitHub](https://github.com/Victortaiwo57)

MIT License
