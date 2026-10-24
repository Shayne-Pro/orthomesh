# OrthoMesh

A Streamlit prototype for facial landmark review using MediaPipe Face Mesh. Upload a photograph, review the estimated points, adjust coordinates, and export the reviewed landmarks as JSON.

The repository contains several historical UI experiments. The default entry point below opens the landmark review and calibration workflow; the original scripts remain available for comparison.

## Quick start

Use Python 3.11 or 3.12 in an isolated environment:

```bash
git clone https://github.com/Shayne-Pro/orthomesh.git
cd orthomesh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. Open the address printed by Streamlit, normally <http://localhost:8501>.

The dependency file pins the legacy MediaPipe `solutions.face_mesh` API used by these scripts. Installing an arbitrary newer MediaPipe release may require migrating the application to the Tasks API. MediaPipe already supplies OpenCV Contrib; avoid installing a second OpenCV wheel into the same environment.

## Review workflow

1. Choose frontal or side-face mode and upload a JPG or PNG image.
2. Review the estimated points and edit their pixel coordinates.
3. Download `landmarks.json`, containing the selected mode, rating, and reviewed coordinates.
4. Optionally save calibration feedback. This accumulates average coordinate offsets for later images; it does **not** retrain MediaPipe or update its model weights.

The default calibration file is `landmark_offsets.json` beside the application and is ignored by Git. Use a separate file for an independent review session:

```bash
ORTHOMESH_FEEDBACK_PATH=/path/to/session-offsets.json python -m streamlit run app.py
```

Offset statistics are shared by landmark name within a calibration file. Use separate files when populations, image setups, or analysis modes should not share calibration.

## Other entry points

| File | Purpose |
| --- | --- |
| `app.py` | Default launcher for the review workflow |
| `mediapipe_face_landmark_improved.py` | Manual coordinate adjustment, calibration feedback, JSON export |
| `app3a.py` | Historical frontal-photo measurement and text-report prototype |
| `app1.py`, `app2.py`, `app3.py` | Earlier application variants |
| `mediapipe_face_landmark.py`, `video.py` | Earlier landmark and video experiments |

Run historical applications from the repository root so their relative image assets resolve, for example `python -m streamlit run app3a.py`.

## Interpretation and project status

Coordinates and distances are measured in image pixels. The mode-specific anatomical mappings and historical proportion thresholds are prototype heuristics, with no clinical accuracy established by this repository. A detection-confidence setting is not a measured diagnostic accuracy. The default workflow is intended for reviewing landmark estimates, rather than producing a clinical diagnosis.

MediaPipe provides the underlying detector and landmark model. See its [Face Mesh documentation](https://chuoling.github.io/mediapipe/solutions/face_mesh.html) for the RGB input contract and model details. Historical UI/research text retained in older variants is not evidence of new affiliations or clinical validation.

The repository currently has no repository-level license grant; dependency licenses remain their own.

## Validation

Dependency import, Streamlit startup, calibration arithmetic, and a CPU Face Mesh smoke check can be verified without a private dataset. These checks do not measure landmark accuracy on patient images.

## Regression checks

```bash
python -m unittest discover -s tests -v
```
