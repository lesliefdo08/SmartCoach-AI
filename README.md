# SmartCoach AI – Cricket Pose Correction System

SmartCoach AI analyzes cricket batting or bowling videos, extracts body pose landmarks with MediaPipe, compares player posture against professional references, and provides corrective coaching feedback.

## Features

- Video upload and shot selection interface with Streamlit
- Login and Sign Up for each player account
- Password hashing with bcrypt
- Persistent user-specific session storage (SQLite)
- Pose detection using MediaPipe Pose
- Feature extraction of key cricket posture angles
- Temporal shot classification with cricket-specific motion heuristics and smoothing windows
- Real-time webcam coaching mode with live suggestions
- Professional analytics dashboard with report export
- Professional dark-themed Streamlit platform UI with sidebar navigation
- Advanced biomechanics analysis (3D pose, bat tracking, ball tracking)
- Pose similarity scoring with:
  - cosine similarity
  - euclidean distance
  - weighted joint deviation
- Rule-based AI coaching tips
- Skeleton overlay visualization with incorrect joints highlighted
- Performance analytics:
  - posture accuracy
  - joint stability
  - consistency across frames

## Project Structure

smartcoach-ai/
- app.py
- requirements.txt
- README.md
- database/
  - database.py
- modules/
  - auth.py
  - shot_classifier.py
  - biomechanics.py
  - pose_detector.py
  - session_manager.py
- ui/
  - login.py
  - signup.py
  - dashboard.py
- core/
  - video_processor.py
  - pose_detector.py
  - feature_extractor.py
  - pose_comparator.py
  - feedback_engine.py
  - shot_classifier.py
  - realtime_coach.py
  - mistake_detector.py
  - pose_3d_estimator.py
  - bat_tracker.py
  - ball_tracker.py
  - frame_pipeline.py
- train_shot_model.py
- generate_reference_pose_dataset.py
- reference_data/
  - cover_drive.json
  - cover_drive_pro.json
  - straight_drive.json
  - straight_drive_pro.json
  - pull_shot.json
  - pull_shot_pro.json
  - defense.json
  - defense_pro.json
- utils/
  - angle_utils.py
  - mediapipe_compat.py
  - visualization.py
- analytics/
  - dashboard.py
  - biomechanics_dashboard.py
  - performance_metrics.py
- assets/

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Run Streamlit app:

   streamlit run app.py

4. (Optional) Re-train shot classifier model:

  python train_shot_model.py

5. (Optional) Build professional statistical reference profiles from videos:

  python generate_reference_pose_dataset.py --source-dir assets/pro_videos --reference-dir reference_data --sample-rate 2

## Deployment (Streamlit Community Cloud)

This repository is deployment-ready for Streamlit Cloud.

- App entry point: `app.py`
- Python runtime: `3.10` (configured in `runtime.txt`)
- Dependencies: `requirements.txt`

Deployment steps:

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select repository: `lesliefdo08/SmartCoach-AI`.
4. Set branch to `main`.
5. Set main file path to `app.py`.
6. Deploy.

If you add any API keys later, store them in Streamlit secrets and do not commit them.

## Usage

1. Open the app in your browser.
2. Upload a cricket video clip (batting or bowling).
3. Select the matching shot type.
4. Click **Process video**.
5. Review:
  - detected shot type and confidence
  - detected technique issues with confidence
  - shot probability chart
   - similarity score trend
   - joint angle comparisons
   - AI coaching feedback
   - pose snapshots

## Performance Report Tab

The Streamlit app includes a **Performance Report** tab with:

- overall posture score
- top mistakes
- improvement suggestions
- chart suite:
  - angle vs frame
  - pose similarity timeline
  - posture heatmap
  - joint deviation radar chart
- frame-by-frame analysis table

Report export options:

- CSV
- PDF

## Advanced Biomechanics

The app includes a dedicated **Advanced Biomechanics** section with:

- 3D pose estimation from MediaPipe landmarks
- biomechanical metrics:
  - shoulder rotation
  - torso twist
  - bat swing plane angle
  - center of gravity estimate
- bat trajectory tracking:
  - bat tip path
  - swing speed
  - swing arc angle
- ball trajectory estimation:
  - ball path
  - impact point estimate
  - bat-ball alignment score
- overlays:
  - bat path curve
  - ball path curve
  - impact zone marker
  - 3D skeleton projection

Advanced performance metrics:

- swing_efficiency
- bat_plane_consistency
- torso_rotation_power
- impact_alignment_score
- Advanced Performance Score

## Runtime Reliability Improvements

- MediaPipe compatibility wrapper supports both:
  - legacy `mp.solutions.pose`
  - modern `mediapipe.tasks` Pose Landmarker
- Shared per-frame processing via `core/frame_pipeline.py` avoids repeated full-frame work.
- Live coaching launches in a background thread from Streamlit and runs in an external OpenCV window.
- Added stronger error handling for missing references, invalid videos, and camera access failures.

## Professional Platform UI

The interface now includes:

- sidebar navigation with sections:
  - Home
  - Video Analysis
  - Live Coaching
  - Performance Report
  - Advanced Biomechanics
- player name input
- SmartCoach AI branding and dark theme styling
- module icons and session statistics
- training session history in the sidebar
- side-by-side player pose vs ideal pose comparison
- session score card:
  - Technique Score
  - Balance Score
  - Consistency Score
  - Overall Score

## Live Coaching Mode

- Open **Live Coaching Mode** inside the Streamlit app.
- Pick a reference shot and launch webcam coaching.
- Live overlays include:
  - skeleton and joint angles
  - similarity score
  - instant correction tips
  - FPS monitor (target 15+ FPS)
- Hotkeys in webcam window:
  - `Q` to quit
  - `S` to save frame analysis (image + JSON)

## Notes

- Best results come from side-on or 45° camera angles with full body visibility.
- Statistical profile format (`*_pro.json`) uses:
  - `joint_angles_mean`
  - `joint_angles_std`
  - `ideal_ranges`
- Comparator now uses statistical tolerance (standard deviation and ideal ranges) for scientifically grounded deviation scoring.
