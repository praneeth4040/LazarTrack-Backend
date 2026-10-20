# 🗺️ Zone Tracker Pipeline (ZTE-001)

Comprehensive technical reference for the 8-step Safe Zone Tracking and Map Alignment Engine in **LazarTrack Backend**.

---

## 1. Executive Summary & Purpose

The Zone Tracker Pipeline processes Free Fire tournament match recordings and detects every safe-zone circle phase (shrink circles) across the match duration. It aligns the live in-game minimap to a high-resolution reference blank map and renders clean, numbered safe zones directly onto the map image.

### Key Outputs:
1. **Rendered Map (`rendered_map.png`)**: High-resolution map with all sequential safe zones, dark contrast halos, and phase indices.
2. **Zones Metadata (`zones.json`)**: Exact circle coordinates (center `x`, `y`, `radius`, and detection confidence score per phase).
3. **Interactive Visual Dashboard (`/api/v1/zone/{id}/visual`)**: Visual verification tool displaying intermediate artifacts from all 8 steps.

---

## 2. Architecture & 8-Step Flow

```
                      Input Video (MP4) + Reference Map
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Video Normalizer (FFmpeg)                                      │
│ • Hardware-accelerated normalization to 1920×1080 native framerate     │
│ • Corrects variable framerates and aspect ratio distortion             │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Minimap Stream Separator (FFmpeg)                              │
│ • Isolates 960×1080 half-screen minimap stream                         │
│ • Extracts initial team roster reference frame                         │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Map Boundary Cleaner (OpenCV)                                  │
│ • Uses UI anchors to isolate clean, un-occluded map bounds             │
│ • Strips spectator HUD, health bars, and kill banner overlays          │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 4: SIFT Feature Aligner & Homography (OpenCV)                     │
│ • SIFT (Scale-Invariant Feature Transform) keypoint extraction         │
│ • FLANN / RANSAC homography estimation against reference blank map     │
│ • Computes perspective transformation matrix H                         │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Frame Registrar & Camera Motion Tracker (NumPy / SciPy)        │
│ • Samples minimap frames at variable/dynamic rate (up to 5 FPS)        │
│ • Tracks camera pan and zoom across frames                             │
│ • Exports frame registration matrices to transforms.npz                │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Multi-Method Zone Detector (OpenCV / Parallel 8-Core)          │
│ • Parallel Hough Circle Transform + Lit Boundary Arc Detection         │
│ • Filters false-positive player radar circles and airdrop markers      │
│ • Records per-frame raw circle candidate observations                  │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 7: DBSCAN Consensus Zone Clusterer (Scikit-Learn)                 │
│ • Density-based clustering (DBSCAN) aggregates per-frame candidates    │
│ • Eliminates transient detection jitter and outlier noise              │
│ • Resolves discrete match phases (Zone 1 through Zone N)               │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 8: High-Resolution Zone Renderer (OpenCV / Pillow)                │
│ • Warps clustered zone circles back onto reference base map canvas     │
│ • Draws white concentric zone boundaries with anti-aliased dark halos  │
│ • Overlays phase numbers and exports rendered_map.png + zones.json     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Technical Details

### Step 1: Normalizer (`app/zone_tracker/normalizer.py`)
- Standardizes incoming video to `1920×1080`.
- Retains native framerate while ensuring pixel format compatibility (`yuv420p`).

### Step 2: Map Separator (`app/zone_tracker/map_separator.py`)
- Extracts the right-hand minimap half-screen (`960×1080`) from the normalized video.
- Extracts the team roster frame from match intro.

### Step 3: Map Cleaner (`app/zone_tracker/map_cleaner.py`)
- Locates in-game HUD anchors (compass, minimap border, spectator icons).
- Masks out distracting overlay elements, leaving an unobstructed map stream.

### Step 4: Map Aligner (`app/zone_tracker/map_aligner.py`)
- Matches live minimap features against the blank base map (`assets/maps/<map_name>.jpeg`).
- Uses **SIFT** with ratio testing (`0.75`) and **RANSAC** outlier rejection to calculate the homography transformation matrix.

### Step 5: Frame Registrar (`app/zone_tracker/frame_registrar.py`)
- Minimizes redundant computation by selecting keyframes where zone boundaries change.
- Tracks map camera position changes and stores coordinate transformation matrices in `transforms.npz`.

### Step 6: Zone Detector (`app/zone_tracker/zone_detector.py`)
- Employs an 8-core parallel processing pool.
- Combines **Hough Gradient Circle Detection** with **Lit-Boundary Arc Fitting** to detect faint safe-zone boundaries even through terrain and building textures.

### Step 7: Zone Clusterer (`app/zone_tracker/zone_clusterer.py`)
- Groups thousands of candidate circle observations using **DBSCAN** across spatial dimensions `(x, y, radius)`.
- Enforces temporal ordering so that Zone $N+1$ is strictly smaller and enclosed within Zone $N$.
- Emits consensus zone definitions.

### Step 8: Zone Renderer (`app/zone_tracker/zone_renderer.py`)
- Overlays the consensus circles on top of the blank reference map.
- Uses high-contrast dual-stroke outlines (inner bright stroke with outer dark halo) ensuring legibility across light desert terrain (Kalahari) and dark forested terrain (Bermuda / Purgatory).

---

## 4. Supported Map Reference Assets

Blank reference maps are stored in `assets/maps/`:
- `bermuda.jpeg`
- `purgatory.jpeg`
- `kalahari.jpeg`
- `solara.jpeg`
- `nexterra.jpeg`

---

## 5. REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/zone/process` | Upload match video or provide `video_path` + `map_name` |
| `GET` | `/api/v1/zone/list` | List all processed zone tracking sessions |
| `GET` | `/api/v1/zone/maps` | List supported reference map templates |
| `GET` | `/api/v1/zone/{video_id}/status` | Check processing status / progress |
| `GET` | `/api/v1/zone/{video_id}/zones` | Download raw `zones.json` |
| `GET` | `/api/v1/zone/{video_id}/zones-map` | Download rendered PNG map with safe zones |
| `GET` | `/api/v1/zone/{video_id}/step/{step_num}/{artifact}` | Download intermediate step artifact file |
| `GET` | `/api/v1/zone/{video_id}/visual` | Interactive visual inspection dashboard |

---

## 6. CLI & Batch Processing

Run the pipeline across an entire directory of tournament recordings:

```bash
# Process all test videos against Bermuda map
python3 scripts/run_all_zone_pipeline.py
```
