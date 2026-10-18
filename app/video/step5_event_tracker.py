"""
Step 5: Event Tracking Pipeline

Detects battle log events, tracks them through row shifts, and extracts crops.

Core algorithm:
1. Load frames and compute signature crops (x=60..260)
2. Classify each frame as moving/redundant/state based on MSE stability
3. Detect row shifts between states
4. Track events through shifts with ID assignment
5. Extract 3 crops per event at different lifespans
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class FrameClassification:
    """Classification of a single frame."""
    frame_id: int
    kind: str  # 'moving', 'redundant', 'state'
    d_value: float  # Mean MSE to next frame
    is_stable: bool
    occupancy: List[bool]  # occupied[r] for r in 0..7


@dataclass
class Shift:
    """A row shift detected between two states."""
    from_frame: int
    to_frame: int
    shift: int  # Number of new rows
    best_error: float
    second_best_error: float
    ratio: float
    is_ambiguous: bool


@dataclass
class Event:
    """A detected event."""
    event_id: int
    arrival_frame: int
    exit_frame: Optional[int]  # None if still on screen
    observations: List[Tuple[int, int]]  # [(frame, row), ...]
    row_arrival: int  # Which row it arrived in


class Step5EventTracker:
    """Track events through battle log rows."""
    
    def __init__(self, sig_x_start: int = 60, sig_x_end: int = 260,
                 occupancy_threshold: float = 0.15,
                 stable_threshold: float = 0.003,
                 redundant_threshold: float = 0.003,
                 ambiguous_ratio: float = 0.6):
        """
        Initialize tracker.
        
        :param sig_x_start: Start x for signature crop
        :param sig_x_end: End x for signature crop
        :param occupancy_threshold: Std threshold for row occupancy (normalized 0-1 scale, ~0.15 for dark battle log)
        :param stable_threshold: MSE threshold for frame stability (normalized 0-1 scale, ~0.003)
        :param redundant_threshold: MSE threshold for redundancy (normalized 0-1 scale, ~0.003)
        :param ambiguous_ratio: Ratio threshold for ambiguous shifts
        """
        self.sig_x_start = sig_x_start
        self.sig_x_end = sig_x_end
        self.sig_width = sig_x_end - sig_x_start
        
        self.occupancy_threshold = occupancy_threshold
        self.stable_threshold = stable_threshold
        self.redundant_threshold = redundant_threshold
        self.ambiguous_ratio = ambiguous_ratio
        
        self.num_rows = 8
        self.row_heights: Dict[int, int] = {}  # Will be determined from first frame
        
        # Current state
        self.frames_color: Dict[int, np.ndarray] = {}  # frame_id -> (8, 62, 960, 3)
        self.frames_sig: Dict[int, np.ndarray] = {}    # frame_id -> (8, 62, sig_width)
        self.frame_ids: List[int] = []
        self.classifications: Dict[int, FrameClassification] = {}
        self.shifts: List[Shift] = []
        self.events: Dict[int, Event] = {}
        self.next_event_id = 0
        self.slots: List[Optional[int]] = [None] * 8  # slots[row] = event_id or None
    
    def load_frames_for_video(self, video_id: int, step4_root: str) -> bool:
        """
        Load all frames for a video from step4.
        
        :param video_id: Video ID (1-6)
        :param step4_root: Root path to step4 output
        :return: True if successful
        """
        video_dir = os.path.join(step4_root, str(video_id))
        
        # Get list of frame IDs from row 0
        row0_dir = os.path.join(video_dir, "row_0_frames")
        if not os.path.exists(row0_dir):
            print(f"  ✗ row_0_frames not found")
            return False
        
        frame_files = sorted([f for f in os.listdir(row0_dir) if f.endswith('.png')])
        frame_ids = [int(f.split('_')[2].split('.')[0]) for f in frame_files]
        
        # Check that all rows have the same frames
        for row_idx in range(self.num_rows):
            row_dir = os.path.join(video_dir, f"row_{row_idx}_frames")
            if not os.path.exists(row_dir):
                print(f"  ✗ row_{row_idx}_frames not found")
                return False
            
            row_files = sorted([f for f in os.listdir(row_dir) if f.endswith('.png')])
            row_ids = [int(f.split('_')[2].split('.')[0]) for f in row_files]
            
            if row_ids != frame_ids:
                print(f"  ✗ Frame ID mismatch: row_0 has {len(frame_ids)} frames, "
                      f"row_{row_idx} has {len(row_ids)} frames")
                missing = set(frame_ids) - set(row_ids)
                extra = set(row_ids) - set(frame_ids)
                if missing:
                    print(f"    Missing in row_{row_idx}: {sorted(missing)[:10]}...")
                if extra:
                    print(f"    Extra in row_{row_idx}: {sorted(extra)[:10]}...")
                return False
        
        self.frame_ids = frame_ids
        
        # Load frames
        print(f"  Loading {len(frame_ids)} frames...")
        for frame_idx, frame_id in enumerate(frame_ids):
            rows = []
            sigs = []
            
            for row_idx in range(self.num_rows):
                row_dir = os.path.join(video_dir, f"row_{row_idx}_frames")
                frame_path = os.path.join(row_dir, f"row_{row_idx}_{frame_id:06d}.png")
                
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"  ✗ Failed to load {frame_path}")
                    return False
                
                rows.append(img)
                
                # Store row height from first frame
                if frame_idx == 0:
                    self.row_heights[row_idx] = img.shape[0]
                
                # Extract signature crop
                sig = img[:, self.sig_x_start:self.sig_x_end]
                sig_float = sig.astype(np.float32) / 255.0
                sigs.append(sig_float)
            
            self.frames_color[frame_id] = np.array(rows, dtype=object)  # Array of variable-height arrays
            self.frames_sig[frame_id] = np.array(sigs, dtype=object)     # Array of variable-height arrays
        
        print(f"  ✓ Loaded {len(frame_ids)} frames")
        return True
    
    def compute_occupancy(self, frame_id: int) -> List[bool]:
        """
        Compute occupancy for each row.
        
        :param frame_id: Frame ID
        :return: [occupied[r] for r in 0..7]
        """
        occupied = []
        sigs = self.frames_sig[frame_id]  # List of variable-height arrays
        
        for row_idx in range(self.num_rows):
            row_sig = sigs[row_idx]  # (height, sig_width, 3)
            row_gray = cv2.cvtColor(row_sig * 255, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            
            std_val = float(np.std(row_gray))
            occupied.append(std_val > self.occupancy_threshold)
        
        return occupied
    
    def compute_frame_distances(self):
        """Compute d[t] = mean MSE between consecutive stable frames."""
        print("  Computing frame distances...")
        
        for i, frame_id in enumerate(self.frame_ids):
            if i == len(self.frame_ids) - 1:
                # Last frame is stable with d = 0
                d_value = 0.0
                is_stable = True
            else:
                next_frame_id = self.frame_ids[i + 1]
                sigs1 = self.frames_sig[frame_id]      # List of variable-height arrays
                sigs2 = self.frames_sig[next_frame_id]
                
                # MSE over all rows (handle variable heights by padding or averaging)
                mses = []
                for row_idx in range(self.num_rows):
                    sig1 = sigs1[row_idx]
                    sig2 = sigs2[row_idx]
                    
                    # Handle different heights by using min
                    min_h = min(sig1.shape[0], sig2.shape[0])
                    mse = np.mean((sig1[:min_h] - sig2[:min_h]) ** 2)
                    mses.append(mse)
                
                d_value = float(np.mean(mses))
                is_stable = d_value < self.stable_threshold
            
            occupancy = self.compute_occupancy(frame_id)
            
            classification = FrameClassification(
                frame_id=frame_id,
                kind='moving' if not is_stable else 'unknown',
                d_value=d_value,
                is_stable=is_stable,
                occupancy=occupancy,
            )
            
            self.classifications[frame_id] = classification
    
    def classify_frames(self):
        """Classify each frame as moving/redundant/state."""
        print("  Classifying frames...")
        
        # First pass: moving vs stable
        self.compute_frame_distances()
        
        # Second pass: stable frames -> redundant vs state
        last_state_id = None
        
        for frame_id in self.frame_ids:
            clf = self.classifications[frame_id]
            
            if not clf.is_stable:
                clf.kind = 'moving'
            else:
                if last_state_id is None:
                    clf.kind = 'state'
                    last_state_id = frame_id
                else:
                    # Compare with last state
                    curr_sigs = self.frames_sig[frame_id]
                    last_sigs = self.frames_sig[last_state_id]
                    
                    mses = []
                    for row_idx in range(self.num_rows):
                        curr_sig = curr_sigs[row_idx]
                        last_sig = last_sigs[row_idx]
                        
                        # Handle variable heights
                        min_h = min(curr_sig.shape[0], last_sig.shape[0])
                        mse = np.mean((curr_sig[:min_h] - last_sig[:min_h]) ** 2)
                        mses.append(mse)
                    
                    overall_mse = np.mean(mses)
                    
                    if overall_mse < self.redundant_threshold:
                        clf.kind = 'redundant'
                    else:
                        clf.kind = 'state'
                        last_state_id = frame_id
    
    def detect_shift(self, old_frame_id: int, new_frame_id: int) -> Shift:
        """
        Detect row shift between two state frames.
        
        :param old_frame_id: Old state frame ID
        :param new_frame_id: New state frame ID
        :return: Shift object
        """
        old_sigs = self.frames_sig[old_frame_id]  # List of variable-height arrays
        new_sigs = self.frames_sig[new_frame_id]
        old_occ = self.classifications[old_frame_id].occupancy
        new_occ = self.classifications[new_frame_id].occupancy
        
        errors = []
        
        for s in range(9):  # 0..8 rows
            row_errors = []
            
            for j in range(self.num_rows):
                old_j = j - s
                
                if old_j < 0:
                    continue  # New row, no old row to compare
                
                if not old_occ[old_j]:
                    continue  # Old row not occupied, skip
                
                # Compare new_sigs[j] with old_sigs[old_j]
                new_sig = new_sigs[j]
                old_sig = old_sigs[old_j]
                
                # Handle variable heights
                min_h = min(new_sig.shape[0], old_sig.shape[0])
                mse = np.mean((new_sig[:min_h] - old_sig[:min_h]) ** 2)
                row_errors.append(mse)
            
            if row_errors:
                mean_error = float(np.mean(row_errors))
            else:
                mean_error = float('inf')
            
            errors.append(mean_error)
        
        # Find best and second-best
        sorted_errors = sorted(enumerate(errors), key=lambda x: x[1])
        best_s, best_error = sorted_errors[0]
        second_best_s, second_best_error = sorted_errors[1]
        
        ratio = best_error / max(second_best_error, 1e-9)
        is_ambiguous = ratio > self.ambiguous_ratio
        
        return Shift(
            from_frame=old_frame_id,
            to_frame=new_frame_id,
            shift=best_s,
            best_error=best_error,
            second_best_error=second_best_error,
            ratio=ratio,
            is_ambiguous=is_ambiguous,
        )
    
    def track_events(self):
        """Track events through all frames."""
        print("  Tracking events...")
        
        state_frames = [fid for fid in self.frame_ids 
                        if self.classifications[fid].kind == 'state']
        
        if not state_frames:
            print("    No state frames found")
            return
        
        # Initialize on first state frame
        first_state_id = state_frames[0]
        occ = self.classifications[first_state_id].occupancy
        
        for row_idx in range(self.num_rows):
            if occ[row_idx]:
                event_id = self.next_event_id
                self.next_event_id += 1
                
                event = Event(
                    event_id=event_id,
                    arrival_frame=first_state_id,
                    exit_frame=None,
                    observations=[(first_state_id, row_idx)],
                    row_arrival=row_idx,
                )
                self.events[event_id] = event
                self.slots[row_idx] = event_id
        
        # Record all observations for first state
        self._record_state_observations(first_state_id)
        
        # Process shifts
        for i in range(len(state_frames) - 1):
            old_state_id = state_frames[i]
            new_state_id = state_frames[i + 1]
            
            shift = self.detect_shift(old_state_id, new_state_id)
            self.shifts.append(shift)
            
            # Handle shift
            old_occ = self.classifications[old_state_id].occupancy
            new_occ = self.classifications[new_state_id].occupancy
            
            # Mark exiting events
            for row_idx in range(8 - shift.shift, 8):
                if self.slots[row_idx] is not None:
                    event_id = self.slots[row_idx]
                    self.events[event_id].exit_frame = old_state_id
                    self.slots[row_idx] = None
            
            # Shift slots
            new_slots = [None] * 8
            for j in range(8):
                old_j = j - shift.shift
                if old_j >= 0:
                    new_slots[j] = self.slots[old_j]
            self.slots = new_slots
            
            # Assign new IDs
            for row_idx in range(min(shift.shift, 8)):
                if new_occ[row_idx]:
                    event_id = self.next_event_id
                    self.next_event_id += 1
                    
                    event = Event(
                        event_id=event_id,
                        arrival_frame=new_state_id,
                        exit_frame=None,
                        observations=[(new_state_id, row_idx)],
                        row_arrival=row_idx,
                    )
                    self.events[event_id] = event
                    self.slots[row_idx] = event_id
            
            # Record observations for new state
            self._record_state_observations(new_state_id)
    
    def _record_state_observations(self, state_id: int):
        """Record frame/row observations for all live events in a state."""
        for row_idx in range(self.num_rows):
            if self.slots[row_idx] is not None:
                event_id = self.slots[row_idx]
                self.events[event_id].observations.append((state_id, row_idx))
    
    def extract_crops_for_events(self, video_id: int, output_dir: str):
        """
        Extract 3 crops per event at different lifespans.
        
        :param video_id: Video ID
        :param output_dir: Output directory
        """
        print("  Extracting crops...")
        
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        
        for event in self.events.values():
            if len(event.observations) < 1:
                continue
            
            observations = event.observations
            n = len(observations)
            part_size = max(1, n // 3)
            
            for part_idx in range(3):
                start_idx = part_idx * part_size
                if part_idx == 2:
                    end_idx = n
                else:
                    end_idx = start_idx + part_size
                
                if start_idx >= n:
                    continue
                
                # Pick sharpest crop in this part
                best_frame_id = None
                best_row = None
                best_sharpness = -1
                
                for frame_id, row_idx in observations[start_idx:end_idx]:
                    rows = self.frames_color[frame_id]
                    row_img = rows[row_idx]  # Variable-height image
                    
                    gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    sharpness = float(np.var(laplacian))
                    
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame_id = frame_id
                        best_row = row_idx
                
                if best_frame_id is not None:
                    rows = self.frames_color[best_frame_id]
                    crop = rows[best_row]  # Variable-height image
                    
                    crop_filename = f"ev_{event.event_id}_{part_idx}_f{best_frame_id}_r{best_row}.png"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
    
    def generate_summary(self, video_id: int) -> Dict:
        """Generate summary statistics."""
        moving_count = sum(1 for c in self.classifications.values() if c.kind == 'moving')
        redundant_count = sum(1 for c in self.classifications.values() if c.kind == 'redundant')
        state_count = sum(1 for c in self.classifications.values() if c.kind == 'state')
        
        events_exited = sum(1 for e in self.events.values() if e.exit_frame is not None)
        events_on_screen = sum(1 for e in self.events.values() if e.exit_frame is None)
        
        ambiguous_shifts = sum(1 for s in self.shifts if s.is_ambiguous)
        
        frame_kind_log = {
            fid: self.classifications[fid].kind 
            for fid in self.frame_ids
        }
        
        return {
            "video_id": video_id,
            "total_frames": len(self.frame_ids),
            "frame_counts": {
                "moving": moving_count,
                "redundant": redundant_count,
                "state": state_count,
            },
            "events": {
                "total": len(self.events),
                "exited": events_exited,
                "on_screen": events_on_screen,
            },
            "shifts": {
                "total": len(self.shifts),
                "ambiguous": ambiguous_shifts,
            },
            "frame_kind": frame_kind_log,
        }
