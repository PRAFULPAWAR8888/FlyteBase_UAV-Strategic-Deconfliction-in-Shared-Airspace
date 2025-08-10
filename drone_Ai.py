# drone_ai_animation.py
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import time
import random

# ------------------------
# Data classes
# ------------------------
@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    time: float

@dataclass
class DroneMission:
    drone_id: str
    waypoints: List[Waypoint]

# ------------------------
# Lightweight AI predictor (original implementation restored)
# ------------------------
class DroneAIPredictor:
    def __init__(self, lookahead_seconds: float = 2.0, steps: int = 8):
        self.lookahead_seconds = lookahead_seconds
        self.steps = steps

    def _fit_linear(self, pts: List[Tuple[float, float]]):
        if len(pts) < 2:
            return None
        t = np.array([p[0] for p in pts])
        v = np.array([p[1] for p in pts])
        A = np.vstack([t, np.ones_like(t)]).T
        slope, intercept = np.linalg.lstsq(A, v, rcond=None)[0]
        return slope, intercept

    def predict_future_positions(self, recent_waypoints: List[Tuple[float, float, float, float]], current_time: float) -> Optional[List[Tuple[float, float, float, float]]]:
        if len(recent_waypoints) < 2:
            return None
        times = [p[0] for p in recent_waypoints]
        xs = [p[1] for p in recent_waypoints]
        ys = [p[2] for p in recent_waypoints]
        zs = [p[3] for p in recent_waypoints]

        coeff_x = self._fit_linear(list(zip(times, xs)))
        coeff_y = self._fit_linear(list(zip(times, ys)))
        coeff_z = self._fit_linear(list(zip(times, zs)))
        if coeff_x is None or coeff_y is None or coeff_z is None:
            return None

        future_ts = np.linspace(current_time, current_time + self.lookahead_seconds, self.steps)
        preds = []
        for ft in future_ts:
            x = coeff_x[0] * ft + coeff_x[1]
            y = coeff_y[0] * ft + coeff_y[1]
            z = coeff_z[0] * ft + coeff_z[1]
            preds.append((ft, float(x), float(y), float(z)))
        return preds

    def detect_predicted_conflicts(self, all_predictions: List[Optional[List[Tuple[float, float, float, float]]]], safety_distance: float) -> List[Tuple[float, Tuple[float,float,float], int, int]]:
        conflicts = []
        if not all_predictions:
            return conflicts
        lengths = [len(p) if p else 0 for p in all_predictions]
        n = max(lengths)
        for k in range(n):
            t = None
            pts = []
            for p in all_predictions:
                if p and len(p) > k:
                    if t is None:
                        t = p[k][0]
                    pts.append(p[k][1:4])
                else:
                    pts.append(None)
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    if pts[i] is None or pts[j] is None:
                        continue
                    d = np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))
                    if d < safety_distance:
                        mid = tuple(((np.array(pts[i]) + np.array(pts[j])) / 2.0).tolist())
                        conflicts.append((t, mid, i, j))
        return conflicts

    def avoidance_offset(self, pos_i, pos_j, safety_distance):
        v = np.array(pos_i) - np.array(pos_j)
        dist = np.linalg.norm(v)
        if dist == 0:
            theta = random.random() * 2 * math.pi
            v = np.array([math.cos(theta), math.sin(theta), 0.2])
            dist = np.linalg.norm(v)
        need = max(0.0, (safety_distance - dist) * 0.6)
        dir_vec = v / dist
        offset = dir_vec * need
        return tuple(offset), tuple(-offset)

# ------------------------
# Conflict detector (updated)
# ------------------------
class DroneConflictDetector:
    def __init__(self, safety_distance: float = 1.0):
        self.safety_distance = safety_distance

    def interpolate_position(self, waypoints: List[Waypoint], target_time: float) -> Optional[Tuple[float, float, float]]:
        if not waypoints:
            return None
        wps = sorted(waypoints, key=lambda w: w.time)
        if target_time < wps[0].time or target_time > wps[-1].time:
            return None
        for i in range(len(wps)-1):
            w1, w2 = wps[i], wps[i+1]
            if w1.time <= target_time <= w2.time:
                if w2.time == w1.time:
                    return (w1.x, w1.y, w1.z)
                frac = (target_time - w1.time) / (w2.time - w1.time)
                x = w1.x + frac * (w2.x - w1.x)
                y = w1.y + frac * (w2.y - w1.y)
                z = w1.z + frac * (w2.z - w1.z)
                return (x, y, z)
        return None

    def calculate_distance(self, a, b):
        return math.sqrt(sum((aa - bb)**2 for aa, bb in zip(a, b)))

    def check_conflicts(self, primary: DroneMission, others: List[DroneMission],
                        mission_window: Tuple[float, float], time_step: float = 0.1) -> List[dict]:
        """
        Scans mission_window from t_start to t_end (inclusive) at `time_step` intervals.
        Returns list of conflict dicts with keys: time, location (midpoint x,y,z), other_drone.
        """
        t_start, t_end = mission_window
        conflicts = []

        if t_end <= t_start:
            return conflicts

        # Build time array
        num = max(1, int(math.ceil((t_end - t_start) / time_step)))
        time_points = np.linspace(t_start, t_end, num + 1)

        for t in time_points:
            p_pos = self.interpolate_position(primary.waypoints, t)
            if p_pos is None:
                continue
            for other in others:
                o_pos = self.interpolate_position(other.waypoints, t)
                if o_pos is None:
                    continue
                dist = self.calculate_distance(p_pos, o_pos)
                if dist < self.safety_distance:
                    midpoint = tuple(((np.array(p_pos) + np.array(o_pos)) / 2.0).tolist())
                    conflicts.append({
                        "time": round(t, 2),
                        "location": tuple(round(v, 3) for v in midpoint),
                        "other_drone": other.drone_id,
                        "distance": round(dist, 3)
                    })
        return conflicts

# ------------------------
# Mission query interface
# ------------------------
def query_mission_status(primary: DroneMission, others: List[DroneMission],
                         mission_window: Tuple[float, float], detector: DroneConflictDetector, time_step: float = 0.1):
    """Simple interface: returns mission status and conflict details."""
    conflicts = detector.check_conflicts(primary, others, mission_window, time_step=time_step)
    if conflicts:
        return "conflict detected", conflicts
    else:
        return "clear", []

# ------------------------
# Visualizer with AI reaction
# ------------------------
class DroneVisualizerAnimated4D:
    def __init__(self, detector: DroneConflictDetector, fps: int = 20, lookahead: float = 2.0):
        self.detector = detector
        self.fps = fps
        self.dt = 1.0 / fps
        self.ai = DroneAIPredictor(lookahead_seconds=lookahead, steps=8)

    def _global_time_array(self, primary: DroneMission, others: List[DroneMission], dt: float):
        pstart = min(w.time for w in primary.waypoints)
        pend = max(w.time for w in primary.waypoints)
        for o in others:
            pstart = min(pstart, min(w.time for w in o.waypoints))
            pend = max(pend, max(w.time for w in o.waypoints))
        return np.arange(pstart, pend + 1e-9, dt)

    def animate_scenario(self, primary: DroneMission, others: List[DroneMission], filename: Optional[str] = None, dt: float = 0.1, show_timeplot: bool = True):
        # First: run the mission-level conflict check and print report
        t_start = min(w.time for w in primary.waypoints)
        t_end = max(w.time for w in primary.waypoints)
        status, details = query_mission_status(primary, others, (t_start, t_end), self.detector, time_step=dt)
        print("=== Mission-level status ===")
        print("Status:", status)
        if details:
            for c in details:
                print(f"  - conflict at t={c['time']}s near {c['location']} with {c['other_drone']} (dist={c['distance']})")
        else:
            print("  - no mission-level conflicts detected")
        print("============================\n")

        # Build time array for animation (global across all missions)
        times = self._global_time_array(primary, others, dt)
        nframes = len(times)

        all_missions = [primary] + others
        n_drones = len(all_missions)

        # --- initial positions, velocities, last_timeseries ---
        positions = []
        velocities = []
        last_timeseries = []
        for m in all_missions:
            start_t = min(w.time for w in m.waypoints)
            pos = self.detector.interpolate_position(m.waypoints, start_t) or (m.waypoints[0].x, m.waypoints[0].y, m.waypoints[0].z)
            positions.append(np.array(pos, dtype=float))
            if len(m.waypoints) >= 2:
                t0, t1 = m.waypoints[0].time, m.waypoints[1].time
                p0 = np.array((m.waypoints[0].x, m.waypoints[0].y, m.waypoints[0].z))
                p1 = np.array((m.waypoints[1].x, m.waypoints[1].y, m.waypoints[1].z))
                dtseg = max(1e-6, (t1 - t0))
                vel = (p1 - p0) / dtseg
            else:
                vel = np.zeros(3)
            velocities.append(vel)
            recent = [(w.time, w.x, w.y, w.z) for w in m.waypoints]
            last_timeseries.append(recent)

        # figure & axes
        fig = plt.figure(figsize=(13, 6), constrained_layout=True)
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax_time = fig.add_subplot(1, 2, 2) if show_timeplot else None

        # static paths (faint)
        for i, m in enumerate(all_missions):
            xs = [w.x for w in m.waypoints]
            ys = [w.y for w in m.waypoints]
            zs = [w.z for w in m.waypoints]
            ax3d.plot(xs, ys, zs, alpha=0.25, linestyle='-' if i==0 else '--', label=m.drone_id)

        drone_scatters = [ax3d.plot([], [], [], marker='o', markersize=7 if i==0 else 6, label=all_missions[i].drone_id)[0] for i in range(n_drones)]
        drone_trails = [ax3d.plot([], [], [], linewidth=2 if i==0 else 1)[0] for i in range(n_drones)]
        drone_labels = [ax3d.text(0,0,0,"", fontsize=8) for _ in range(n_drones)]
        ai_ghost = ax3d.scatter([],[],[], c='yellow', s=20, alpha=0.6, label='AI predicted (ghost)')
        ai_conf = ax3d.scatter([],[],[], c='orange', s=80, marker='X', label='AI predicted conflict')
        actual_conf = ax3d.scatter([],[],[], c='red', s=100, marker='X', label='Conflict')

        if ax_time is not None:
            ax_time.set_xlabel("Time")
            ax_time.set_ylabel("Altitude (Z)")
            ax_time.set_xlim(times[0], times[-1])
            time_lines = [ax_time.plot([], [], label=m.drone_id)[0] for m in all_missions]
            ax_time.legend(loc='upper right')

        # axis limits and labels
        all_x = np.concatenate([[w.x for w in m.waypoints] for m in all_missions])
        all_y = np.concatenate([[w.y for w in m.waypoints] for m in all_missions])
        all_z = np.concatenate([[w.z for w in m.waypoints] for m in all_missions])
        margin = max(0.5, 0.1 * max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)))
        ax3d.set_xlim(np.min(all_x)-margin, np.max(all_x)+margin)
        ax3d.set_ylim(np.min(all_y)-margin, np.max(all_y)+margin)
        ax3d.set_zlim(np.min(all_z)-margin, np.max(all_z)+margin)
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        ax3d.set_title("4D Visualization (3D + time)")
        ax3d.legend(loc='upper left', bbox_to_anchor=(1.02, 0.98))

        trails_x = [[] for _ in range(n_drones)]
        trails_y = [[] for _ in range(n_drones)]
        trails_z = [[] for _ in range(n_drones)]
        times_history = []

        last_ai_log = -1.0

        def update(frame_idx):
            nonlocal last_ai_log
            t = times[frame_idx]
            times_history.append(t)

            # update last_timeseries with current measured positions
            for i in range(n_drones):
                last_timeseries[i].append((t, float(positions[i][0]), float(positions[i][1]), float(positions[i][2])))
                if len(last_timeseries[i]) > 12:
                    last_timeseries[i] = last_timeseries[i][-12:]

            # AI predicts
            all_preds = []
            for i in range(n_drones):
                pred = self.ai.predict_future_positions(last_timeseries[i], t)
                all_preds.append(pred)

            pred_conflicts = self.ai.detect_predicted_conflicts(all_preds, self.detector.safety_distance)

            offsets = [np.zeros(3) for _ in range(n_drones)]
            counts = [0 for _ in range(n_drones)]
            for (tt, mid, i, j) in pred_conflicts:
                # get first pred pos or fallback to current pos
                pi = all_preds[i][0][1:4] if all_preds[i] else positions[i]
                pj = all_preds[j][0][1:4] if all_preds[j] else positions[j]
                off_i, off_j = self.ai.avoidance_offset(pi, pj, self.detector.safety_distance)
                offsets[i] += np.array(off_i)
                offsets[j] += np.array(off_j)
                counts[i] += 1
                counts[j] += 1
                if t - last_ai_log > 0.9:
                    print(f"[AI] Predicted conflict at t+{tt - t:.2f}s between {all_missions[i].drone_id} & {all_missions[j].drone_id} -> nudging.")
                    last_ai_log = t

            # apply averaged offsets to velocities smoothly
            for i in range(n_drones):
                if counts[i] > 0:
                    avg_off = offsets[i] / counts[i]
                    velocities[i] += avg_off * 0.5  # tune for smoothness

            # integrate positions
            for i in range(n_drones):
                positions[i] = positions[i] + velocities[i] * dt
                velocities[i] *= 0.995
                trails_x[i].append(positions[i][0])
                trails_y[i].append(positions[i][1])
                trails_z[i].append(positions[i][2])

            # update artists
            for i in range(n_drones):
                drone_scatters[i].set_data([positions[i][0]], [positions[i][1]])
                drone_scatters[i].set_3d_properties([positions[i][2]])
                drone_trails[i].set_data(trails_x[i], trails_y[i])
                drone_trails[i].set_3d_properties(trails_z[i])
                dx = 0.2 * (1 + (i % 2))
                drone_labels[i].set_position((positions[i][0] + dx, positions[i][1] + dx))
                drone_labels[i].set_3d_properties(positions[i][2] + 0.2)
                drone_labels[i].set_text(f"{all_missions[i].drone_id}\nZ:{positions[i][2]:.1f}")

            # AI ghost points
            ghost_pts = []
            for p in all_preds:
                if p:
                    for (_, x, y, z) in p:
                        ghost_pts.append((x, y, z))
            if ghost_pts:
                arr = np.array(ghost_pts)
                ai_ghost._offsets3d = (arr[:,0], arr[:,1], arr[:,2])
            else:
                ai_ghost._offsets3d = ([], [], [])

            # predicted conflict midpoints
            if pred_conflicts:
                ac = np.array([mid for (_, mid, _, _) in pred_conflicts])
                ai_conf._offsets3d = (ac[:,0], ac[:,1], ac[:,2])
            else:
                ai_conf._offsets3d = ([], [], [])

            # actual conflicts (current positions)
            actuals = []
            for i in range(n_drones):
                for j in range(i+1, n_drones):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < self.detector.safety_distance:
                        actuals.append(((positions[i] + positions[j]) / 2.0))
            if actuals:
                ac = np.array(actuals)
                actual_conf._offsets3d = (ac[:,0], ac[:,1], ac[:,2])
            else:
                actual_conf._offsets3d = ([], [], [])

            # update time plot
            if ax_time is not None:
                for i, tl in enumerate(time_lines):
                    zsline = trails_z[i]
                    # pad zsline to same length as times_history
                    padded = zsline + [np.nan] * (len(times_history) - len(zsline))
                    tl.set_data(times_history, padded)
                ax_time.relim(); ax_time.autoscale_view()

            fig.suptitle(f"AI-reactive Animation — time = {t:.2f}")
            artists = drone_scatters + drone_trails + [ai_ghost, ai_conf, actual_conf] + drone_labels
            if ax_time is not None:
                artists += time_lines
            return artists

        anim = FuncAnimation(fig, update, frames=nframes, interval=1000 * dt, blit=False)

        # SAVE VIDEO: try ffmpeg then Pillow fallback
        SAVE_VIDEO = True
        if filename and SAVE_VIDEO:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            try:
                writer = FFMpegWriter(fps=int(1/dt))
                print("Saving video to", filename, "(using ffmpeg). This may take a while...")
                anim.save(filename, writer=writer)
                print("Saved", filename)
            except Exception as e:
                print("FFMpeg save failed:", e)
                gifname = os.path.splitext(filename)[0] + ".gif"
                try:
                    print("Falling back to GIF:", gifname)
                    writer2 = PillowWriter(fps=int(1/dt))
                    anim.save(gifname, writer=writer2)
                    print("Saved", gifname)
                except Exception as e2:
                    print("Failed to save animation with Pillow:", e2)

        plt.show()

# ------------------------
# Example scenarios
# ------------------------
def create_conflict_scenario():
    primary = DroneMission("PRIMARY", [
        Waypoint(0, 0, 5, 0),
        Waypoint(5, 0, 5, 5),
        Waypoint(10, 0, 5, 10)
    ])
    d1 = DroneMission("DRONE_1", [
        Waypoint(5, -6, 5, 0),
        Waypoint(5, 0, 5, 5),
        Waypoint(5, 6, 5, 10)
    ])
    d2 = DroneMission("DRONE_2", [
        Waypoint(0, 3, 6, 0),
        Waypoint(5, 0, 5, 5),
        Waypoint(10, 3, 6, 10)
    ])
    return primary, [d1, d2]

def create_no_conflict_scenario():
    primary = DroneMission("PRIMARY", [
        Waypoint(0, 0, 5, 0),
        Waypoint(5, 0, 5, 5),
        Waypoint(10, 0, 5, 10)
    ])
    d1 = DroneMission("DRONE_1_SAFE", [
        Waypoint(5, -15, 5, 0),
        Waypoint(5, -15, 5, 5),
        Waypoint(5, -15, 5, 10)
    ])
    d2 = DroneMission("DRONE_2_SAFE", [
        Waypoint(0, 15, 6, 0),
        Waypoint(5, 15, 6, 5),
        Waypoint(10, 15, 6, 10)
    ])
    return primary, [d1, d2]

# ------------------------
# Run demo
# ------------------------
def main():
    detector = DroneConflictDetector(safety_distance=1.8)
    visualizer = DroneVisualizerAnimated4D(detector, fps=20, lookahead=2.0)

    primary, others = create_conflict_scenario()
    # Print an example interpolation
    print("Scenario A (example interpolated primary pos at t=5):", detector.interpolate_position(primary.waypoints, 5.0))
    visualizer.animate_scenario(primary, others, filename="conflict_ai_reactive.mp4", dt=0.05, show_timeplot=True)

    primary2, others2 = create_no_conflict_scenario()
    print("Scenario B (example interpolated primary pos at t=5):", detector.interpolate_position(primary2.waypoints, 5.0))
    visualizer.animate_scenario(primary2, others2, filename="no_conflict_ai_reactive.mp4", dt=0.05, show_timeplot=True)

if __name__ == "__main__":
    main()
