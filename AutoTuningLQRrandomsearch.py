import time
import math
import numpy as np
import csv
from array import array
from scipy.linalg import solve_continuous_are
from quanser.hardware import HIL

print("QUBE AUTO-SWINGUP & IMPROVED LQR RANDOM SEARCH")

# ============================================================
# CSV FILNAVN
# ============================================================
base_name = input("Indtast base-navn til CSV-filer, fx 'iter_search1': ").strip()
if base_name == "":
    base_name = "iter_search"

summary_filename = base_name + "_FULL_SUMMARY.csv"

# ============================================================
# RANDOM SEARCH SETTINGS
# ============================================================
np.random.seed(7)

N_ROUNDS = 3
CANDIDATES_PER_ROUND = 2
TRIALS_PER_CANDIDATE = 2

SHRINK_FACTOR = 0.70

Q_THETA_DOT_FIXED = 0.1

START_Q = np.diag([47.684347, 0.1, 539.873612, 15.036553])
START_R = np.array([[21.686888]])

# Smallere og mere realistiske ranges omkring kendt god kandidat
current_ranges = {
    "Q_theta":     (40.0, 120.0),
    "Q_alpha":     (350.0, 700.0),
    "Q_alpha_dot": (6.0, 35.0),
    "R":           (18.0, 80.0),
}

GLOBAL_BOUNDS = {
    "Q_theta":     (1.0, 1000.0),
    "Q_alpha":     (50.0, 3000.0),
    "Q_alpha_dot": (0.5, 80.0),
    "R":           (2.0, 500.0),
}

# ============================================================
# SYSTEM PARAMETERS
# ============================================================
mp, Lp, g = 0.024, 0.129, 9.81
Jp = (1 / 12) * mp * Lp**2

A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-10.9651, -5.0384, 75.2427, -0.3320],
    [0.0, 0.0, 0.0, 1.0],
    [-13.8292, -6.3544, 240.4531, -1.0608]
], dtype=float)

B = np.array([
    [0.0],
    [48.5804],
    [0.0],
    [61.2695]
], dtype=float)

# ============================================================
# CONTROL SETTINGS
# ============================================================
VEL_FILTER_BETA = 0.0

MU = 31.0
E_REF = 0.05

CPR = 2048.0
TWO_PI = 2.0 * math.pi
DT = 0.005
RUN_TIME = 20.0
U_MAX = 10.0

ALPHA_ENABLE_DEG = 2.0
ALPHA_DISABLE_DEG = 25.0

THETA_SAFETY_DEG = 120.0

LQR_SIGN = -1.0
x_eq = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def wrap_to_pi(a):
    return (a + math.pi) % TWO_PI - math.pi


def random_log_uniform(low, high):
    return 10 ** np.random.uniform(np.log10(low), np.log10(high))


def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ (B.T @ P)


def set_led_rgb(card, r, g, b):
    try:
        card.write_other(
            array("I", [11000, 11001, 11002]),
            3,
            array("d", [r, g, b])
        )
    except Exception:
        pass


def led_red(card):
    set_led_rgb(card, 1.0, 0.0, 0.0)


def led_green(card):
    set_led_rgb(card, 0.0, 1.0, 0.0)


def led_off(card):
    set_led_rgb(card, 0.0, 0.0, 0.0)


def q_from_values(q_theta, q_alpha, q_alpha_dot):
    return np.diag([q_theta, Q_THETA_DOT_FIXED, q_alpha, q_alpha_dot])


def values_from_QR(Q, R):
    return {
        "Q_theta": float(Q[0, 0]),
        "Q_alpha": float(Q[2, 2]),
        "Q_alpha_dot": float(Q[3, 3]),
        "R": float(R[0, 0]),
    }


def clip_range(low, high, global_low, global_high):
    low = max(low, global_low)
    high = min(high, global_high)

    if low >= high:
        mid = (global_low + global_high) / 2
        low = max(global_low, mid * 0.8)
        high = min(global_high, mid * 1.2)

    return low, high


def shrink_ranges_around_best(ranges, best_values):
    new_ranges = {}

    for key in ranges:
        old_low, old_high = ranges[key]
        global_low, global_high = GLOBAL_BOUNDS[key]

        old_width = old_high - old_low
        new_width = old_width * SHRINK_FACTOR
        center = best_values[key]

        new_low = center - new_width / 2
        new_high = center + new_width / 2

        new_ranges[key] = clip_range(new_low, new_high, global_low, global_high)

    return new_ranges


def print_ranges(ranges):
    print("\nCurrent search ranges:")
    for key, (low, high) in ranges.items():
        print(f"{key:12s}: {low:8.3f} to {high:8.3f}")


def generate_candidates_for_round(round_idx, ranges, current_best_Q, current_best_R):
    candidates = []

    candidates.append((f"R{round_idx}_00_CURRENT_BEST", current_best_Q, current_best_R))

    for i in range(1, CANDIDATES_PER_ROUND):
        q_theta = random_log_uniform(*ranges["Q_theta"])
        q_alpha = random_log_uniform(*ranges["Q_alpha"])
        q_alpha_dot = random_log_uniform(*ranges["Q_alpha_dot"])
        r_val = random_log_uniform(*ranges["R"])

        Q = q_from_values(q_theta, q_alpha, q_alpha_dot)
        R = np.array([[r_val]])

        candidates.append((f"R{round_idx}_{i:02d}_RAND", Q, R))

    return candidates


def compute_metrics(data_log):
    if len(data_log) == 0:
        return {
            "success": False,
            "duration": 0.0,
            "rms_alpha_error_deg": float("inf"),
            "max_alpha_error_deg": float("inf"),
            "rms_theta_deg": float("inf"),
            "max_theta_deg": float("inf"),
            "rms_theta_dot": float("inf"),
            "rms_alpha_dot": float("inf"),
            "rms_u": float("inf"),
            "max_u": float("inf"),
            "rms_du_dt": float("inf"),
            "saturation_fraction": 1.0,
            "score": float("inf")
        }

    arr = np.array(data_log, dtype=object)

    t = arr[:, 0].astype(float)
    theta_dot = arr[:, 2].astype(float)
    alpha_dot = arr[:, 4].astype(float)
    alpha_error_deg = arr[:, 5].astype(float)
    theta_deg = arr[:, 6].astype(float)
    u = arr[:, 10].astype(float)

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    rms_alpha_error_deg = float(np.sqrt(np.mean(alpha_error_deg**2)))
    max_alpha_error_deg = float(np.max(np.abs(alpha_error_deg)))

    rms_theta_deg = float(np.sqrt(np.mean(theta_deg**2)))
    max_theta_deg = float(np.max(np.abs(theta_deg)))

    rms_theta_dot = float(np.sqrt(np.mean(theta_dot**2)))
    rms_alpha_dot = float(np.sqrt(np.mean(alpha_dot**2)))

    rms_u = float(np.sqrt(np.mean(u**2)))
    max_u = float(np.max(np.abs(u)))

    if len(t) > 2:
        dt = np.diff(t)
        du = np.diff(u)
        valid = dt > 1e-6
        du_dt = du[valid] / dt[valid]
        rms_du_dt = float(np.sqrt(np.mean(du_dt**2))) if len(du_dt) > 0 else 0.0
    else:
        rms_du_dt = 0.0

    saturation_fraction = float(np.mean(np.abs(u) > 0.95 * U_MAX))

    success = (
        duration >= RUN_TIME - 0.1
        and max_alpha_error_deg <= ALPHA_DISABLE_DEG
        and max_theta_deg <= THETA_SAFETY_DEG
    )

    score = (
    3.00 * rms_alpha_error_deg
    + 0.20 * max_alpha_error_deg
    + 1.50 * rms_alpha_dot
    + 0.05 * rms_theta_deg
    + 0.02 * max_theta_deg
    + 0.50 * rms_u
    + 0.05 * rms_du_dt
    + 50.0 * saturation_fraction
    )

    if not success:
        score += 1000.0

    return {
        "success": success,
        "duration": duration,
        "rms_alpha_error_deg": rms_alpha_error_deg,
        "max_alpha_error_deg": max_alpha_error_deg,
        "rms_theta_deg": rms_theta_deg,
        "max_theta_deg": max_theta_deg,
        "rms_theta_dot": rms_theta_dot,
        "rms_alpha_dot": rms_alpha_dot,
        "rms_u": rms_u,
        "max_u": max_u,
        "rms_du_dt": rms_du_dt,
        "saturation_fraction": saturation_fraction,
        "score": score
    }


def save_candidate_csv(filename, data_log):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "t",
            "theta",
            "theta_dot",
            "alpha_error",
            "alpha_dot",
            "alpha_error_deg",
            "theta_deg",
            "alpha_raw_deg",
            "alpha_ctrl_deg",
            "energy",
            "u",
            "mode",
            "Q_theta",
            "Q_theta_dot",
            "Q_alpha",
            "Q_alpha_dot",
            "R"
        ])
        writer.writerows(data_log)


def run_single_trial(card, candidate_name, Q, R, trial_idx):
    try:
        K = lqr(A, B, Q, R)
    except Exception as e:
        print(f"\nLQR computation failed for {candidate_name}: {e}")
        return []

    print("\n============================================================")
    print(f"Candidate: {candidate_name}")
    print(f"Trial: {trial_idx}")
    print("Q =")
    print(Q)
    print("R =")
    print(R)
    print("K =")
    print(K)
    print("============================================================")

    input("Lad pendulet hænge NED og tryk Enter for at starte trial...")

    card.set_encoder_counts(array("I", [0, 1]), 2, array("i", [0, 0]))
    card.write_analog(array("I", [0]), 1, array("d", [0.0]))
    led_red(card)

    monitor_t0 = time.perf_counter()
    next_time = monitor_t0

    lqr_active = False
    lqr_started_once = False
    lqr_start_time = None

    theta_dot_filt = 0.0
    alpha_dot_filt = 0.0
    filter_initialized = False
    last_print = 0.0
    u = 0.0

    data_log = []

    while True:
        now = time.perf_counter()
        if now < next_time:
            time.sleep(max(0.0, next_time - now))

        counts = array("i", [0, 0])
        other_buf = array("d", [0.0, 0.0])

        card.read_encoder(array("I", [0, 1]), 2, counts)
        card.read_other(array("I", [14000, 14001]), 2, other_buf)

        theta = (counts[0] / CPR) * TWO_PI
        alpha_raw = (counts[1] / CPR) * TWO_PI
        alpha_for_control = wrap_to_pi(math.pi - alpha_raw)

        theta_dot = (other_buf[0] / CPR) * TWO_PI
        alpha_dot = (other_buf[1] / CPR) * TWO_PI

        if not filter_initialized:
            theta_dot_filt = theta_dot
            alpha_dot_filt = alpha_dot
            filter_initialized = True
        else:
            theta_dot_filt = (
                VEL_FILTER_BETA * theta_dot_filt
                + (1.0 - VEL_FILTER_BETA) * theta_dot
            )
            alpha_dot_filt = (
                VEL_FILTER_BETA * alpha_dot_filt
                + (1.0 - VEL_FILTER_BETA) * alpha_dot
            )

        alpha_err_deg = abs(math.degrees(alpha_for_control))
        theta_deg = math.degrees(theta)

        # HARD SAFETY
        if abs(theta_deg) > THETA_SAFETY_DEG:
            print("SAFETY STOP: theta too large")
            break

        energy = (
            0.5 * Jp * alpha_dot_filt**2
            + 0.5 * mp * g * Lp * (1.0 + math.cos(alpha_for_control))
        )

        # ====================================================
        # CONTROL LOGIC
        # ====================================================
        if not lqr_active:
            mode = "SWING"

            if alpha_err_deg < ALPHA_ENABLE_DEG:
                lqr_active = True
                mode = "LQR"
                led_green(card)

                if not lqr_started_once:
                    lqr_started_once = True
                    lqr_start_time = now
                    print(">>> LQR CATCH! Starting LQR timer at t = 0.000 s <<<")

            else:
                swing_direction = np.sign(alpha_dot_filt * math.cos(alpha_for_control))
                if swing_direction == 0:
                    swing_direction = 1.0

                u = MU * (energy - E_REF) * swing_direction

        if lqr_active:
            mode = "LQR"

            x = np.array([
                theta,
                theta_dot_filt,
                alpha_for_control,
                -alpha_dot_filt
            ], dtype=float)

            u = LQR_SIGN * float(K @ (x - x_eq))
            u = max(min(u, U_MAX), -U_MAX)

            t_log = now - lqr_start_time

            data_log.append([
                t_log,
                theta,
                theta_dot_filt,
                alpha_for_control,
                -alpha_dot_filt,
                alpha_err_deg,
                theta_deg,
                math.degrees(alpha_raw),
                math.degrees(alpha_for_control),
                energy,
                u,
                mode,
                Q[0, 0],
                Q[1, 1],
                Q[2, 2],
                Q[3, 3],
                R[0, 0]
            ])

            if alpha_err_deg > ALPHA_DISABLE_DEG:
                print(">>> FALLING - BACK TO SWING <<<")
                break

        else:
            u = max(min(u, U_MAX), -U_MAX)

        card.write_analog(array("I", [0]), 1, array("d", [u]))

        if lqr_started_once:
            t_now = now - lqr_start_time
        else:
            t_now = now - monitor_t0

        if t_now - last_print >= 0.2:
            print(
                f"t={t_now:5.2f}s | {mode:5s} | "
                f"Err={alpha_err_deg:6.2f} deg | "
                f"Theta={theta_deg:7.2f} deg | "
                f"U={u:6.2f} V"
            )
            last_print = t_now

        if lqr_started_once and (now - lqr_start_time > RUN_TIME):
            break

        next_time += DT

    card.write_analog(array("I", [0]), 1, array("d", [0.0]))
    led_red(card)

    return data_log


def average_metrics(metrics_list):
    valid = [m for m in metrics_list if np.isfinite(m["score"])]

    if len(valid) == 0:
        return compute_metrics([])

    avg = {}

    keys = valid[0].keys()

    for key in keys:
        if key == "success":
            avg[key] = all(m[key] for m in valid)
        else:
            avg[key] = float(np.mean([m[key] for m in valid]))

    return avg


def write_summary_csv(filename, rows):
    rows_sorted = sorted(rows, key=lambda row: row[-1])

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Round",
            "Candidate",
            "Q_theta",
            "Q_theta_dot",
            "Q_alpha",
            "Q_alpha_dot",
            "R",
            "Success",
            "LQR_Duration_s",
            "RMS_AlphaError_deg",
            "Max_AlphaError_deg",
            "RMS_Theta_deg",
            "Max_Theta_deg",
            "RMS_u_V",
            "Max_u_V",
            "Score"
        ])
        writer.writerows(rows_sorted)

    return rows_sorted


def print_best(rows_sorted, top_n=5):
    print("\n===== BEST CANDIDATES BY SCORE =====")
    for row in rows_sorted[:top_n]:
        print(
            f"{row[1]} | "
            f"Q=diag([{row[2]:.2f}, {row[3]:.2f}, {row[4]:.2f}, {row[5]:.2f}]), "
            f"R={row[6]:.2f} | "
            f"Success={row[7]} | "
            f"RMS alpha={row[9]:.3f} deg | "
            f"Max alpha={row[10]:.3f} deg | "
            f"RMS theta={row[11]:.3f} deg | "
            f"Score={row[15]:.3f}"
        )


# ============================================================
# MAIN LOOP
# ============================================================
all_summary_rows = []

current_best_Q = START_Q
current_best_R = START_R
current_best_score = float("inf")

card = HIL()

try:
    card.open("qube_servo3_usb", "0")
    card.write_digital(array("I", [0]), 1, array("b", [1]))

    print("\nSensor-konvention: ned = 0 grader, op = 180 grader.")
    print("Data logges kun når LQR er aktiv.")
    print("Ctrl+C kan bruges til nødstop.")

    for round_idx in range(1, N_ROUNDS + 1):
        print("\n\n############################################################")
        print(f"STARTING RANDOM SEARCH ROUND {round_idx}/{N_ROUNDS}")
        print("############################################################")

        print_ranges(current_ranges)

        candidates = generate_candidates_for_round(
            round_idx,
            current_ranges,
            current_best_Q,
            current_best_R
        )

        round_rows = []

        for idx, (candidate_name, Q, R) in enumerate(candidates, start=1):
            full_name = f"{base_name}_R{round_idx}_C{idx:02d}_{candidate_name}"

            metrics_list = []

            for trial_idx in range(1, TRIALS_PER_CANDIDATE + 1):
                trial_name = f"{full_name}_T{trial_idx}"

                data_log = run_single_trial(card, trial_name, Q, R, trial_idx)

                if data_log:
                    candidate_csv = trial_name + ".csv"
                    save_candidate_csv(candidate_csv, data_log)
                    print(f"Saved candidate data: {candidate_csv}")
                    metrics = compute_metrics(data_log)
                else:
                    print("No LQR data collected for this trial.")
                    metrics = compute_metrics([])

                metrics_list.append(metrics)

                print("\nTrial metrics:")
                print(f"Success:             {metrics['success']}")
                print(f"Duration:            {metrics['duration']:.3f} s")
                print(f"RMS alpha error:     {metrics['rms_alpha_error_deg']:.3f} deg")
                print(f"Max alpha error:     {metrics['max_alpha_error_deg']:.3f} deg")
                print(f"RMS theta:           {metrics['rms_theta_deg']:.3f} deg")
                print(f"Max theta:           {metrics['max_theta_deg']:.3f} deg")
                print(f"RMS voltage:         {metrics['rms_u']:.3f} V")
                print(f"Max voltage:         {metrics['max_u']:.3f} V")
                print(f"Saturation fraction: {metrics['saturation_fraction']:.3f}")
                print(f"Score:               {metrics['score']:.3f}")

                input("\nTryk Enter for næste trial/kandidat...")

            avg_metrics = average_metrics(metrics_list)

            print("\nAveraged candidate metrics:")
            print(f"Success:         {avg_metrics['success']}")
            print(f"Average score:   {avg_metrics['score']:.3f}")
            print(f"RMS alpha avg:   {avg_metrics['rms_alpha_error_deg']:.3f} deg")
            print(f"Max alpha avg:   {avg_metrics['max_alpha_error_deg']:.3f} deg")
            print(f"RMS theta avg:   {avg_metrics['rms_theta_deg']:.3f} deg")

            row = [
                round_idx,
                full_name,
                Q[0, 0],
                Q[1, 1],
                Q[2, 2],
                Q[3, 3],
                R[0, 0],
                avg_metrics["success"],
                avg_metrics["duration"],
                avg_metrics["rms_alpha_error_deg"],
                avg_metrics["max_alpha_error_deg"],
                avg_metrics["rms_theta_deg"],
                avg_metrics["max_theta_deg"],
                avg_metrics["rms_u"],
                avg_metrics["max_u"],
                avg_metrics["score"]
            ]

            round_rows.append(row)
            all_summary_rows.append(row)

        round_sorted = sorted(round_rows, key=lambda row: row[-1])

        round_summary_filename = f"{base_name}_R{round_idx}_SUMMARY.csv"
        write_summary_csv(round_summary_filename, round_rows)

        print(f"\nRound {round_idx} summary saved to {round_summary_filename}")
        print_best(round_sorted, top_n=min(5, len(round_sorted)))

        best_round_row = round_sorted[0]
        best_round_score = best_round_row[-1]

        best_Q = q_from_values(
            best_round_row[2],
            best_round_row[4],
            best_round_row[5]
        )
        best_R = np.array([[best_round_row[6]]])

        if best_round_score < current_best_score:
            current_best_score = best_round_score
            current_best_Q = best_Q
            current_best_R = best_R

            print("\nNew overall best found:")
            print(f"Score = {current_best_score:.3f}")
            print(
                f"Q = np.diag([{current_best_Q[0,0]:.6f}, "
                f"{current_best_Q[1,1]:.6f}, "
                f"{current_best_Q[2,2]:.6f}, "
                f"{current_best_Q[3,3]:.6f}])"
            )
            print(f"R = np.array([[{current_best_R[0,0]:.6f}]])")
        else:
            print("\nNo new overall best this round. Keeping previous best.")

        best_values = values_from_QR(current_best_Q, current_best_R)
        current_ranges = shrink_ranges_around_best(current_ranges, best_values)

        input("\nTryk Enter for næste random-search round...")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    try:
        card.write_analog(array("I", [0]), 1, array("d", [0.0]))
        card.write_digital(array("I", [0]), 1, array("b", [0]))
        led_off(card)
        card.close()
    except Exception:
        pass

    if all_summary_rows:
        print(f"\nSaving full summary to {summary_filename}...")
        all_sorted = write_summary_csv(summary_filename, all_summary_rows)

        print_best(all_sorted, top_n=min(10, len(all_sorted)))

        best = all_sorted[0]

        print("\n===== FINAL BEST CANDIDATE =====")
        print(f"Candidate: {best[1]}")
        print(f"Q = np.diag([{best[2]:.6f}, {best[3]:.6f}, {best[4]:.6f}, {best[5]:.6f}])")
        print(f"R = np.array([[{best[6]:.6f}]])")
        print(f"Score = {best[15]:.6f}")
        print(f"RMS alpha error = {best[9]:.6f} deg")
        print(f"Max alpha error = {best[10]:.6f} deg")
        print(f"RMS theta = {best[11]:.6f} deg")
        print(f"Max theta = {best[12]:.6f} deg")
        print(f"RMS voltage = {best[13]:.6f} V")

    print("Færdig.")
