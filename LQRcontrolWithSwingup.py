import time
import math
import numpy as np
import csv
from array import array
from scipy.linalg import solve_continuous_are
from quanser.hardware import HIL

print("QUBE AUTO-SWINGUP & LQR CONTROL - UNLIMITED ARM MODE")

# ============================================================
# CSV FILNAVN KONFIGURATION
# ============================================================
filename_input = input("Indtast navn på CSV-fil : ")
if not filename_input.lower().endswith('.csv'):
    filename = filename_input + ".csv"
else:
    filename = filename_input

# ============================================================
# SYSTEM PARAMETERS
# ============================================================
mp, Lp, g = 0.024, 0.129, 9.81
Jp = (1 / 12) * mp * Lp**2

# Lineariseret model omkring "op" positionen
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
Q = np.diag([47.684347, 0.1, 539.873612, 15.036553])
R = np.array([[31.686888]])

VEL_FILTER_BETA = 0.1 # Direkte hastighedsmåling
MU = 46.0
E_REF = 0.05

CPR = 2048.0
TWO_PI = 2.0 * math.pi
DT = 0.005
RUN_TIME = 10.0  # Sat til 5 sekunders LQR-tid for at holde data ren
U_MAX = 10.0

ALPHA_ENABLE_DEG = 2.0
ALPHA_DISABLE_DEG = 25.0

LQR_SIGN = -1.0
x_eq = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

data_log = []

def wrap_to_pi(a):
    return (a + math.pi) % TWO_PI - math.pi

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

# Beregn K-matrix én gang
K = lqr(A, B, Q, R)
card = HIL()

try:
    card.open("qube_servo3_usb", "0")
    print("\nSensor-konvention: ned = 0 grader, op = 180 grader.")
    input(f"Lad pendulet hænge NED. Hovedforløb logges fuldt ud nu.\nGemmer til: {filename}\nTryk Enter for at starte...")

    # Nulstil encodere
    card.set_encoder_counts(array("I", [0, 1]), 2, array("i", [0, 0]))
    card.write_digital(array("I", [0]), 1, array("b", [1]))
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

    while True:
        now = time.perf_counter()
        if now < next_time:
            time.sleep(max(0.0, next_time - now))

        # Læs sensorer
        counts = array("i", [0, 0])
        other_buf = array("d", [0.0, 0.0])
        card.read_encoder(array("I", [0, 1]), 2, counts)
        card.read_other(array("I", [14000, 14001]), 2, other_buf)

        theta = (counts[0] / CPR) * TWO_PI
        alpha_raw = (counts[1] / CPR) * TWO_PI
        alpha_for_control = wrap_to_pi(math.pi - alpha_raw)

        theta_dot = (other_buf[0] / CPR) * TWO_PI
        alpha_dot = (other_buf[1] / CPR) * TWO_PI

        # Hastighedsfiltrering
        if not filter_initialized:
            theta_dot_filt = theta_dot
            alpha_dot_filt = alpha_dot
            filter_initialized = True
        else:
            theta_dot_filt = VEL_FILTER_BETA * theta_dot_filt + (1.0 - VEL_FILTER_BETA) * theta_dot
            alpha_dot_filt = VEL_FILTER_BETA * alpha_dot_filt + (1.0 - VEL_FILTER_BETA) * alpha_dot

        alpha_err_deg = abs(math.degrees(alpha_for_control))
        theta_deg = math.degrees(theta)

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
                    swing_up_duration = now - monitor_t0
                    print(f">>> LQR CATCH! Swing-up tog: {swing_up_duration:.3f} s. Starter LQR-periode <<<")
                else:
                    print(">>> LQR CATCH! <<<")

            else:
                swing_direction = np.sign(alpha_dot_filt * math.cos(alpha_for_control))
                if swing_direction == 0:
                    swing_direction = 1.0
                u = MU * (energy - E_REF) * swing_direction

        if lqr_active:
            mode = "LQR"

            # State vector: [theta, theta_dot, alpha, alpha_dot]
            x = np.array([theta, theta_dot_filt, alpha_for_control, -alpha_dot_filt], dtype=float)
            u = LQR_SIGN * float(K @ (x - x_eq))

            if alpha_err_deg > ALPHA_DISABLE_DEG:
                lqr_active = False
                mode = "SWING"
                led_red(card)
                print(">>> FALLING - BACK TO SWING <<<")

        # Output begrænsning og skrivning
        u = max(min(u, U_MAX), -U_MAX)
        card.write_analog(array("I", [0]), 1, array("d", [u]))

        # --- GLOBAL LOGNING (Kører nu uanset mode) ---
        t_log = now - monitor_t0  # Tid tæller helt fra start af eksperimentet
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

        # Console print styring
        if lqr_started_once:
            t_now = now - lqr_start_time
        else:
            t_now = now - monitor_t0

        if t_now - last_print >= 0.2:
            if lqr_started_once:
                print(f"t={t_now:5.2f}s | {mode:5s} | Err={alpha_err_deg:6.2f} deg | U={u:6.2f} V")
            else:
                print(f"t=wait {t_now:5.2f}s | {mode:5s} | Err={alpha_err_deg:6.2f} deg | U={u:6.2f} V")
            last_print = t_now

        # Stop efter RUN_TIME sekunders LQR balance-tid
        if lqr_started_once and (now - lqr_start_time > RUN_TIME):
            break

        next_time += DT

finally:
    # Stop hardware sikkert
    try:
        card.write_analog(array("I", [0]), 1, array("d", [0.0]))
        card.write_digital(array("I", [0]), 1, array("b", [0]))
        led_off(card)
        card.close()
    except Exception:
        pass

    # Gem alt indsamlet data
    if data_log:
        print(f"\nGemmer komplet datasæt (Swing-up + LQR) til {filename}...")
        with open(filename, mode='w', newline='') as file:
            csv.writer(file).writerow([
                "t", "theta", "theta_dot", "alpha_error", "alpha_dot",
                "alpha_error_deg", "theta_deg", "alpha_raw_deg", "alpha_ctrl_deg",
                "energy", "u", "mode", "Q_theta", "Q_theta_dot", "Q_alpha", "Q_alpha_dot", "R"
            ])
            csv.writer(file).writerows(data_log)
        print("Fil gemt korrekt.")
    else:
        print("\nIngen data indsamlet.")

    print("Færdig.")
