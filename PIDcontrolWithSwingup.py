import time
import math
import numpy as np
import csv
from array import array
from quanser.hardware import HIL
import matplotlib.pyplot as plt

print("QUBE AUTO-SWINGUP & PD/PID CONTROL - UNLIMITED ARM MODE")

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

# ============================================================
# PD/PID CONTROLLER SETTINGS
# Taken from groupmates' PD/PID controller
# ============================================================

# Arm reference is theta = 0
K_theta        = -5.0
K_tdot         = -2.0
K_alpha        = 20.0
K_adot         = 3.0
K_i_theta      = 1.0
INTEGRATOR_SAT = 0.5

theta_int = 0.0

# ============================================================
# SHARED SWING-UP / CONTROL SETTINGS
# ============================================================
# Same swing-up structure/constants as LQR code
VEL_FILTER_BETA = 0.1
MU = 35.0
E_REF = 0.05

CPR = 2048.0
TWO_PI = 2.0 * math.pi
DT = 0.001
RUN_TIME = 30.0
U_MAX = 10.0

ALPHA_ENABLE_DEG = 2.0
ALPHA_DISABLE_DEG = 25.0

data_log = []

def wrap_to_pi(a):
    return (a + math.pi) % TWO_PI - math.pi

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

card = HIL()

try:
    card.open("qube_servo3_usb", "0")
    print("\nSensor-konvention: ned = 0 grader, op = 180 grader.")
    input(
        f"Lad pendulet hænge NED. Data logges KUN når PD/PID er aktiv.\n"
        f"Gemmer til: {filename}\n"
        f"Tryk Enter for at starte..."
    )

    # Nulstil encodere
    card.set_encoder_counts(array("I", [0, 1]), 2, array("i", [0, 0]))
    card.write_digital(array("I", [0]), 1, array("b", [1]))
    led_red(card)

    monitor_t0 = time.perf_counter()
    next_time = monitor_t0

    balance_active = False
    balance_started_once = False
    balance_start_time = None

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

        # Same alpha convention as LQR code:
        # down = 0 raw, upright = pi raw, so upright error is pi - raw
        alpha_for_control = wrap_to_pi(math.pi - alpha_raw)

        theta_dot = (other_buf[0] / CPR) * TWO_PI
        alpha_dot = (other_buf[1] / CPR) * TWO_PI

        # ====================================================
        # Velocity filtering
        # IMPORTANT:
        # Keep alpha_dot_filt with the SAME sign as the LQR swing-up code.
        # Only flip sign later inside the PD/PID balance controller.
        # ====================================================
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

        energy = (
            0.5 * Jp * alpha_dot_filt**2
            + 0.5 * mp * g * Lp * (1.0 + math.cos(alpha_for_control))
        )

        # ====================================================
        # CONTROL LOGIC
        # ====================================================
        if not balance_active:
            mode = "SWING"

            if alpha_err_deg < ALPHA_ENABLE_DEG:
                balance_active = True
                mode = "BALANCE"
                led_green(card)

                theta_int = 0.0

                if not balance_started_once:
                    balance_started_once = True
                    balance_start_time = now
                    print(">>> PD/PID CATCH! Starting balance timer at t = 0.000 s <<<")
                else:
                    print(">>> PD/PID CATCH! <<<")

            else:
                # Same swing-up as LQR code
                swing_direction = np.sign(alpha_dot_filt * math.cos(alpha_for_control))
                if swing_direction == 0:
                    swing_direction = 1.0
                u = MU * (energy - E_REF) * swing_direction

        if balance_active:
            mode = "BALANCE"

            # ====================================================
            # PD/PID BALANCE CONTROL
            # Alpha velocity sign is flipped here only, because
            # alpha_for_control = pi - alpha_raw.
            # ====================================================
            theta_err = theta
            theta_int += theta_err * DT
            theta_int = max(min(theta_int, INTEGRATOR_SAT), -INTEGRATOR_SAT)

            alpha_dot_balance = -alpha_dot_filt

            u = -(
                K_theta   * theta_err
                + K_tdot    * theta_dot_filt
                + K_alpha   * alpha_for_control
                + K_adot    * alpha_dot_balance
                + K_i_theta * theta_int
            )

            # Saturate before logging/writing
            u = max(min(u, U_MAX), -U_MAX)

            # Time relative to first balance catch
            t_log = now - balance_start_time

            # Log only when PD/PID balance is active
            data_log.append([
                t_log,
                theta,
                theta_dot_filt,
                alpha_for_control,
                alpha_dot_balance,
                alpha_err_deg,
                theta_deg,
                math.degrees(alpha_raw),
                math.degrees(alpha_for_control),
                energy,
                u,
                mode,
                K_theta,
                K_tdot,
                K_alpha,
                K_adot,
                K_i_theta
            ])

            if alpha_err_deg > ALPHA_DISABLE_DEG:
                balance_active = False
                mode = "SWING"
                led_red(card)
                print(">>> FALLING - BACK TO SWING <<<")

        else:
            # Saturate swing-up voltage too
            u = max(min(u, U_MAX), -U_MAX)

        # Output begrænsning og skrivning
        card.write_analog(array("I", [0]), 1, array("d", [u]))

        # Console time
        if balance_started_once:
            t_now = now - balance_start_time
        else:
            t_now = now - monitor_t0

        if t_now - last_print >= 0.2:
            if balance_started_once:
                print(
                    f"t={t_now:5.2f}s | {mode:7s} | "
                    f"Err={alpha_err_deg:6.2f} deg | "
                    f"theta={theta_deg:+7.2f} deg | "
                    f"U={u:+6.2f} V"
                )
            else:
                print(
                    f"t=wait {t_now:5.2f}s | {mode:7s} | "
                    f"Err={alpha_err_deg:6.2f} deg | "
                    f"theta={theta_deg:+7.2f} deg | "
                    f"U={u:+6.2f} V"
                )
            last_print = t_now

        # Stop after RUN_TIME seconds of balance time, not total monitoring time
        if balance_started_once and (now - balance_start_time > RUN_TIME):
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

    # Gem kun hvis der er indsamlet balance data
    if data_log:
        print(f"\nGemmer PD/PID balance-data til {filename}...")
        with open(filename, mode='w', newline='') as file:
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
                "K_theta",
                "K_tdot",
                "K_alpha",
                "K_adot",
                "K_i_theta"
            ])
            writer.writerows(data_log)
        print("Fil gemt korrekt.")
    else:
        print("\nIngen PD/PID balance-data blev indsamlet. Ingen fil gemt.")

    print("Færdig.")

df = None
with open(filename, mode='r') as file:
    reader = csv.DictReader(file)
    df = [row for row in reader]

df = pd.DataFrame(df)

plt.figure()
plt.plot(df['t'], df['alpha_error_deg'], label='alpha (deg)')
plt.plot(df['t'], df['theta_deg'], label='theta (deg)')
plt.legend()
ax2 = plt.twinx()
ax2.plot(df['t'], df['u'], color='k', alpha=0.4, label='u (V)')
ax2.set_ylabel('u (V)')
plt.xlabel('t [s]')
plt.show()
