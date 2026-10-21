# =============================================================================
#  Drug diffusion down into keloid dermis: three delivery conditions
#  Fickian model (1D finite elements), one GIF per condition, live clock
#
#  HOW TO USE IN GOOGLE COLAB
#    Paste this WHOLE file into ONE cell and press Shift+Enter (~2-3 min).
#    The three GIFs (and MP4s) appear in the folder icon on the left.
#
#  MODEL
#    dc/dt = D d2c/dz2, z = depth below the skin surface (pointing down)
#    Surface (z = 0): drug held at c0 (delivery site). Tissue continues
#    below the keloid, so the dermis is modelled as deep (semi-infinite);
#    the domain runs to 12 mm with zero flux there, far below the front.
#
#    "Time to diffuse a depth L" = time for the 50%-concentration front to
#    reach depth L. For Fick's law this front sits at z = 0.954 * sqrt(D t),
#    so t is proportional to L^2 and one D per condition reproduces both the
#    1 mm and the 2.5 mm times (ratio 2.5^2 = 6.25).
#
#    Condition 1 (current method) is modelled as NO diffusion (D = 0): the
#    drug stays at the surface and never enters the keloid. Its GIF runs the
#    clock for SHOW_YEARS to show that nothing moves.
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc, erfcinv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation

YEAR, DAY, MIN = 365.25 * 86400, 86400.0, 60.0

# ---------------------------------------------------------------- PARAMETERS
CONDITIONS = [
    dict(label="Current Method (No Pressure, No Collagenase)",
         file="keloid_1_current_method",
         no_diffusion=True),                       # D = 0, drug never enters
    dict(label="Collagenase Added Before Drug Delivery",
         file="keloid_2_collagenase",
         t_1mm=19.9 * DAY,    t_25mm_given=124.3 * DAY),
    dict(label="Pressure with Microneedle & Collagenase BOTH Added",
         file="keloid_3_microneedle_collagenase",
         t_1mm=2.37 * MIN,    t_25mm_given=14.82 * MIN),
]
DEPTHS_MM     = [1.0, 2.5]     # keloid dermis depths shown as columns
FRONT_LEVEL   = 0.5            # "arrived" = concentration reaches 50% of surface
DOMAIN_MM     = 12.0           # model depth (tissue continues below keloid)
N_ELEM        = 600
N_STEPS       = 3000
GIF_SECONDS   = 15
FPS           = 20
RUN_PAST      = 1.15           # run to 1.15 x the 2.5 mm time
SHOW_YEARS    = 5.0            # clock length for the no-diffusion GIF

Z_FRONT = 2 * erfcinv(FRONT_LEVEL)          # front depth = Z_FRONT * sqrt(D t)


# --------------------------------------------------------------------- SOLVER
def solve(D, t_end):
    """Backward-Euler P1 FEM on [0, DOMAIN_MM], c(0)=1, zero flux at bottom."""
    n = N_ELEM + 1
    h = DOMAIN_MM / N_ELEM
    z = np.linspace(0, DOMAIN_MM, n)
    # tridiagonal mass and stiffness (diagonal, off-diagonal)
    Md = np.full(n, 4 * h / 6); Md[[0, -1]] = 2 * h / 6; Mo = np.full(n - 1, h / 6)
    Kd = np.full(n, 2 / h);     Kd[[0, -1]] = 1 / h;     Ko = np.full(n - 1, -1 / h)
    t = np.concatenate([[0.0], np.geomspace(t_end * 1e-6, t_end, N_STEPS)])
    c = np.zeros(n); c[0] = 1.0
    H = np.zeros((len(t), n)); H[0] = c
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        Ad, Ao = Md + dt * D * Kd, Mo + dt * D * Ko
        b = Md * c; b[1:] += Mo * c[:-1]; b[:-1] += Mo * c[1:]      # M @ c
        rhs = b[1:].copy(); rhs[0] -= Ao[0] * 1.0                   # c(0) = 1 known
        ab = np.zeros((3, n - 1))
        ab[0, 1:] = Ao[1:]; ab[1] = Ad[1:]; ab[2, :-1] = Ao[1:]
        c[1:] = solve_banded((1, 1), ab, rhs); c[0] = 1.0
        H[k] = c
    return z, t, H


def arrival(z, t, H, depth):
    """Time when c at `depth` first reaches FRONT_LEVEL."""
    y = np.array([np.interp(depth, z, row) for row in H])
    i = int(np.argmax(y >= FRONT_LEVEL))
    if y[i] < FRONT_LEVEL:
        return np.nan
    return t[i-1] + (t[i] - t[i-1]) * (FRONT_LEVEL - y[i-1]) / (y[i] - y[i-1])


def fmt_dur(s):
    if s >= 0.5 * YEAR: return f"{s / YEAR:.2f} years"
    if s >= 2 * DAY:    return f"{s / DAY:.1f} days"
    if s >= 2 * 3600:   return f"{s / 3600:.1f} h"
    if s >= 120:        return f"{s / MIN:.2f} min"
    return f"{s:.1f} s"


def fmt_clock(s, t_end):
    if t_end >= 0.5 * YEAR:
        return f"{s / YEAR:5.2f} years  ({s / DAY:6.0f} days)"
    if t_end >= 2 * DAY:
        return f"{s / DAY:6.1f} days"
    m, sec = divmod(s, 60)
    return f"{int(m):02d}:{sec:04.1f}  (min:s)"


# --------------------------------------------------------------- RUN + CHECK
for cnd in CONDITIONS:
    if cnd.get("no_diffusion"):
        cnd["D"] = 0.0
        cnd["t_end"] = SHOW_YEARS * YEAR
        z = np.linspace(0, DOMAIN_MM, N_ELEM + 1)
        t = np.array([0.0, cnd["t_end"]])
        H = np.zeros((2, len(z))); H[:, 0] = 1.0     # drug only at the surface
        cnd["z"], cnd["t"], cnd["H"] = z, t, H
        cnd["fem_arrival"] = {L: np.nan for L in DEPTHS_MM}
        continue
    cnd["D"] = (1.0 / Z_FRONT) ** 2 / cnd["t_1mm"]                # mm^2/s
    cnd["t_end"] = RUN_PAST * cnd["t_25mm_given"]
    cnd["z"], cnd["t"], cnd["H"] = solve(cnd["D"], cnd["t_end"])
    cnd["fem_arrival"] = {L: arrival(cnd["z"], cnd["t"], cnd["H"], L) for L in DEPTHS_MM}

print(f"{'condition':<52}{'D (m^2/s)':>11} | {'1 mm given':>11}{'FEM':>11} | "
      f"{'2.5 mm given':>13}{'FEM':>11}")
print("-" * 114)
for c in CONDITIONS:
    if c.get("no_diffusion"):
        print(f"{c['label']:<52}{'0':>11} | {'never':>11}{'never':>11} | "
              f"{'never':>13}{'never':>11}")
        continue
    print(f"{c['label']:<52}{c['D'] * 1e-6:11.2e} | {fmt_dur(c['t_1mm']):>11}"
          f"{fmt_dur(c['fem_arrival'][1.0]):>11} | {fmt_dur(c['t_25mm_given']):>13}"
          f"{fmt_dur(c['fem_arrival'][2.5]):>11}")
print("\nArrival = drug at that depth reaches 50% of the surface concentration.")

# ---------------------------------------------------------------------- GIFS
tissue = LinearSegmentedColormap.from_list(
    "dermis_drug", ["#f6dcd2", "#e7a9b8", "#b0679f", "#6a3a8f", "#2b1a5e"])
ZMAX = max(DEPTHS_MM) * 1.1


def sample(t, H, tf):
    k = int(np.clip(np.searchsorted(t, tf), 1, len(t) - 1))
    w = np.clip((tf - t[k - 1]) / (t[k] - t[k - 1]), 0, 1)
    return (1 - w) * H[k - 1] + w * H[k]


def make_gif(cnd):
    z, t, H, t_end = cnd["z"], cnd["t"], cnd["H"], cnd["t_end"]
    blocked = bool(cnd.get("no_diffusion"))
    n_frames = GIF_SECONDS * FPS
    t_frames = np.linspace(0, t_end, n_frames)
    speed = t_end / GIF_SECONDS

    fig = plt.figure(figsize=(10, 7.6), dpi=72, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.32, 1], width_ratios=[1, 1, 1.5],
                          hspace=0.12, wspace=0.35, left=0.07, right=0.96,
                          top=0.97, bottom=0.09)
    head = fig.add_subplot(gs[0, :]); head.axis("off")
    head.text(0, 0.88, cnd["label"], fontsize=15, fontweight="bold", va="top")
    clock = head.text(0, 0.38, "", fontsize=24, fontweight="bold",
                      family="monospace", va="center")
    model_txt = ("No diffusion, D = 0" if blocked else
                 f"Fickian model, D = {cnd['D'] * 1e-6:.2e} m²/s")
    head.text(1, 0.38, f"GIF clock: 1 s = {fmt_dur(speed)}\n" + model_txt,
              fontsize=10, color="gray", ha="right", va="center")

    cols = []
    for j, L in enumerate(DEPTHS_MM):
        ax = fig.add_subplot(gs[1, j])
        ax.set_xlim(0, 1); ax.set_ylim(ZMAX, -0.32); ax.set_xticks([])
        ax.add_patch(plt.Rectangle((0.2, -0.28), 0.6, 0.26, color=tissue(1.0)))
        ax.text(0.5, -0.15, "drug", color="white", ha="center", va="center",
                fontsize=9, fontweight="bold")
        if blocked:                                   # drug held at the surface
            ax.plot([0.12, 0.88], [0, 0], color="#c62828", lw=4, zorder=5)
            ax.plot([0.38, 0.62], [0.08, 0.32], color="#c62828", lw=4, zorder=5)
            ax.plot([0.38, 0.62], [0.32, 0.08], color="#c62828", lw=4, zorder=5)
            ax.text(0.5, 0.42, "drug cannot\nenter keloid", ha="center", va="top",
                    fontsize=9, fontweight="bold", color="#c62828", zorder=5)
        else:
            ax.annotate("", xy=(0.5, 0.18), xytext=(0.5, -0.02),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
        zz = np.linspace(0, L, 300)
        im = ax.imshow(np.zeros((300, 1)), extent=(0.2, 0.8, L, 0), aspect="auto",
                       cmap=tissue, vmin=0, vmax=1, interpolation="bilinear")
        ax.add_patch(plt.Rectangle((0.2, 0), 0.6, L, fill=False, lw=1.5, ec="#333"))
        if L < ZMAX:
            ax.add_patch(plt.Rectangle((0.2, L), 0.6, ZMAX - L, hatch="///",
                                       fill=False, ec="#bbbbbb", lw=0))
            ax.text(0.5, (L + ZMAX) / 2, "below\nkeloid", ha="center", va="center",
                    fontsize=8, color="gray")
        ax.axhline(L, xmin=0.12, xmax=0.88, color="#c62828", ls="--", lw=1.5)
        front, = ax.plot([0.8, 0.92], [0, 0], color="#111", lw=2.5)
        flab = ax.text(0.93, 0, "", ha="left", va="center", fontsize=8,
                       family="monospace")
        ax.set_title(f"{L:.1f} mm keloid", fontsize=12, fontweight="bold")
        ax.set_ylabel("depth below skin surface (mm)" if j == 0 else "")
        status = ax.text(0.5, -0.02, "", transform=ax.transAxes, ha="center",
                         va="top", fontsize=10, fontweight="bold", color="#c62828")
        cols.append((L, zz, im, front, flab, status))

    axp = fig.add_subplot(gs[1, 2])
    axp.set(xlim=(0, 1.02), ylim=(ZMAX, -0.32), xlabel="drug concentration / surface",
            title="Concentration vs depth")
    for L in DEPTHS_MM:
        axp.axhline(L, color="#c62828", ls="--", lw=1)
        axp.text(1.0, L - 0.04, f"{L:.1f} mm", color="#c62828", ha="right", fontsize=8)
    axp.axvline(FRONT_LEVEL, color="gray", ls=":", lw=1)
    axp.text(FRONT_LEVEL + 0.01, -0.2, "50%", color="gray", fontsize=8)
    prof, = axp.plot([], [], color=tissue(0.8), lw=3)
    dot, = axp.plot([], [], "o", color="#111", ms=7, label="50% front")
    axp.legend(loc="lower right", fontsize=8)
    axp.grid(alpha=0.25)

    def frame(i):
        tf = t_frames[i]
        c = sample(t, H, tf)
        clock.set_text("t = " + fmt_clock(tf, t_end))
        zf = Z_FRONT * np.sqrt(cnd["D"] * tf)              # 50% front depth
        for L, zz, im, front, flab, status in cols:
            im.set_data(np.interp(zz, z, c)[:, None])
            if blocked:
                front.set_ydata([0, 0]); flab.set_y(0); flab.set_text("")
                status.set_text("penetration: 0 mm"); status.set_color("#c62828")
                continue
            show = min(zf, L)
            front.set_ydata([show, show]); flab.set_y(show)
            flab.set_text(f"50%\n{zf:.2f}\nmm" if zf <= L else "")
            ta = cnd["fem_arrival"][L]
            status.set_text(f"reached {L:.1f} mm at {fmt_dur(ta)}" if tf >= ta
                            else "drug front moving down")
            status.set_color("#c62828" if tf >= ta else "gray")
        if blocked:                                        # step: drug only at surface
            prof.set_data([1, 0, 0], [0, 0, ZMAX])
        else:
            prof.set_data(c[z <= ZMAX], z[z <= ZMAX])
        dot.set_data([FRONT_LEVEL], [min(zf, ZMAX)])
        return []

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=1000 / FPS)
    anim.save(cnd["file"] + ".gif", writer=animation.PillowWriter(fps=FPS))
    try:
        anim.save(cnd["file"] + ".mp4", writer=animation.FFMpegWriter(
            fps=FPS, codec="libx264", bitrate=2500,
            extra_args=["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt",
                        "yuv420p", "-profile:v", "main", "-movflags", "+faststart"]))
    except Exception as err:
        print("MP4 not saved:", err)
    plt.close(fig)
    print("Saved", cnd["file"] + ".gif")


for cnd in CONDITIONS:
    make_gif(cnd)

# show the three GIFs in Colab
try:
    from IPython.display import Image, display
    for cnd in CONDITIONS:
        display(Image(filename=cnd["file"] + ".gif"))
except ImportError:
    pass
