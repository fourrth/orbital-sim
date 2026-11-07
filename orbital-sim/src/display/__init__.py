"""This is a simple display for the N-Body sim before I write the rust implementation"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import orbital_sim as orb

def main():
    AU = 1.496e11  # meters

    sim = orb.Simulation()
    sim.add_body(orb.Body("Sun", 1.989e30, (0, 0, 0), fixed=False))
    sim.add_circular_satellite("Mercury", 3.301e23, "Sun", 0.387 * AU, phase=0.0,inclination=1.0)
    sim.add_circular_satellite("Venus",   4.867e24, "Sun", 0.723 * AU, phase=2.1,inclination=1.0)
    sim.add_circular_satellite("Earth",   5.972e24, "Sun", 1.000 * AU, phase=4.0,inclination=1.0)
    sim.add_circular_satellite("Mars",    6.417e23, "Sun", 1.524 * AU, phase=5.4,inclination=1.0)

    COLORS = {"Mercury": "silver", "Venus": "orange",
            "Earth": "deepskyblue", "Mars": "indianred"}
    DOTSIZE = {"Mercury": 4, "Venus": 7, "Earth": 7, "Mars": 5}
    TRAIL = 250  # frames of trail to keep bc it looks cool

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_xlim(-1.75 * AU, 1.75 * AU)
    ax.set_ylim(-1.75 * AU, 1.75 * AU)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    ax.plot(0, 0, "o", color="gold", markersize=16, zorder=3)

    dots = {}
    trails = {}
    trail_x = {}
    trail_y = {}


    for name in COLORS:
        dots[name], = ax.plot([], [], "o", color=COLORS[name],
                            markersize=DOTSIZE[name], zorder=2)
        trails[name], = ax.plot([], [], "-", color=COLORS[name],
                                lw=0.6, alpha=0.55, zorder=1)
        trail_x[name], trail_y[name] = [], []

    # two days per dt
    DT = 2 * 60*60*24
    SUBSTEPS = 10
    FRAMES = 60000

    def init():
        for a in (*dots.values(), *trails.values()):
            a.set_data([], [])
        return (*dots.values(), *trails.values())

    def update(_frame):
        sim.step(DT, substeps=SUBSTEPS)
        for b in sim.bodies:
            if b.name not in COLORS:
                continue
            x, y = b.position[0], b.position[1]
            dots[b.name].set_data([x], [y])
            trail_x[b.name].append(x)
            trail_y[b.name].append(y)
            if len(trail_x[b.name]) > TRAIL:
                trail_x[b.name].pop(0)
                trail_y[b.name].pop(0)
            trails[b.name].set_data(trail_x[b.name], trail_y[b.name])
        return (*dots.values(), *trails.values())

    ani = FuncAnimation(fig, update, frames=FRAMES, init_func=init,
                        blit=True, interval=25, repeat=False)
    plt.show()