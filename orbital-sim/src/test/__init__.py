"""Tests for the project."""
import numpy as np
import orbital_sim as orb
def main():

    # do we have circular orbits
    sim = orb.Simulation()
    sim.add_body(orb.Body("Sun", 1.989e30, (0, 0, 0), fixed=True))
    earth = sim.add_circular_satellite("Earth", 5.972e24, "Sun", 1.496e11)
    sun = sim.get("Sun")

    r0 = np.linalg.norm(earth.position - sun.position)
    E0 = sim.total_energy()

    # 10 years at 1hr steps 
    for _ in range(10 * 365 * 24):
        sim.step(3600.0)

    r1 = np.linalg.norm(earth.position - sun.position)
    E1 = sim.total_energy()

    print(f"[1] radius  {r0:.4e} -> {r1:.4e}   drift = {abs(r1-r0)/r0:.2e}")
    print(f"[1] energy  drift = {abs((E1-E0)/E0):.2e}")
    assert abs(r1 - r0) / r0 < 1e-4, "orbit is not circular"
    assert abs((E1 - E0) / E0) < 1e-6, "energy drifted... dt could be too large"

    # check if momentum is conserved
    sim2 = orb.Simulation()
    # equal and opposite reactions... so nothing should happen
    sim2.add_body(orb.Body("A", 1.0e20, (0, 0, 0),      (0, -2.5e4, 0)))
    sim2.add_body(orb.Body("B", 1.0e20, (1.0e10, 0, 0), (0, +2.5e4, 0)))

    def barycenter(s):
        m = sum(b.mass for b in s.bodies)
        return sum((b.mass * b.position for b in s.bodies), np.zeros(3)) / m

    p0 = sim2.total_momentum()
    bc0 = barycenter(sim2)

    for _ in range(2000):
        sim2.step(60.0, substeps=4)

    p1 = sim2.total_momentum()
    bc1 = barycenter(sim2)

    p_drift  = np.linalg.norm(p1 - p0)
    bc_drift = np.linalg.norm(bc1 - bc0)
    print(f"[2]  momentum drift   = {p_drift:.2e} kg·m/s")
    print(f"[2]  barycenter drift = {bc_drift:.2e} m")
    assert p_drift  < 1e12, "total momentum not conserved"
    assert bc_drift < 1e6,  "zero momentum means the barycenter should be fixed"


    # does a fixed body move?
    sim3 = orb.Simulation()
    sim3.add_body(orb.Body("Pinned", 1e30, (0, 0, 0), fixed=True))
    sim3.add_body(orb.Body("Free",   1e20, (1e10, 0, 0), (0, 1e4, 0)))

    pinned_before = sim3.get("Pinned").position.copy()
    for _ in range(100):
        sim3.step(60.0)

    assert np.array_equal(sim3.get("Pinned").position, pinned_before), \
        "body with fixed=True moved"
    print("[2b] fixed body stayed pinned")

    print("\nAll checks passed!")