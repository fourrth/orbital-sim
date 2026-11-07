"""
Orbital, N-Body Simulator
Units are SI: meters, kilograms, seconds.

Essentially, just a collection with methods using:
F = G (m1 m2) /r^2 in the direction of the two masses
you can convert this to find acceleration (like we do below)
and also there you can put a softening term is useful if you
have objects going inside each other because without it, 
you could have a divide by 0 and the accel-->inf.

"""

import math
from dataclasses import dataclass

import numpy as np

G_SI = 6.67430e-11  # m^3 kg^-1 s^-2


@dataclass
class Body:
    """Simple Body class"""
    name: str
    mass: float
    position: np.ndarray | tuple | list
    velocity: np.ndarray | tuple | list = (0.0, 0.0, 0.0)
    radius: float = 0.0 # only for display/collisions maybe in future
    fixed: bool = False # for like the sun so it doesn't wobble during the orbit

    def __post_init__(self) -> None:
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)

        if self.position.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError(f"{self.name}: position and velocity must be 3-vectors")


class Simulation:
    """
    Owns the bodies and advances them in time.
    """

    def __init__(self, G: float = G_SI, softening: float = 0.0) -> None:
        self.G = float(G)
        self.softening = float(softening)
        self.bodies: list[Body] = []
        self.time: float = 0.0

    # ---- assembly -------------------------------------------------------

    def add_body(self, body: Body) -> Body:
        self.bodies.append(body)
        return body

    def get(self, name: str) -> Body:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    def remove(self, name: str) -> None:
        self.bodies = [b for b in self.bodies if b.name != name]

    def add_circular_satellite(
        self,
        name: str,
        mass: float,
        primary: str | Body,
        distance: float,
        phase: float = 0.0,
        radius: float = 0.0,
        inclination:float = 0.0,
    ) -> Body:
        """
        Adds a body in a circular orbit around primary.
        This just does the definition of centripetal acceleration
        and uses that in the force formula.
        In the simulation without fixed, nothing moves in a complete circle
        because (assuming sun and planet) the planet pulls on the sun too,
        so the sun moves and changes the force on the planet and everything
        wobbles a little off of the actual circular path. But that's not too
        important for simple displays"""
        if isinstance(primary, str):
            primary = self.get(primary)
        
        ci, si = math.cos(inclination), math.sin(inclination)

        speed = math.sqrt(self.G * primary.mass / distance)
        c, s = math.cos(phase), math.sin(phase)
        pos = primary.position + distance * np.array([c, s, 0.0])
        vel = primary.velocity + speed * np.array([-s, c, 0.0])
        pos[1], pos[2] = ci * pos[1] - si * pos[2], si * pos[1] + ci * pos[2]
        vel[1], vel[2] = ci * vel[1] - si * vel[2], si * vel[1] + ci * vel[2]
        return self.add_body(Body(name, mass, pos, vel, radius=radius))

    def _accelerations(self) -> np.ndarray:
        """Gravitational acceleration on every body, vec 3."""
        n = len(self.bodies)
        acc = np.zeros((n, 3))
        if n < 2:
            return acc
        # sometimes array comprehension is nice to have
        pos = np.array([b.position for b in self.bodies])
        mass = np.array([b.mass for b in self.bodies])
        eps2 = self.softening ** 2

        for i in range(n):
            # shape (n,) broadcasted like (n,3) for rj & ri
            # again, not sure if I like that though
            d = pos - pos[i]
            # this function is fun. this translates to:
            # for arrays indexed by ij, do i * j and put into 
            # an array indexed by i. this is mul_inner^2
            r2 = np.einsum("ij,ij->i", d, d) + eps2
            r2[i] = np.inf
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))
            acc[i] = self.G * (d * (mass * inv_r3)[:, None]).sum(axis=0)
        return acc

    def step(self, dt: float, substeps: int = 1) -> None:
        """
        Advance by dt seconds using substeps of that dt. This uses the Verlet method,
        which separates the velocity and position updates 
        into different steps to prevent drift. This method is guaranteed 
        to make sure that the energy of the system is constrained to a 
        constant width band (see the test script to see how we test for a small offset,
        because it shouldn't --> 0, it will go to some small energy band).
        Name for this is symplectic.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        h = dt / substeps
        # probably a better & cooler way to do this, but good enough
        acc = self._accelerations()
        for _ in range(substeps):
            for b, a in zip(self.bodies, acc):
                if not b.fixed:
                    b.velocity += 0.5 * a * h 
                    b.position += b.velocity * h

            acc = self._accelerations()

            for b, a in zip(self.bodies, acc):
                if not b.fixed:
                    b.velocity += 0.5 * a * h

            self.time += h

    def snapshot(self) -> dict:
        return {
            "time": self.time,
            "bodies": [
                {
                    "name": b.name,
                    "mass": b.mass,
                    "position": b.position.copy(),
                    "velocity": b.velocity.copy(),
                    "radius": b.radius,
                }
                for b in self.bodies
            ],
        }

    def total_momentum(self) -> np.ndarray:
        return sum((b.mass * b.velocity for b in self.bodies), np.zeros(3))

    def total_energy(self) -> float:
        ke = sum(0.5 * b.mass * float(b.velocity @ b.velocity) for b in self.bodies)
        pe = 0.0
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                r = float(np.linalg.norm(self.bodies[j].position - self.bodies[i].position))
                if r > 0.0:
                    pe -= self.G * self.bodies[i].mass * self.bodies[j].mass / r
        return ke + pe