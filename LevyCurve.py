import math
import numpy as np
import matplotlib.pyplot as plt

class LevyCCurve:
    """
    Lévy C-curve generator.
    
    Parameters
    ----------
    method : str
        Algorithm used to generate points. Options:
        - "lsystem"     : L-system approach (default)
        - "ifs"         : Iterated Function System (chaos game)
        - "ifs-modified": IFS with changed matrixes
    iterations : int, optional
        Depth of recursion in L-system
    angle_deg : float, optional
        Angle used by the L-system (default: 45°)
    lsystem_rules : dict, optional
        Rules for replacing symbols in the L-system
        Example: {"F": "+F--F+"} (default for Lévy C-curve)
    n_pouint : int, optional
        Number of points generated in IFS methods
    """

    def __init__(self, method="lsystem", iterations=12, angle_deg=45.0, lsystem_rules=None, n_points=50000):
        self.method = method.lower()
        self.iterations = iterations
        self.angle_deg = angle_deg
        self.instructions = ""
        self.points = []
        self.lsystem_rules = lsystem_rules or {"F": "+F--F+"}
        self.n_points = n_points

    # ==============================
    # Public Methods
    # ==============================
    def generate(self):
        """Generate points based on selected method."""
        if self.method == "lsystem":
            self.points = self._generate_lsystem_points()
        elif self.method == "ifs":
            self.points = self._generate_ifs_points()
        elif self.method == "ifs-modified":
            self.points = self._generate_ifs_modified()
        else:
            raise ValueError(f"Unknown method '{self.method}'. Supported: 'lsystem', 'ifs', 'ifs-modified'")
        return self.points

    def plot(self, show=True, figsize=(6,6), title=None, linewidth = 0.8):
        """Plot generated points."""
        if not self.points:
            self.generate()

        xs, ys = zip(*self.points)
        plt.figure(figsize=figsize)
        if self.method == "ifs":
            plt.scatter(xs, ys, s=linewidth/4)
            plt.title(title or f"Lévy C-curve (method={self.method})", pad=12)
        elif self.method == "ifs-modified":
            plt.scatter(xs, ys, s=linewidth/4)
            plt.title(title or f"Lévy C-curve (method={self.method})", pad=12)
        else:
            plt.plot(xs, ys, linewidth=linewidth)
            plt.title(title or f"Lévy C-curve (method={self.method}, {self.angle_deg}) — iterations={self.iterations}", pad=12)
        plt.axis('equal')
        plt.axis('off')            
        plt.tight_layout()
        if show:
            plt.show()

    # ==============================
    # Private methods
    # ==============================
    # ---- L-system version ----
    def _generate_lsystem_points(self):
        """Generate points from L-system rules."""
        instructions = self._expand_lsystem(axiom="F", rules=self.lsystem_rules)
        return self._interpret_lsystem(instructions)

    def _expand_lsystem(self, axiom, rules):
        """Expand the L-system string over iterations."""
        s = axiom
        for _ in range(self.iterations):
            s = "".join(rules.get(ch, ch) for ch in s)
        self.instructions = s
        return s

    def _interpret_lsystem(self, instructions, step=None):
        """Interpret the L-system instructions into drawable points."""
        if step is None:
            step = 1.0 / (2 ** (self.iterations / 2))

        x, y = 0.0, 0.0
        heading = 0.0
        angle = math.radians(self.angle_deg)
        points = [(x, y)]

        for ch in instructions:
            if ch == "F":
                x += step * math.cos(heading)
                y += step * math.sin(heading)
                points.append((x, y))
            elif ch == "+":
                heading -= angle
            elif ch == "-":
                heading += angle

        return points

    # ---- IFS version ----
    def _generate_ifs_points(self, discard=50):
        """Generate points using Iterated Function System (IFS) for Lévy C-curve."""
        rng = np.random.default_rng(123)
        x = np.array([0.0, 0.0], dtype=float)

        theta = np.pi / 4 
        scale = 1 / np.sqrt(2)
        R1 = scale * np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        R2 = scale * np.array([[np.cos(-theta), -np.sin(-theta)],
                            [np.sin(-theta),  np.cos(-theta)]])
        b1 = np.array([0, 0])
        b2 = np.array([0.5, 0.5])

        transforms = [(R1, b1), (R2, b2)]
        probs = [0.5, 0.5]

        points = []
        for i in range(self.n_points + discard):
            k = rng.choice(len(transforms), p=probs)
            x = transforms[k][0] @ x + transforms[k][1]
            if i >= discard:
                points.append(x.copy())

        return points
    
    def _generate_ifs_modified(self, discard=50):
        """Generate points using Iterated Function System (IFS) for Lévy C-curve."""
        rng = np.random.default_rng(123)
        x = np.array([0.0, 0.0], dtype=float)

        theta = np.pi / 6
        scale = 1 / np.sqrt(2)
        R1 = scale * np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        R2 = scale * np.array([[np.cos(-theta), -np.sin(-theta)],
                            [np.sin(-theta),  np.cos(-theta)]])
        b1 = np.array([0, 0])
        b2 = np.array([0.5, 0.5])

        transforms = [(R1, b1), (R2, b2)]
        probs = [0.5, 0.5]

        points = []
        for i in range(self.n_points + discard):
            k = rng.choice(len(transforms), p=probs)
            x = transforms[k][0] @ x + transforms[k][1]
            if i >= discard:
                points.append(x.copy())

        return points


