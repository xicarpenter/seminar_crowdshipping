from collect_lines import generate, generate_times
import random


class InstanceGenerator:
    def __init__(self, num_crowdshippers, num_parcels, seed=42):
        self.num_crowdshippers = num_crowdshippers
        self.num_parcels = num_parcels
        self.seed = seed

        self.lines, self.stations = generate()
        self.times = generate_times(self.lines)

        self.I = [f"C{i}" for i in range(1, self.num_crowdshippers + 1)]
        self.J = [f"P{j}" for j in range(1, self.num_parcels + 1)]
        self.S = list(self.stations.keys())

        self.l = {s: random.randint(1, 4) for s in self.S}

    
    def return_kwargs(self):
        return {
            "I": self.I,
            "J": self.J,
            "S": self.S,
            "alpha": self.lines,
            "omega": self.lines,
            "r": self.times,
            "d": self.times,
            "p": self.p,
            "t": self.t,
            "f": self.f,
            "l": self.l,
            "seed": self.seed
        }


if __name__ == "__main__":
    num_crowdshippers = 10
    num_parcels = 10
    generator = InstanceGenerator(num_crowdshippers, num_parcels)
    print(generator.I)
    print(generator.J)
    print(generator.S)
