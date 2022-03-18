class Experiment:
    def __init__(self, time_unit, horizion):
        self.time_unit = time_unit
        self.horizon = horizion
        self.name = f"{horizion}"
    
    def print_setup(self):
        print(f"About to execute for '{self.horizon}'")


def get_experiments():
    return [
        Experiment('T', '15 minutes'),
        Experiment('T', '30 minutes'),
        Experiment('T', '60 minutes'),
        Experiment('H', '1 hours'),
        Experiment('H', '6 hours'),
        Experiment('H', '12 hours'),
        Experiment('H', '24 hours'),
        Experiment('D', '7 days'),
        Experiment('D', '14 days'),
        Experiment('D', '30 days'),
        Experiment('D', '60 days'),
        Experiment('D', '180 days'),
        Experiment('D', '365 days'),
    ]