from mimetypes import init


class Experiment:
    def __init__(self, time_unit, horizion):
        self.time_unit = time_unit
        self.horizon = horizion
        self.name = f"{horizion}"
    
    def _convert_unit(self):
        converter = {
            'H':'hours',
            'M':'months',
            'Y':'years',
            'T':'minutes',
            'D':'days'
        }

        return converter[self.time_unit]

    def get_time_unit(self):
        return self.time_unit
    
    def get_horizon(self):
        return f"{self.horizon} {self._convert_unit()}"
        
    def print_setup(self):
        print(f"About to execute for '{self.get_horizon()}'")
    
    def get_initial(self, end_date, start_date, forecast_count=32):
        period = end_date - start_date
        period_in_hours = (period.seconds / 60 / 60) + (period.days * 24)
        forecast_count = 32

        if self.time_unit=='H':
            convert_to_correct_unit = lambda value : value
            convert_back = lambda value : value
        elif self.time_unit == 'T':
            convert_to_correct_unit = lambda value : value / 60
            convert_back = lambda value : value * 60
        elif self.time_unit == 'D':
            convert_to_correct_unit = lambda value : value * 24
            convert_back = lambda value : value / 24
        elif self.time_unit == 'M':
            convert_to_correct_unit = lambda value : value * 30 * 24
            convert_back = lambda value : value / 30 / 24
        elif self.time_unit == 'Y':
            convert_to_correct_unit = lambda value : value * 365
            convert_back = lambda value : value / 365
        else:
            raise Exception(f"Unit type {self.time_unit} is not supported")

        initial = period_in_hours - (convert_to_correct_unit(self.horizon) * forecast_count)

        return convert_back(initial)
    
    def get_str_initial(self, end_date, start_date, forecast_count=32):
        initial = int(self.get_initial(end_date, start_date, forecast_count=forecast_count))

        return f"{initial} {self._convert_unit()}"

        



def get_experiments():
    return [
        # Experiment('T', '15 minutes'),
        # Experiment('T', '30 minutes'),
        # Experiment('T', '60 minutes'),
        Experiment('H', '1 hours'),
        # Experiment('H', '6 hours'),
        # Experiment('H', '12 hours'),
        Experiment('H', '24 hours'),
        Experiment('D', '7 days'),
        # Experiment('D', '14 days'),
        Experiment('D', '30 days'),
        # Experiment('D', '60 days'),
        Experiment('D', '180 days'),
        Experiment('D', '365 days'),
    ]