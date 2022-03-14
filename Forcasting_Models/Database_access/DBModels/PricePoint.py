class PricePoint:
    id = ""
    identifier = ""
    close = ""
    high = ""
    interest = ""
    low = ""
    open = ""
    time = ""
    volume = ""
    def __init__(self, id, identifier, close, high, interest, low, open, time, volume):
        self.id = id
        self.identifier = identifier
        self.close = close
        self.high = high
        self.interest = interest
        self.low = low
        self.open = open
        self.time = time
        self.volume = volume