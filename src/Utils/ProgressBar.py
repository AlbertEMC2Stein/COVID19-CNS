from numpy import clip


class ProgressBar:
    def __init__(self, start_at: int, minimum: int, maximum: int):
        self.min = minimum
        self.max = maximum
        self.current = start_at
        self.printing = True

    def update(self, step: int):
        self.current = clip(self.current + step, self.min, self.max)
        percentage = 100 * (self.current - self.min) / (self.max - self.min)

        if int(percentage) % 5 == 0:
            if self.printing:
                self.printing = False
                p_as_int = int(percentage)
                print("Progress: %s%s (%s%%)" % (p_as_int // 5 * '#',
                                                 (20 - p_as_int // 5) * '|',
                                                 p_as_int))
        else:
            self.printing = True
