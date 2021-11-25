class Counter:
    def __init__(self, start):
        self.n = start

    def get_count(self):
        return self.n

    def increment(self, k=1, return_when='after'):
        self.n += k

        if return_when == 'after':
            return self.n
        elif return_when == 'before':
            return self.n - k
        else:
            raise ValueError(str(return_when) + " is not a valid value. Try 'after' or 'before'.")

    def decrement(self, k=1, return_when='after'):
        old = self.n
        self.n = max(0, self.n - k)

        if return_when == 'after':
            return self.n
        elif return_when == 'before':
            return old
        else:
            raise ValueError(str(return_when) + " is not a valid value. Try 'after' or 'before'.")