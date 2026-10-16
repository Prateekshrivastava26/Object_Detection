class SuspicionEngine:
    def __init__(self):
        self.suspicion_score = 0

    def increase(self, amount=1):
        self.suspicion_score += amount

    def reset(self):
        self.suspicion_score = 0

    def get_score(self):
        return self.suspicion_score