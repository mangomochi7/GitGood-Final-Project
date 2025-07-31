class Database:

    def __init__(self):
        self.db = {}

    def add_participant(self, id):
        self.db[id] = 0

    # if sad, angry, fearful, or disgusted
    def negative_detected(self, id):
        self.db[id] += 1

    def reset(self):
        for id in self.db.keys():
            self.db[id] = 0

    def get_count(self, id):
        return self.db[id]