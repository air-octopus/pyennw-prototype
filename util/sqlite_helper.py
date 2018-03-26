# coding=utf-8

class TransactionAutoCommit:
    def __init__(self, db, stepMax = 0):
        self.db = db
        self.stepMax = stepMax
        self.stepCount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.commit()

    def commit(self):
        self.db.execute('COMMIT TRANSACTION')
        self.db.execute('BEGIN TRANSACTION')
        # self.db.commit()
        self.stepCount = 0

    def step(self):
        if self.stepMax > 0:
            self.stepCount += 1
            if self.stepCount > self.stepMax:
                self.commit()