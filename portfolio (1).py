import numpy as np
np.random.seed(99)

class Portfolio:
    vec = None

    def __init__(self):
        self.epsilon = 15
        self.window = 9
        self.flag = True


    def get_portfolio(self, data):
        if self.flag:
            self.vec = np.ones(len(data['Adj Close'].columns)) / len(data['Adj Close'].columns)
            self.flag = False
        data = data['Adj Close']
        end_window = data.iloc[-self.window:].sum()
        prediction = end_window/data.iloc[-1]
        next_weights_olmar = self.update(prediction)
        self.vec = next_weights_olmar
        return self.vec


    def update(self, prediction):
        x_pred_mean = np.mean(prediction)
        lam = max(0, (self.epsilon - np.dot(self.vec, prediction)) / np.linalg.norm(prediction - x_pred_mean) ** 2)
        lam = min(100000, lam)
        next_b = self.vec + lam * (prediction - x_pred_mean)
        s_b = sorted(next_b)
        pos = [b for b in s_b if b > 0]
        neg = [b for b in s_b if b <= 0]
        med_neg, med_pos = 0,0
        if len(pos)>0:
            med_pos = pos[int(len(pos)/2)]
        if len(neg)>0:
            med_neg = neg[int(len(neg)/2)]
        new_b = []
        for n in neg:
            if n<med_neg:
                new_b.append(-1)
            else:
                new_b.append(0)
        for p in pos:
            if p<med_pos:
                new_b.append(0)
            else:
                new_b.append(1)
        if np.isclose(sum(new_b), 0):
          return next_b/sum(next_b)
        final_b = np.array(new_b)/sum(new_b)
        if not np.isclose(final_b.sum(), 1):
          return next_b/sum(next_b)
        else:
          return final_b