import numpy as np
import os
from features import extract_features, CONTEXT_DIM

PRICE_MULTIPLIERS = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0]
N_ARMS = len(PRICE_MULTIPLIERS)
BASE_PRICE_DEFAULT = 500.0
STATE_DIR = "bandit_states"

os.makedirs(STATE_DIR, exist_ok=True)


class LinUCB:
    def __init__(self, n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=0.8):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        self.A = [np.identity(self.d) for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]

    def select_arm(self, x):
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
        max_p = np.max(p)
        best_arms = np.where(p == max_p)[0]
        return int(np.random.choice(best_arms))

    def all_estimates(self, x):
        means, ucbs = [], []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = theta @ x
            bonus = self.alpha * np.sqrt(x @ A_inv @ x)
            means.append(mean)
            ucbs.append(mean + bonus)
        return np.array(means), np.array(ucbs)

    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def save(self, user_id: str):
        path = os.path.join(STATE_DIR, f"{user_id}.npz")
        np.savez(path, A=np.array(self.A), b=np.array(self.b), alpha=self.alpha)

    @classmethod
    def load_or_create(cls, user_id: str):
        path = os.path.join(STATE_DIR, f"{user_id}.npz")
        if os.path.exists(path):
            data = np.load(path)
            obj = cls(alpha=float(data["alpha"]))
            obj.A = [data["A"][i] for i in range(N_ARMS)]
            obj.b = [data["b"][i] for i in range(N_ARMS)]
            return obj
        return cls()


def get_price_curve(user_id: str, description: str, base_price: float = BASE_PRICE_DEFAULT):
    bandit = LinUCB.load_or_create(user_id)
    x = extract_features(description)
    means, ucbs = bandit.all_estimates(x)

    curve = []
    for i, mult in enumerate(PRICE_MULTIPLIERS):
        price = base_price * mult
        win_prob = float(np.clip(means[i], 0, 1))  
        curve.append({"price": round(price, 2), "win_probability": round(win_prob, 3)})

    best_arm = int(np.argmax(ucbs))
    recommended = {"arm": best_arm, "price": round(base_price * PRICE_MULTIPLIERS[best_arm], 2)}

    return {"curve": curve, "recommended": recommended}


def update_bandit(user_id: str, description: str, arm: int, reward: float):
    bandit = LinUCB.load_or_create(user_id)
    x = extract_features(description)
    bandit.update(arm, x, reward)
    bandit.save(user_id)