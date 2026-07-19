
import numpy as np
from pricing import update_bandit, PRICE_MULTIPLIERS, BASE_PRICE_DEFAULT
from database import SessionLocal, engine
import models

models.Base.metadata.create_all(bind=engine)

np.random.seed(1)
USER_ID = "demo_user"

SEED_PROJECTS = [
    ("Build a React dashboard for an enterprise client, scalable architecture needed", "enterprise"),
    ("Machine learning integration for a large corporation, complex data pipeline", "enterprise"),
    ("Enterprise CRM integration, long-term ongoing retainer contract", "enterprise"),
    ("Scalable backend architecture for a fortune 500 company", "enterprise"),
    ("Complex AI-powered analytics dashboard for a corporation", "enterprise"),
    ("Ongoing retainer work for a startup, monthly maintenance", "startup"),
    ("Early-stage startup MVP, small team, moving fast", "startup"),
    ("Seed-stage startup needs a landing page and basic backend", "startup"),
    ("Startup dashboard, long-term engagement expected", "startup"),
    ("Quick landing page fix, simple task, urgent turnaround", "simple"),
    ("Simple bug fix on an existing website, small team, quick job", "simple"),
    ("Basic WordPress landing page for a small business, asap", "simple"),
    ("Small fix needed immediately, simple one-page site", "simple"),
    ("Quick urgent task, basic HTML page update", "simple"),
    ("Simple contact form addition, rush job", "simple"),
]

def true_win_prob(project_type, multiplier):
    sensitivity = {"enterprise": 0.6, "startup": 0.4, "simple": 0.2}[project_type]
    return 1 / (1 + np.exp((multiplier - 1.0) / sensitivity))

def simulate_outcome(project_type, multiplier):
    return np.random.rand() < true_win_prob(project_type, multiplier)

def seed():
    db = SessionLocal()
    print(f"Seeding {len(SEED_PROJECTS) * 5} synthetic outcomes for user '{USER_ID}'...")

    for description, ptype in SEED_PROJECTS:
        for _ in range(5):  
            multiplier = np.random.choice(PRICE_MULTIPLIERS)
            arm = PRICE_MULTIPLIERS.index(multiplier)
            won = simulate_outcome(ptype, multiplier)
            reward = multiplier if won else 0.0

            update_bandit(USER_ID, description, arm, reward)

            project = models.Project(
                user_id=USER_ID,
                description=description,
                arm=arm,
                base_price=BASE_PRICE_DEFAULT,
                quoted_price=BASE_PRICE_DEFAULT * multiplier,
                outcome="won" if won else "lost",
                final_price=BASE_PRICE_DEFAULT * multiplier if won else None,
            )
            db.add(project)

    db.commit()
    db.close()
    print("Seeding complete: bandit_states/demo_user.npz updated + DB rows inserted.")

if __name__ == "__main__":
    seed()