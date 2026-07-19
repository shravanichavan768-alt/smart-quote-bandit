from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from base_price import estimate_base_price  

import models
from database import engine, SessionLocal
from pricing import get_price_curve, update_bandit, BASE_PRICE_DEFAULT

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class QuoteRequest(BaseModel):
    description: str
    user_id: str = "demo_user"
    base_price: Optional[float] = None

class OutcomeUpdate(BaseModel):
    project_id: int
    outcome: str          
    final_price: Optional[float] = None

@app.post("/quote")
def get_quote(req: QuoteRequest, db: Session = Depends(get_db)):
    if req.base_price:
       base_price = req.base_price
    else:
       price_info = estimate_base_price(req.description)
       base_price = price_info["estimated_base_price"]
    result = get_price_curve(req.user_id, req.description, base_price)

    project = models.Project(
        user_id=req.user_id,
        description=req.description,
        arm=result["recommended"]["arm"],
        base_price=base_price,
        quoted_price=result["recommended"]["price"],
        outcome="pending",
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return {
        "project_id": project.id,
        "curve": result["curve"],
        "recommended_price": result["recommended"]["price"],
    }

@app.post("/outcome")
def log_outcome(update: OutcomeUpdate, db: Session = Depends(get_db)):
    project = db.query(models.Project).filter(models.Project.id == update.project_id).first()
    if not project:
        return {"error": "project not found"}

    project.outcome = update.outcome
    project.final_price = update.final_price
    db.commit()

    price_used = project.final_price or project.quoted_price
    reward = (price_used / project.base_price) if update.outcome == "won" else 0.0

    update_bandit(project.user_id, project.description, project.arm, reward)

    return {"status": "updated", "reward_applied": reward}

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(models.Project).all()

@app.get("/history/{user_id}")
def get_user_history(user_id: str, db: Session = Depends(get_db)):
    return db.query(models.Project).filter(models.Project.user_id == user_id).all()


@app.post("/explain")
def explain_quote(req: QuoteRequest, db: Session = Depends(get_db)):
    from pricing import LinUCB, PRICE_MULTIPLIERS
    from features import extract_features
    import numpy as np

    bandit = LinUCB.load_or_create(req.user_id)
    x = extract_features(req.description)
    means, ucbs = bandit.all_estimates(x)

    base_price = req.base_price or BASE_PRICE_DEFAULT
    best_arm = int(np.argmax(ucbs))

   
    confidence_gap = float(ucbs[best_arm] - means[best_arm])
    confidence_score = round(max(0.0, 1.0 - confidence_gap), 3)  

   
    past = db.query(models.Project).filter(models.Project.user_id == req.user_id).all()
    desc_words = set(req.description.lower().split())
    similar = []
    for p in past:
        overlap = len(desc_words & set(p.description.lower().split()))
        if overlap > 0:
            similar.append({
                "description": p.description,
                "quoted_price": p.quoted_price,
                "outcome": p.outcome,
                "overlap_score": overlap,
            })
    similar.sort(key=lambda s: -s["overlap_score"])

    return {
        "recommended_price": round(base_price * PRICE_MULTIPLIERS[best_arm], 2),
        "confidence_score": confidence_score,
        "similar_past_projects": similar[:3],
        "reasoning": f"Recommended {PRICE_MULTIPLIERS[best_arm]}x base price based on "
                     f"{len(past)} past projects for this user, with confidence {confidence_score}."
    }

@app.get("/metrics/{user_id}")
def get_metrics(user_id: str, db: Session = Depends(get_db)):
    projects = db.query(models.Project).filter(
        models.Project.user_id == user_id,
        models.Project.outcome != "pending"
    ).all()

    if not projects:
        return {"message": "No completed projects yet for this user."}

    total = len(projects)
    wins = [p for p in projects if p.outcome == "won"]
    win_rate = len(wins) / total

    actual_revenue = sum((p.final_price or p.quoted_price) for p in wins)

    baseline_wins = [p for p in projects if p.outcome == "won"]  # same outcomes, hypothetical fixed price
    baseline_revenue = sum(p.base_price for p in baseline_wins)

    return {
        "total_projects": total,
        "win_rate": round(win_rate, 3),
        "actual_revenue_with_bandit": round(actual_revenue, 2),
        "naive_baseline_revenue": round(baseline_revenue, 2),
        "revenue_lift_pct": round(((actual_revenue - baseline_revenue) / baseline_revenue) * 100, 2) if baseline_revenue > 0 else None,
    }

@app.get("/metrics/{user_id}")
def get_metrics(user_id: str, db: Session = Depends(get_db)):
    projects = db.query(models.Project).filter(
        models.Project.user_id == user_id,
        models.Project.outcome != "pending"
    ).all()

    if not projects:
        return {"message": "No completed projects yet for this user."}

    total = len(projects)
    wins = [p for p in projects if p.outcome == "won"]
    win_rate = len(wins) / total

    actual_revenue = sum((p.final_price or p.quoted_price) for p in wins)
    baseline_revenue = sum(p.base_price for p in wins)  # naive: always quote flat base_price

    return {
        "total_projects": total,
        "win_rate": round(win_rate, 3),
        "actual_revenue_with_bandit": round(actual_revenue, 2),
        "naive_baseline_revenue": round(baseline_revenue, 2),
        "revenue_lift_pct": round(((actual_revenue - baseline_revenue) / baseline_revenue) * 100, 2) if baseline_revenue > 0 else None,
    }