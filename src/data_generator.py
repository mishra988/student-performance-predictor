import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples: int = 2000, random_state: int = 42, output_path: str = None) -> pd.DataFrame:
    """
    Generates a realistic student academic performance dataset with non-linear
    relationships, interactions, and educational behavioral patterns.
    """
    np.random.seed(random_state)

    # 1. Independent features
    student_ids = [f"STU_{10000 + i}" for i in range(n_samples)]
    
    # Hours studied per week (clipped 2 to 35 hours)
    hours_studied = np.clip(np.random.normal(loc=15.0, scale=6.5, size=n_samples), 2.0, 36.0)
    
    # Attendance percentage (clipped 45% to 100%)
    attendance_rate = np.clip(np.random.beta(a=7, b=2, size=n_samples) * 100, 45.0, 100.0)
    
    # Previous semester score (40 to 100)
    previous_score = np.clip(np.random.normal(loc=72.0, scale=12.0, size=n_samples), 38.0, 99.0)
    
    # Daily sleep hours (4 to 10 hours)
    sleep_hours = np.clip(np.random.normal(loc=7.2, scale=1.2, size=n_samples), 4.0, 10.0)
    
    # Tutoring sessions per month (0 to 6)
    tutoring_sessions = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.35, 0.25, 0.20, 0.12, 0.05, 0.03])
    
    # Parental education level
    parental_edu_levels = ["High School", "Some College", "Bachelor's", "Master's", "Doctorate"]
    parental_education = np.random.choice(parental_edu_levels, size=n_samples, p=[0.25, 0.30, 0.28, 0.13, 0.04])
    
    # Categorical access and participation factors
    internet_access = np.random.choice(["Yes", "No"], size=n_samples, p=[0.88, 0.12])
    extracurricular = np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45])
    peer_study_group = np.random.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58])
    
    # Subject stress level (1 to 10 scale)
    stress_level = np.random.randint(1, 11, size=n_samples)

    # 2. Compute Target (Final Exam Score) with realistic multi-factor non-linear logic
    # Base baseline
    base_score = 15.0
    
    # Prior performance contribution
    score_from_prev = 0.38 * previous_score
    
    # Study hours with diminishing returns: log curve
    score_from_study = 9.5 * np.log1p(hours_studied * 0.9)
    
    # Attendance impact: threshold penalty below 75%
    attendance_factor = np.where(attendance_rate >= 75.0, 0.25 * attendance_rate, 0.35 * attendance_rate - 12.0)
    
    # Sleep curve (optimal between 7 and 8.5 hours; steep drops below 6 or above 9.5)
    sleep_effect = -1.8 * ((sleep_hours - 7.6) ** 2) + 4.5
    
    # Tutoring boost
    tutoring_boost = tutoring_sessions * 2.2
    
    # Stress penalty (exponential degradation above stress level 6)
    excess_stress = np.maximum(0, stress_level - 6)
    stress_penalty = np.where(stress_level > 6, (excess_stress ** 1.35) * -2.1, 0.0)
    
    # Categorical bonuses
    edu_weights = {"High School": 0.0, "Some College": 1.5, "Bachelor's": 3.2, "Master's": 4.8, "Doctorate": 5.5}
    edu_boost = np.array([edu_weights[e] for e in parental_education])
    
    internet_boost = np.where(internet_access == "Yes", 2.2, -3.0)
    peer_boost = np.where(peer_study_group == "Yes", 1.8, 0.0)
    extracurricular_boost = np.where(extracurricular == "Yes", 1.2, 0.0)
    
    # Noise (random test variability)
    random_noise = np.random.normal(loc=0.0, scale=3.2, size=n_samples)

    # Calculate final continuous score
    raw_final_score = (
        base_score
        + score_from_prev
        + score_from_study
        + attendance_factor
        + sleep_effect
        + tutoring_boost
        + stress_penalty
        + edu_boost
        + internet_boost
        + peer_boost
        + extracurricular_boost
        + random_noise
    )
    
    final_score = np.clip(np.round(raw_final_score, 1), 20.0, 100.0)

    # Categorical Performance Tier
    def categorize(score):
        if score >= 85.0:
            return "Distinction"
        elif score >= 70.0:
            return "Merit"
        elif score >= 50.0:
            return "Pass"
        else:
            return "At-Risk"

    performance_tier = [categorize(s) for s in final_score]

    df = pd.DataFrame({
        "student_id": student_ids,
        "hours_studied": np.round(hours_studied, 1),
        "attendance_rate": np.round(attendance_rate, 1),
        "previous_score": np.round(previous_score, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "tutoring_sessions": tutoring_sessions,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "extracurricular_activities": extracurricular,
        "peer_study_group": peer_study_group,
        "stress_level": stress_level,
        "final_score": final_score,
        "performance_tier": performance_tier
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[OK] Generated {n_samples} student records saved to: {output_path}")

    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(current_dir, "..", "data", "raw", "student_data.csv")
    generate_student_dataset(n_samples=2500, output_path=target_csv)
