import os
import joblib
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from data_preprocessing import engineer_features

class StudentPerformancePredictor:
    def __init__(self, models_dir: str = None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        
        if models_dir is None:
            models_dir = os.path.join(project_root, "models")
            
        self.model_path = os.path.join(models_dir, "student_performance_model.joblib")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError("Model or preprocessor artifact not found. Please run train.py first.")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def _determine_tier(self, score: float) -> Dict[str, str]:
        if score >= 85.0:
            return {
                "tier": "Distinction",
                "badge": "[Top Tier] High Performer",
                "risk_level": "Minimal",
                "color": "green"
            }
        elif score >= 70.0:
            return {
                "tier": "Merit",
                "badge": "[Good] Good Standing",
                "risk_level": "Low",
                "color": "blue"
            }
        elif score >= 50.0:
            return {
                "tier": "Pass",
                "badge": "[Warning] Borderline",
                "risk_level": "Moderate",
                "color": "yellow"
            }
        else:
            return {
                "tier": "At-Risk",
                "badge": "[Alert] Critical Intervention Needed",
                "risk_level": "High",
                "color": "red"
            }

    def predict_single(self, student_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on a single student's profile, identifies key risk factors,
        strengths, and generates actionable simulated uplift scenarios.
        """
        raw_df = pd.DataFrame([student_input])
        engineered_df = engineer_features(raw_df)

        # Transform features
        X_trans = self.preprocessor.transform(engineered_df)
        predicted_score = float(np.clip(self.model.predict(X_trans)[0], 0.0, 100.0))

        tier_info = self._determine_tier(predicted_score)

        # Analyze strengths & weaknesses
        strengths = []
        weaknesses = []
        recommendations = []

        # Attendance check
        att = student_input.get("attendance_rate", 80.0)
        if att >= 88.0:
            strengths.append(f"Strong attendance record ({att:.1f}%) helps continuous mastery.")
        elif att < 75.0:
            weaknesses.append(f"Sub-optimal attendance ({att:.1f}%): Falling below 75% triggers significant academic penalties.")
            recommendations.append("Prioritize attending all weekly lecture sessions to restore baseline attendance.")

        # Study hours
        hours = student_input.get("hours_studied", 12.0)
        if hours >= 18.0:
            strengths.append(f"Diligent study schedule ({hours:.1f} hours/week).")
        elif hours < 10.0:
            weaknesses.append(f"Low self-study commitment ({hours:.1f} hours/week).")
            recommendations.append("Establish a structured daily study block to increase weekly self-study to at least 14-16 hours.")

        # Sleep check
        sleep = student_input.get("sleep_hours", 7.0)
        if 7.0 <= sleep <= 8.5:
            strengths.append(f"Healthy cognitive rest balance ({sleep:.1f} hours/night).")
        elif sleep < 6.0:
            weaknesses.append(f"Sleep deprivation ({sleep:.1f} hours/night) impairs memory consolidation.")
            recommendations.append("Maintain a consistent sleep window targeting 7.5 hours per night.")

        # Stress check
        stress = student_input.get("stress_level", 5)
        if stress >= 7:
            weaknesses.append(f"High subjective stress level ({stress}/10) degrades exam retention.")
            recommendations.append("Incorporate active stress management techniques and academic advising.")

        # Counterfactual simulations (What-If analysis)
        simulations = []

        # Simulation 1: Increase study hours by 5
        sim_input_1 = student_input.copy()
        sim_input_1["hours_studied"] = min(sim_input_1["hours_studied"] + 6.0, 35.0)
        sim_df_1 = engineer_features(pd.DataFrame([sim_input_1]))
        sim_score_1 = float(np.clip(self.model.predict(self.preprocessor.transform(sim_df_1))[0], 0.0, 100.0))
        delta_1 = round(sim_score_1 - predicted_score, 1)
        if delta_1 > 0:
            simulations.append({
                "action": f"Increase weekly study by +6 hours (to {sim_input_1['hours_studied']:.1f}h)",
                "projected_score": round(sim_score_1, 1),
                "uplift": f"+{delta_1} pts"
            })

        # Simulation 2: Improve attendance to 95%
        if att < 90.0:
            sim_input_2 = student_input.copy()
            sim_input_2["attendance_rate"] = 95.0
            sim_df_2 = engineer_features(pd.DataFrame([sim_input_2]))
            sim_score_2 = float(np.clip(self.model.predict(self.preprocessor.transform(sim_df_2))[0], 0.0, 100.0))
            delta_2 = round(sim_score_2 - predicted_score, 1)
            if delta_2 > 0:
                simulations.append({
                    "action": "Increase class attendance to 95%",
                    "projected_score": round(sim_score_2, 1),
                    "uplift": f"+{delta_2} pts"
                })

        # Simulation 3: Add tutoring sessions
        if student_input.get("tutoring_sessions", 0) < 3:
            sim_input_3 = student_input.copy()
            sim_input_3["tutoring_sessions"] = sim_input_3.get("tutoring_sessions", 0) + 2
            sim_df_3 = engineer_features(pd.DataFrame([sim_input_3]))
            sim_score_3 = float(np.clip(self.model.predict(self.preprocessor.transform(sim_df_3))[0], 0.0, 100.0))
            delta_3 = round(sim_score_3 - predicted_score, 1)
            if delta_3 > 0:
                simulations.append({
                    "action": "Attend 2 additional tutoring sessions/month",
                    "projected_score": round(sim_score_3, 1),
                    "uplift": f"+{delta_3} pts"
                })

        return {
            "predicted_score": round(predicted_score, 1),
            "tier": tier_info["tier"],
            "badge": tier_info["badge"],
            "risk_level": tier_info["risk_level"],
            "strengths": strengths,
            "risk_factors": weaknesses,
            "recommendations": recommendations,
            "simulated_uplifts": simulations
        }

if __name__ == "__main__":
    predictor = StudentPerformancePredictor()
    
    sample_student = {
        "hours_studied": 10.5,
        "attendance_rate": 72.0,
        "previous_score": 68.0,
        "sleep_hours": 5.5,
        "tutoring_sessions": 1,
        "parental_education": "Some College",
        "internet_access": "Yes",
        "extracurricular_activities": "Yes",
        "peer_study_group": "No",
        "stress_level": 8
    }

    result = predictor.predict_single(sample_student)
    print("\n--- Student Academic Performance Prediction ---")
    print(f"Predicted Final Exam Score: {result['predicted_score']}/100")
    print(f"Academic Tier: {result['badge']} ({result['tier']})")
    print(f"Risk Level: {result['risk_level']}")
    print("\nRisk Factors:")
    for rf in result["risk_factors"]:
        print(f" - {rf}")
    print("\nActionable Uplift Simulations:")
    for sim in result["simulated_uplifts"]:
        print(f" -> {sim['action']}: {sim['projected_score']} ({sim['uplift']})")
