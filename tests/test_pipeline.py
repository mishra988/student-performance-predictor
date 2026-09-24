import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure src is on python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
sys.path.insert(0, src_dir)

from data_generator import generate_student_dataset
from data_preprocessing import engineer_features, prepare_data
from predict import StudentPerformancePredictor

class TestAcademicPredictionPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_df = generate_student_dataset(n_samples=60, random_state=42)

    def test_data_generator_shape_and_columns(self):
        self.assertEqual(len(self.sample_df), 60)
        expected_cols = [
            "student_id", "hours_studied", "attendance_rate", "previous_score",
            "sleep_hours", "tutoring_sessions", "parental_education",
            "internet_access", "extracurricular_activities", "peer_study_group",
            "stress_level", "final_score", "performance_tier"
        ]
        for col in expected_cols:
            self.assertIn(col, self.sample_df.columns)
            self.assertFalse(self.sample_df[col].isnull().any())

    def test_feature_engineering(self):
        engineered = engineer_features(self.sample_df)
        new_features = ["study_efficiency", "attendance_risk", "academic_momentum", "wellbeing_ratio"]
        for f in new_features:
            self.assertIn(f, engineered.columns)
            self.assertFalse(np.isinf(engineered[f]).any())
            self.assertFalse(engineered[f].isnull().any())

    def test_preprocessing_and_split(self):
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(self.sample_df, test_size=0.25)
        self.assertEqual(X_train.shape[0], 45)
        self.assertEqual(X_test.shape[0], 15)
        self.assertEqual(len(y_train), 45)
        self.assertEqual(len(y_test), 15)
        self.assertGreater(len(feature_names), 10)

    def test_predictor_inference(self):
        predictor = StudentPerformancePredictor()
        student = {
            "hours_studied": 15.0,
            "attendance_rate": 88.0,
            "previous_score": 75.0,
            "sleep_hours": 7.5,
            "tutoring_sessions": 2,
            "parental_education": "Bachelor's",
            "internet_access": "Yes",
            "extracurricular_activities": "Yes",
            "peer_study_group": "Yes",
            "stress_level": 4
        }
        res = predictor.predict_single(student)
        self.assertIsInstance(res["predicted_score"], float)
        self.assertGreaterEqual(res["predicted_score"], 0.0)
        self.assertLessEqual(res["predicted_score"], 100.0)
        self.assertIn(res["tier"], ["Distinction", "Merit", "Pass", "At-Risk"])
        self.assertIn(res["risk_level"], ["Minimal", "Low", "Moderate", "High"])
        self.assertIsInstance(res["strengths"], list)
        self.assertIsInstance(res["simulated_uplifts"], list)

if __name__ == "__main__":
    unittest.main()
