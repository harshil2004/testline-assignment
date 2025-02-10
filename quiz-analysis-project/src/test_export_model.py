import pandas as pd
from quizAnalysis import QuizAnalyzer
from sklearn.linear_model import LogisticRegression
import random

def main():
    # Initialize analyzer
    analyzer = QuizAnalyzer()
    
    # Fetch and process data
    if not analyzer.fetch_data():
        return
    
    try:
        # Select a random user ID
        user_id = random.choice(analyzer.historical_quiz['user_id'].unique())
        # user_id = pd.read_json('test_data.json')['user_id']
        print(f"Performing analysis for user: {user_id}")
        
        # Generate comprehensive analysis
        analyzer.analyze_performance()
        
        # Define student persona
        persona = analyzer.define_student_persona(user_id)
        print("\nStudent Persona:")
        print(f"Learning Style: {persona['learning_style']}")
        print(f"Performance Level: {persona['performance_level']}")
        print(f"Strengths: {', '.join(persona['strengths'])}")
        print(f"Areas for Improvement: {', '.join(persona['improvement_areas'])}")
        print("\nStudy Recommendations:")
        for i, rec in enumerate(persona['study_recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Generate rank prediction
        rank_prediction = analyzer.predict_neet_rank(user_id)
        print("\nNEET Rank Prediction:")
        print(f"Predicted Range: {rank_prediction['predicted_range']}")
        print(f"Confidence Score: {rank_prediction['confidence_score']}%")
        print("\nKey Performance Factors:")
        for factor, value in rank_prediction['key_factors'].items():
            print(f"{factor.replace('_', ' ').title()}: {value}%")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()