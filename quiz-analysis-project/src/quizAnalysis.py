import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class QuizAnalyzer:
    def __init__(self):
        self.current_quiz = None
        self.historical_quiz = None
        self.performance_summary = None
        self.student_personas = {}
        
    def fetch_data(self):
        """Fetches and processes quiz data from API endpoints."""
        try:
            # self.current_quiz = requests.get("https://jsonkeeper.com/b/LLQT", verify=False).json()
            # self.historical_quiz = requests.get("https://api.jsonserve.com/XgAgFJ", verify=False).json()
            self.current_quiz = pd.read_json("Current Quiz.json")
            self.historical_quiz = pd.read_json("API Endpoint.json")

            # Convert to DataFrames
            self.current_quiz = pd.DataFrame(self.current_quiz)
            self.historical_quiz = pd.DataFrame(self.historical_quiz)
            
            # Process nested JSON in historical quiz data
            self.historical_quiz['topic'] = self.historical_quiz['quiz'].apply(
                lambda x: x.get('title') if isinstance(x, dict) else None
            )
            self.historical_quiz['difficulty'] = self.historical_quiz['quiz'].apply(
                lambda x: x.get('difficulty') if isinstance(x, dict) else None
            )
            
            # Create performance metrics
            self.historical_quiz['correct'] = self.historical_quiz['score'] > 0
            self.historical_quiz['response_time'] = self.historical_quiz['quiz'].apply(
                lambda x: x.get('time_taken', 0) if isinstance(x, dict) else 0
            )
            
            # Fill NaN values
            self.historical_quiz['topic'] = self.historical_quiz['topic'].fillna('Unknown')
            self.historical_quiz['difficulty'] = self.historical_quiz['difficulty'].fillna('medium')
            self.historical_quiz['correct'] = self.historical_quiz['correct'].fillna(False)
            self.historical_quiz['response_time'] = self.historical_quiz['response_time'].fillna(0)
            
            return True
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False

    def analyze_performance(self):
        """Comprehensive performance analysis across multiple dimensions."""
        try:
            # Topic-wise performance
            topic_performance = self.historical_quiz.groupby(['user_id', 'topic']).agg({
                'correct': ['mean', 'count'],
                'response_time': 'mean'
            }).round(2)
            
            # Difficulty-wise performance
            difficulty_performance = self.historical_quiz.groupby(['user_id', 'difficulty']).agg({
                'correct': ['mean', 'count'],
                'response_time': 'mean'
            }).round(2)
            
            # Time-based improvement trends
            self.historical_quiz['quiz_number'] = self.historical_quiz.groupby('user_id').cumcount() + 1
            improvement_trends = self.historical_quiz.groupby(['user_id', 'quiz_number'])['correct'].mean()
            
            self.performance_summary = {
                'topic': topic_performance,
                'difficulty': difficulty_performance,
                'trends': improvement_trends
            }
            
            return self.performance_summary
        except Exception as e:
            print(f"Error in performance analysis: {str(e)}")
            return None

    def _generate_study_recommendations(self, characteristics):
        """Generates personalized study recommendations based on student characteristics."""
        try:
            recommendations = []
            
            # Base recommendations on performance level
            if characteristics['accuracy_level'] == 'Low':
                recommendations.extend([
                    "Focus on understanding fundamental concepts before advancing",
                    "Start with basic questions and gradually increase difficulty",
                    "Create concept maps for better understanding",
                    "Spend extra time reviewing incorrect answers",
                    "Consider joining study groups for peer support"
                ])
            elif characteristics['accuracy_level'] == 'Medium':
                recommendations.extend([
                    "Practice mixed difficulty questions",
                    "Focus on topics where accuracy is between 40-70%",
                    "Time yourself during practice sessions",
                    "Review and understand patterns in incorrect answers",
                    "Create summary notes for quick revision"
                ])
            else:  # High
                recommendations.extend([
                    "Challenge yourself with advanced problems",
                    "Focus on speed without compromising accuracy",
                    "Help explain concepts to others",
                    "Practice with time constraints",
                    "Work on connecting concepts across topics"
                ])
            
            # Speed-based recommendations
            if characteristics['speed'] == 'Careful':
                recommendations.extend([
                    "Practice timed mock tests regularly",
                    "Learn quick problem-solving techniques",
                    "Focus on identifying question patterns",
                    "Work on mental calculations"
                ])
            elif characteristics['speed'] == 'Fast':
                recommendations.extend([
                    "Double-check answers before submission",
                    "Practice careful reading of questions",
                    "Focus on accuracy over speed",
                    "Review careless mistakes patterns"
                ])
            
            # Consistency-based recommendations
            if characteristics['consistency'] == 'Variable':
                recommendations.extend([
                    "Maintain a regular study schedule",
                    "Track performance patterns by topic",
                    "Focus on weaker areas consistently",
                    "Review topics periodically"
                ])
            
            return list(set(recommendations))  # Remove any duplicates
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return ["Error generating personalized recommendations"]

    def define_student_persona(self, user_id):
        """Analyzes student behavior patterns to define their learning persona."""
        try:
            user_data = self.historical_quiz[self.historical_quiz['user_id'] == user_id]
            
            if user_data.empty:
                return {
                    'learning_style': 'Insufficient Data',
                    'performance_level': 'Unknown',
                    'strengths': [],
                    'improvement_areas': [],
                    'study_recommendations': ['Gather more quiz data to generate personalized recommendations']
                }
            
            # Calculate key metrics with error handling
            accuracy = user_data['correct'].mean() if not user_data['correct'].empty else 0
            avg_response_time = user_data['response_time'].mean() if not user_data['response_time'].empty else 0
            topic_consistency = user_data.groupby('topic')['correct'].std().mean() if not user_data.empty else 0
            
            # Define persona characteristics
            characteristics = {
                'accuracy_level': 'High' if accuracy > 0.7 else 'Medium' if accuracy > 0.4 else 'Low',
                'speed': 'Fast' if avg_response_time < 60 else 'Moderate' if avg_response_time < 120 else 'Careful',
                'consistency': 'Consistent' if topic_consistency < 0.2 else 'Variable',
            }
            
            # Determine primary strengths and areas for improvement
            topic_performance = user_data.groupby('topic')['correct'].mean()
            weak_topics = topic_performance[topic_performance < 0.6].index.tolist()
            strong_topics = topic_performance[topic_performance >= 0.7].index.tolist()
            
            # Generate recommendations
            recommendations = self._generate_study_recommendations(characteristics)
            
            persona = {
                'learning_style': f"{characteristics['speed']} {characteristics['consistency']} Learner",
                'performance_level': characteristics['accuracy_level'],
                'strengths': strong_topics,
                'improvement_areas': weak_topics if weak_topics else ['No specific weak areas identified'],
                'study_recommendations': recommendations
            }
            
            self.student_personas[user_id] = persona
            return persona
            
        except Exception as e:
            print(f"Error defining student persona: {str(e)}")
            return {
                'learning_style': 'Error',
                'performance_level': 'Unknown',
                'strengths': [],
                'improvement_areas': [],
                'study_recommendations': ['Error generating recommendations']
            }

    def predict_neet_rank(self, user_id):
        """Predicts NEET rank range based on quiz performance patterns."""
        try:
            user_data = self.historical_quiz[self.historical_quiz['user_id'] == user_id]
            
            if user_data.empty:
                return {
                    'predicted_range': 'Insufficient data',
                    'confidence_score': 0,
                    'key_factors': {
                        'accuracy': 0,
                        'consistency': 0,
                        'difficulty_handling': 0
                    }
                }
            
            # Calculate key performance indicators with error handling
            accuracy = user_data['correct'].mean() if not user_data['correct'].empty else 0
            consistency = user_data.groupby('topic')['correct'].std().mean() if not user_data.empty else 0
            difficulty_handling = (
                user_data[user_data['difficulty'] == 'hard']['correct'].mean()
                if not user_data[user_data['difficulty'] == 'hard'].empty
                else 0
            )
            
            # Handle NaN values
            accuracy = 0 if np.isnan(accuracy) else accuracy
            consistency = 0 if np.isnan(consistency) else consistency
            difficulty_handling = 0 if np.isnan(difficulty_handling) else difficulty_handling
            
            # Calculate rank score with safe values
            rank_score = (accuracy * 0.5 + (1 - consistency) * 0.3 + difficulty_handling * 0.2)
            rank_score = max(0, min(1, rank_score))  # Ensure score is between 0 and 1
            
            rank_ranges = {
                0: "90-95 percentile",
                1: "80-90 percentile",
                2: "70-80 percentile",
                3: "Below 70 percentile"
            }
            
            rank_index = min(3, int((1 - rank_score) * 4))
            
            prediction = {
                'predicted_range': rank_ranges[rank_index],
                'confidence_score': round(rank_score * 100, 2),
                'key_factors': {
                    'accuracy': round(accuracy * 100, 2),
                    'consistency': round((1 - consistency) * 100, 2),
                    'difficulty_handling': round(difficulty_handling * 100, 2)
                }
            }
            
            return prediction
            
        except Exception as e:
            print(f"Error in rank prediction: {str(e)}")
            return {
                'predicted_range': 'Error in prediction',
                'confidence_score': 0,
                'key_factors': {
                    'accuracy': 0,
                    'consistency': 0,
                    'difficulty_handling': 0
                }
            }
    def save_model(self, model, filename):
        """Saves the given model to a file."""
        try:
            joblib.dump(model, filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, filename):
        """Loads a model from a file."""
        try:
            model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

def main():
    # Initialize analyzer
    analyzer = QuizAnalyzer()
    
    # Fetch and process data
    if not analyzer.fetch_data():
        return
    
    try:
        # Perform analysis for a sample user
        user_id = analyzer.historical_quiz['user_id'].iloc[0]
        
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
        
        # Save the model (example with LogisticRegression)
        model = LogisticRegression()
        analyzer.save_model(model, "quiz_model.pkl")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()