# Quiz Analysis Project

This project is designed to analyze quiz data and provide insights into student performance. It includes functionalities for fetching quiz data, analyzing performance metrics, defining student personas, and predicting NEET ranks.

## Project Structure

```
quiz-analysis-project
├── src
│   ├── quizAnalysis.py       # Main logic for analyzing quiz data
│   ├── test_export_model.py  # Test cases for the exported model  
├── requirements.txt          # Lists project dependencies
└── README.md                 # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd quiz-analysis-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Running the Analysis**:
   To run the quiz analysis, execute the `quizAnalysis.py` file:
   ```
   python src/quizAnalysis.py
   ```

2. **Exporting the Model**:
   The `quizAnalysis.py` file allows you to export the `QuizAnalyzer` class for use in other modules. You can import it as follows:
   ```python
   from quizAnalysis import QuizAnalyzer
   ```

3. **Testing**:
   The project includes test cases located in the `src` directory. You can run the tests using:
   ```
   pytest src/test_export_model.py
   ```

## Features

- **Data Fetching**: Automatically fetches quiz data from specified API endpoints.
- **Performance Analysis**: Provides comprehensive analysis of student performance across various dimensions.
- **Student Persona Definition**: Defines student personas based on quiz performance metrics.
- **NEET Rank Prediction**: Predicts NEET rank ranges based on quiz performance patterns.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.