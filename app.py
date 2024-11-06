from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import pipeline  # Import Hugging Face pipeline

app = Flask(__name__)
CORS(app)
  # Enable CORS for specific route

# Global variables for data, model, and fitness tips generator
data = None
model = None
fitness_tips_generator = pipeline('text-generation', model='gpt2', max_length=100)

# Load and clean the dataset with detailed diagnostics
def load_and_clean_data():
    global data, model
    try:
        # Load the data
        data = pd.read_csv('calories.csv')

        # Clean the columns
        data.columns = ['FoodCategory', 'FoodItem', 'per100grams', 'Cals_per100grams', 'KJ_per100grams']
        data['per100grams'] = data['per100grams'].replace({'g': '', 'ml': ''}, regex=True).astype(float)
        data['Cals_per100grams'] = data['Cals_per100grams'].replace({' cal': ''}, regex=True).astype(float)
        data['KJ_per100grams'] = data['KJ_per100grams'].replace({' kJ': ''}, regex=True).astype(float)

        # Drop rows with NaN values in essential columns
        data.dropna(subset=['per100grams', 'Cals_per100grams', 'KJ_per100grams'], inplace=True)

        # Check if data is empty after cleaning
        if data.empty:
            raise ValueError("No valid data available after cleaning. Please inspect your dataset.")

        # Prepare features (X) and target (y)
        X = data[['per100grams', 'KJ_per100grams']]
        y = data['Cals_per100grams']

        # Train a linear regression model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)

        # Print model metrics
        mse = mean_squared_error(y_test, model.predict(X_test))
        print(f"Mean Squared Error: {mse}")

    except Exception as e:
        print("An error occurred during data loading and cleaning:", str(e))

# Load and prepare data on startup
load_and_clean_data()

@app.route('/get_calories', methods=['POST'])
def get_calories():
    try:
        content = request.json
        print("Received JSON data:", content)  # Debug print statement

        # Validate input structure
        if not isinstance(content, dict):
            return jsonify({'error': 'Invalid input format. Expected a JSON object.'}), 400

        # Retrieve and normalize input values
        food_item = content.get('food_item', '').strip().lower()
        food_category = content.get('food_category', '').strip().lower()

        # Check if at least one of the required fields is provided
        if not food_item and not food_category:
            return jsonify({'error': 'Please provide either "food_item" or "food_category".'}), 400

        # Filter data based on user input
        filtered_data = data
        if food_item:
            filtered_data = filtered_data[filtered_data['FoodItem'].str.lower() == food_item]
        if food_category:
            filtered_data = filtered_data[filtered_data['FoodCategory'].str.contains(food_category, case=False, na=False)]

        # Check if the filtered data is empty
        if filtered_data.empty:
            return jsonify({'error': 'No matching data found for the provided food item or category.'}), 404

        # Prediction logic for the first matched item
        row = filtered_data.iloc[0]
        per_100g = row['per100grams']
        kj_per_100g = row['KJ_per100grams']

        if pd.notnull(per_100g) and pd.notnull(kj_per_100g):
            # Ensure prediction data format matches model
            input_data = pd.DataFrame([[per_100g, kj_per_100g]], columns=['per100grams', 'KJ_per100grams'])
            print("Input data for prediction:", input_data)  # Debug print statement
            predicted_calories = model.predict(input_data)[0]

            # Generate fitness tips based on predicted calories
            fitness_tips = generate_fitness_tips(predicted_calories, row['FoodCategory'])

            # Return result with fitness tips
            result = {
                'food_item': row['FoodItem'],
                'food_category': row['FoodCategory'],
                'predicted_calories': round(predicted_calories, 2),
                'fitness_tips': fitness_tips
            }
            return jsonify(result)

        # If prediction data is not valid
        return jsonify({'error': 'Unable to make a prediction due to missing values in the data.'}), 500

    except Exception as e:
        print("Error processing request:", str(e))
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

def generate_fitness_tips(predicted_calories, food_category):
    # Create a prompt to generate fitness tips based on calorie prediction and food category
    prompt = (f"Provide fitness tips based on a food item in the '{food_category}' category "
              f"with approximately {round(predicted_calories)} calories per 100 grams. "
              "Include advice on physical activities, portion control, and dietary adjustments.")
    
    # Generate tips using the Hugging Face model
    tips = fitness_tips_generator(prompt, max_length=100, num_return_sequences=1)
    return tips[0]['generated_text'].strip()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)
