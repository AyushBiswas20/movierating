# movierating

# Importing Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load Dataset with Encoding and Error Handling
dataset_path = '/kaggle/input/imdb-india-movies/IMDb Movies India.csv'  # Replace with actual path

try:
    df = pd.read_csv(dataset_path, encoding='latin1', on_bad_lines='skip')  # Handles encoding issues and bad lines
    print("Dataset loaded successfully!")
except UnicodeDecodeError as e:
    print(f"Encoding Error: {e}")
    exit()

# Step 2: Data Preprocessing
# Handle missing values
numerical_columns = df.select_dtypes(include=['float', 'int']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())  # Fill numerical columns with mean
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])  # Fill categorical columns with mode

# Encode categorical variables
for col in categorical_columns:
    try:
        df[col] = LabelEncoder().fit_transform(df[col])  # Label encoding for categorical columns
    except Exception as e:
        print(f"Error encoding column {col}: {e}")

# Step 3: Feature Engineering
# Adding Director's Success Rate
if 'Director' in df.columns and 'Rating' in df.columns:
    director_success_rate = df.groupby('Director')['Rating'].mean()
    df['Director_Success_Rate'] = df['Director'].map(director_success_rate)

# Adding Average Rating of Similar Movies (by Genre)
if 'Genre' in df.columns and 'Rating' in df.columns:
    genre_avg_rating = df.groupby('Genre')['Rating'].mean()
    df['Genre_Avg_Rating'] = df['Genre'].map(genre_avg_rating)

# Step 4: Define Inputs and Outputs
if 'Rating' in df.columns:
    X = df.drop(['Rating'], axis=1)  # Features
    y = df['Rating']  # Target variable

# Step 5: Split Data into Training and Testing Sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split successfully!")
except ValueError as ve:
    print(f"Data split error: {ve}")
    exit()

# Step 6: Build Predictive Model
try:
    model = RandomForestRegressor(random_state=42)  # Using Random Forest Regressor
    model.fit(X_train, y_train)
    print("Model trained successfully!")
except Exception as e:
    print(f"Model training error: {e}")
    exit()

# Step 7: Evaluate Model
try:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error metric
    print(f"Model RMSE: {rmse}")
except Exception as e:
    print(f"Model evaluation error: {e}")

# Step 8: Save Model for Future Use
try:
    joblib.dump(model, 'movie_rating_predictor.pkl')  # Save model as a .pkl file
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")
