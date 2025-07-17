```python
# Flight Fare Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Loading and Preprocessing
def load_and_clean_data(csv_files):
    """Load and combine data from multiple CSV files"""
    all_dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['route_file'] = file  # Keep track of source file
        all_dataframes.append(df)
        print(f"Loaded {file}: {df.shape[0]} records")
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Clean Price column
    combined_df['Price_SAR'] = combined_df['Price'].apply(lambda x: float(str(x).replace('\xa0SAR', '').replace(',', '')))
    
    # Convert Duration to minutes
    def duration_to_minutes(duration_str):
        hours, minutes = 0, 0
        parts = str(duration_str).strip().split('h')
        if len(parts) > 1:
            hours = int(parts[0].strip())
            minutes_part = parts[1].strip().replace('m', '')
            if minutes_part:
                minutes = int(minutes_part)
        return hours * 60 + minutes
    
    combined_df['Duration_minutes'] = combined_df['Duration'].apply(duration_to_minutes)
    
    # Clean stops
    combined_df['Stops'] = combined_df['Total stops'].apply(lambda x: 0 if 'nonstop' in str(x).lower() 
                                                 else int(str(x).split()[0]) if str(x).split()[0].isdigit() else 0)
    
    # Convert date and extract features
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='mixed')
    combined_df['Day_of_week'] = combined_df['Date'].dt.dayofweek
    combined_df['Day_of_month'] = combined_df['Date'].dt.day
    combined_df['Month'] = combined_df['Date'].dt.month
    
    # Clean airline names (take first airline for multi-airline routes)
    combined_df['Primary_Airline'] = combined_df['Airline'].apply(lambda x: str(x).split(',')[0].strip())
    
    return combined_df

# 2. Feature Engineering
def apply_feature_engineering(df):
    """Apply mean encoding and create polynomial features"""
    # Mean encoding for categorical features
    df['Airline_Mean_Price'] = df.groupby('Primary_Airline')['Price_SAR'].transform('mean')
    df['Source_Mean_Price'] = df.groupby('Source')['Price_SAR'].transform('mean')
    df['Destination_Mean_Price'] = df.groupby('Destination')['Price_SAR'].transform('mean')
    df['Route_Mean_Price'] = df.groupby(['Source', 'Destination'])['Price_SAR'].transform('mean')
    
    # Create additional features
    df['Duration_hours'] = df['Duration_minutes'] / 60
    df['Price_per_minute'] = df['Price_SAR'] / df['Duration_minutes']
    
    # Select numerical features for polynomial transformation
    numerical_features = ['Duration_minutes', 'Stops', 'Day_of_week', 'Day_of_month', 'Month']
    X_numerical = df[numerical_features]
    
    # Create polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_numerical)
    feature_names = poly.get_feature_names_out(numerical_features)
    
    # Create DataFrame with polynomial features
    poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
    
    # Add polynomial features to main dataframe
    df = pd.concat([df, poly_df], axis=1)
    
    return df

# 3. Visualizations
def create_visualizations(df):
    """Create various visualizations for analysis"""
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # 1. Price Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Price_SAR'], bins=50, kde=True)
    plt.title('Flight Price Distribution', fontsize=15)
    plt.xlabel('Price (SAR)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(df['Price_SAR'].mean(), color='red', linestyle='--', label=f'Mean: {df["Price_SAR"].mean():.0f} SAR')
    plt.axvline(df['Price_SAR'].median(), color='green', linestyle='--', label=f'Median: {df["Price_SAR"].median():.0f} SAR')
    plt.legend()
    plt.tight_layout()
    plt.savefig('price_distribution.png', dpi=300)
    plt.close()
    
    # 2. Average Price by Airline (Top 10)
    airline_stats = df.groupby('Primary_Airline').agg({
        'Price_SAR': 'mean',
        'Primary_Airline': 'count'
    }).rename(columns={'Primary_Airline': 'count'})
    
    # Filter airlines with at least 100 flights and get top 10
    airline_stats = airline_stats[airline_stats['count'] >= 100].sort_values('Price_SAR', ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=airline_stats['Price_SAR'], y=airline_stats.index)
    plt.title('Average Flight Prices by Airline (Top 10)', fontsize=15)
    plt.xlabel('Average Price (SAR)', fontsize=12)
    plt.ylabel('Airline', fontsize=12)
    plt.tight_layout()
    plt.savefig('airline_prices.png', dpi=300)
    plt.close()
    
    # 3. Duration vs Price Scatter Plot
    plt.figure(figsize=(12, 8))
    sample_data = df.sample(n=min(2000, len(df)))  # Sample for better visualization
    scatter = sns.scatterplot(
        x='Duration_hours', 
        y='Price_SAR', 
        hue='Stops',
        palette='viridis',
        alpha=0.7,
        data=sample_data
    )
    plt.title('Relationship Between Flight Duration and Price', fontsize=15)
    plt.xlabel('Duration (Hours)', fontsize=12)
    plt.ylabel('Price (SAR)', fontsize=12)
    plt.tight_layout()
    plt.savefig('duration_vs_price.png', dpi=300)
    plt.close()
    
    # 4. Box Plot - Price by Route
    plt.figure(figsize=(15, 10))
    route_data = df.groupby(['Source', 'Destination'])['Price_SAR'].mean().reset_index()
    route_data['Route'] = route_data['Source'] + ' → ' + route_data['Destination']
    route_data = route_data.sort_values('Price_SAR', ascending=False)
    
    # Create route identifier
    df['Route'] = df['Source'] + ' → ' + df['Destination']
    
    # Create the boxplot
    sns.boxplot(x='Route', y='Price_SAR', data=df, order=route_data['Route'])
    plt.title('Price Distribution by Route', fontsize=15)
    plt.xlabel('Route', fontsize=12)
    plt.ylabel('Price (SAR)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_by_route.png', dpi=300)
    plt.close()
    
    # 5. Violin Plot - Airline vs Price for top 6 airlines
    top_airlines = df['Primary_Airline'].value_counts().head(6).index.tolist()
    plt.figure(figsize=(14, 8))
    violin_data = df[df['Primary_Airline'].isin(top_airlines)]
    sns.violinplot(x='Primary_Airline', y='Price_SAR', data=violin_data)
    plt.title('Price Distribution by Top Airlines', fontsize=15)
    plt.xlabel('Airline', fontsize=12)
    plt.ylabel('Price (SAR)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_by_airline_violin.png', dpi=300)
    plt.close()
    
    print("All visualizations have been created successfully.")

# 4. Simple Model Building with Feature Engineering
def build_prediction_model(df):
    """Build a simple price prediction model using engineered features"""
    # Prepare features
    features = [
        'Duration_minutes', 'Stops', 'Day_of_week', 'Month',
        'Airline_Mean_Price', 'Source_Mean_Price', 'Destination_Mean_Price', 'Route_Mean_Price',
        'Duration_minutes^2', 'Duration_minutes Stops', 'Stops^2', 'Day_of_week Month'
    ]
    
    # Filter out features that might not exist in the dataset
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df['Price_SAR']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print(f"Model Performance:")
    print(f"Training RMSE: {train_rmse:.2f} SAR")
    print(f"Test RMSE: {test_rmse:.2f} SAR")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Feature Importance for Price Prediction', fontsize=15)
    plt.xlabel('Importance (Absolute Coefficient Value)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    
    return model, feature_importance

# Main execution
def main():
    # 1. Specify the CSV files to load
    csv_files = [f for f in os.listdir() if f.endswith('.csv') and f != 'flight_data_processed.csv']
    
    # 2. Load and clean the data
    print("Loading and cleaning data...")
    df = load_and_clean_data(csv_files)
    
    # 3. Apply feature engineering
    print("\nApplying feature engineering...")
    df_engineered = apply_feature_engineering(df)
    
    # 4. Save processed data
    df_engineered.to_csv('flight_data_processed.csv', index=False)
    print("Processed data saved as 'flight_data_processed.csv'")
    
    # 5. Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df_engineered)
    
    # 6. Build prediction model
    print("\nBuilding prediction model...")
    model, feature_importance = build_prediction_model(df_engineered)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    import os
    main()
```