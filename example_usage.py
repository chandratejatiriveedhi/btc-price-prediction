from main import BTCPricePredictor
from sklearn.model_selection import train_test_split

def main():
    # Initialize predictor
    predictor = BTCPricePredictor(start_date='2020-01-01')
    
    # Fetch and prepare data
    data = predictor.fetch_data()
    X, y = predictor.prepare_sequences()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    history = predictor.train_model(X_train, y_train)
    
    # Make predictions
    predictions = predictor.make_prediction(X_test)
    y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate model
    metrics = predictor.evaluate_model(y_test_actual, predictions)
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    predictor.plot_results(y_test_actual, predictions)

if __name__ == "__main__":
    main()