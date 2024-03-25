import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle


def random_forest_model(df: pd.DataFrame, input_columns: list, predict_column: str, ticker_name: str):
    X = df[input_columns]
    y = df[predict_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model = RandomForestRegressor(
        n_estimators=130,
        criterion="squared_error",
        max_depth=100,
        random_state=10,
        n_jobs=10,
        verbose=1,
    )
    model.fit(X_train, y_train)

    return model
    # with open(f"source_files/{ticker_name}_random_forest.pkl", "wb") as f:
    #     pickle.dump(model, f)
