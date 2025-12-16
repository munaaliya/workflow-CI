import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# mlflow.set_tracking_uri() DIHAPUS
# karena GitHub Actions tidak memiliki MLflow server lokal

# mlflow.autolog() DIHAPUS
# karena kriteria Advanced mewajibkan manual logging

df train_model(data_path: str):
    df = pd.read_csv(data_path)
    
    X = df.drop("SmartHomeEfficiency", axis=1)
    y = df["SmartHomeEfficiency"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    
        # Manual logging metric (wajib Advanced)
        mlflow.log_metric("accuracy", acc)
    
        # Log model sebagai artifact "model/"
        # agar bisa digunakan oleh MLflow Project & Docker
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )
        print("Training selesai. Accuracy:", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str,required=True,help="Path ke dataset preprocessing")
    args = parser.parse_args()
    train_model(args.data_path)


