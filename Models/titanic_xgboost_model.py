import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def load_and_preprocess(path):
    df = pd.read_csv(path)
    # avoid chained assignment warnings
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

def train_xgb(df, features, target_col='Survived'):
    X = df[features]
    y = df[target_col]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    print(f'Validation Accuracy: {accuracy_score(y_valid, y_pred):.4f}')
    print(classification_report(y_valid, y_pred))
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f'5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    return model

if __name__ == '__main__':
    df = load_and_preprocess('train.csv')
    FEATURES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    train_xgb(df, FEATURES)
