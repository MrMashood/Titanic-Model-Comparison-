import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import optuna

from tabm_reference import Model, make_parameter_groups  # Make sure this is available in your environment

# 1. Load and preprocess data
def load_and_preprocess(path, scaler=None):
    df = pd.read_csv(path)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Numeric features
    num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    X_num = df[num_cols].values.astype(float)
    if scaler is None:
        scaler = StandardScaler().fit(X_num)
    X_num = scaler.transform(X_num)

    # Categorical features
    df['Sex'] = df['Sex'].astype('category').cat.codes
    df['Embarked'] = df['Embarked'].astype('category').cat.codes
    cat_cols = ['Sex', 'Embarked']
    X_cat = df[cat_cols].values.astype(int)

    # Target
    y = df['Survived'].values.astype(int)
    return X_num, X_cat, y, scaler


# 2. Train one fold or a holdout split
def train_one_fold(params, X_num, X_cat, y, fold_idx=None):
    if fold_idx is None:
        Xn_tr, Xn_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
            X_num, X_cat, y, test_size=0.2, random_state=42
        )
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, (tr, val) in enumerate(kf.split(X_num)):
            if i == fold_idx:
                Xn_tr, Xn_val = X_num[tr], X_num[val]
                Xc_tr, Xc_val = X_cat[tr], X_cat[val]
                y_tr, y_val = y[tr], y[val]
                break

    # Dataloaders
    train_ds = TensorDataset(
        torch.tensor(Xn_tr, dtype=torch.float32),
        torch.tensor(Xc_tr, dtype=torch.long),
        torch.tensor(y_tr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(Xn_val, dtype=torch.float32),
        torch.tensor(Xc_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'])

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    cat_cards = [int(np.unique(Xc_tr[:, i]).size) for i in range(Xc_tr.shape[1])]
    model = Model(
        n_num_features=Xn_tr.shape[1],
        cat_cardinalities=cat_cards,
        n_classes=2,
        backbone={
            'type': 'MLP',
            'n_blocks': params['n_blocks'],
            'd_block': params['d_block'],
            'dropout': params['dropout'],
        },
        bins=None,
        num_embeddings=None,
        arch_type='tabm',
        k=params['k'],
        share_training_batches=True,
    ).to(device)

    # Optimizer & Scheduler
    optim = torch.optim.AdamW(
        make_parameter_groups(model),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='max', factor=0.5, patience=5
    )

    # Training loop
    best_acc = 0.0
    best_state = None
    no_imp = 0
    for epoch in range(1, params['epochs'] + 1):
        model.train()
        for xn, xc, yb in train_loader:
            xn, xc, yb = xn.to(device), xc.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xn, xc).mean(dim=1)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()

        # Validation
        model.eval()
        preds = []
        with torch.no_grad():
            for xn, xc, _ in val_loader:
                xn, xc = xn.to(device), xc.to(device)
                logits = model(xn, xc).mean(dim=1)
                preds.append(torch.argmax(logits, dim=1).cpu())
        y_pred = torch.cat(preds).numpy()
        val_acc = accuracy_score(y_val, y_pred)

        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= params['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:2d} | Val Acc: {val_acc:.4f} | Best Acc: {best_acc:.4f}")

    model.load_state_dict(best_state)
    return best_acc


# 3. Optuna objective
def objective(trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'k': trial.suggest_categorical('k', [8, 16, 32]),
        'n_blocks': trial.suggest_int('n_blocks', 2, 4),
        'd_block': trial.suggest_categorical('d_block', [128, 256, 512]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'epochs': 30,
        'patience': 7,
    }
    X_num, X_cat, y, _ = load_and_preprocess('train.csv')
    return train_one_fold(params, X_num, X_cat, y)


# 4. Entry point
if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print("Best params:", study.best_trial.params)
    print("Best validation accuracy:", study.best_value)
