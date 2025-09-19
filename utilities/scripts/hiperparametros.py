from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
import optuna

# Muestra información en cada iteración de la búsqueda de hiperparámetros bayesiana realizada por Optuna.
# Se utiliza como callback al llamar al método optimize de optuna.
def champion_callback(study, frozen_trial):
    """
    Mostramos menos información, sino es demasiado verboso
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

# Función objetivo para SVR, necesaria para trabajar con optuna.
def objective_SVR(trial, X_train, y_train, cv, scoring = "neg_root_mean_squared_error"):
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-4, 1.0, log=True)
    C_ = trial.suggest_float("C", 1, 1000, log=True)
    
    match kernel:
        case "poly":
            degree = trial.suggest_int("degree", 2, 5)
            svr_model = SVR(kernel=kernel, degree=degree, gamma=gamma, epsilon=epsilon, C=C_)
        case "sigmoid" | "rbf":
            svr_model = SVR(kernel=kernel, gamma=gamma, epsilon=epsilon, C=C_)
        case _: # "linear"
            svr_model = SVR(kernel=kernel)
    
    score = cross_val_score(svr_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    return score.mean() * (-1)

# Función objetivo para Decision Tree, necesaria para trabajar con optuna.
def objective_DecisionTree(trial, random_state, X_train, y_train, cv, scoring = "neg_root_mean_squared_error"):
    max_depth = trial.suggest_categorical("max_depth", [None,10,20,30])
    min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10])
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 4])
    ccp_alpha = trial.suggest_categorical("ccp_alpha", [0.1, 0.01, 0.001, 0.0])

    dt_model = DecisionTreeRegressor(random_state=random_state,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     ccp_alpha=ccp_alpha
                                     )
    
    score = cross_val_score(dt_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    return score.mean() * (-1)

# Función objetivo para Random Forest, necesaria para trabajar con optuna.
def objective_RF(trial, random_state, X_train, y_train, cv, scoring = "neg_root_mean_squared_error"):
    n_estimators = trial.suggest_categorical("n_estimators", [100, 500, 1000, 1500])
    criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"])
    max_depth = trial.suggest_categorical("max_depth", [None,10,20,30])
    min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10])
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 4])
    max_features = trial.suggest_categorical("max_features", ["sqrt","log2", None])
    ccp_alpha = trial.suggest_categorical("ccp_alpha", [0.1, 0.01, 0.001, 0.0])

    rf_model = RandomForestRegressor(random_state=random_state, n_jobs=-1,
                          n_estimators=n_estimators,
                          criterion=criterion,
                          max_depth=max_depth,
                          min_samples_split=min_samples_split,
                          min_samples_leaf=min_samples_leaf,
                          max_features=max_features,
                          ccp_alpha=ccp_alpha
                          )
    
    score = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    return score.mean() * (-1)

# Función objetivo para HGB, necesaria para trabajar con optuna.
def objective_HGB(trial, random_state, X_train, y_train, cv, scoring = "neg_root_mean_squared_error"):
    loss = trial.suggest_categorical("loss", ["squared_error", "absolute_error", "gamma", "poisson", "quantile"])
    learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1])
    max_iter = trial.suggest_int("max_iter", 100, 1000, step=100)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 32, step=5)
    max_depth = trial.suggest_int("max_depth", 3, 10, step=1)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15, step=2)
    l2_regularization = trial.suggest_float("l2_regularization", 1e-4, 1e-2, log=True)
    
    match loss:
        case "quantile":
            quantile = trial.suggest_categorical("quantile", [0.001, 0.01, 0.1, 0.5, 0.9])
        case _:
            quantile = None
    
    hgb_model = HistGradientBoostingRegressor(random_state=random_state,
                                              loss=loss,
                                              learning_rate=learning_rate,
                                              max_iter=max_iter,
                                              max_leaf_nodes=max_leaf_nodes,
                                              max_depth=max_depth,
                                              min_samples_leaf=min_samples_leaf,
                                              l2_regularization=l2_regularization,
                                              quantile=quantile
                                            )
    
    score = cross_val_score(hgb_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    return score.mean() * (-1)

# Función objetivo para XGBoost, necesaria para trabajar con optuna.
def objective_XGBoost(trial, random_state, X_train, y_train, cv, scoring = "neg_root_mean_squared_error"):
    max_depth = trial.suggest_int("max_depth", 2, 10, step=1)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10, step=1)
    subsample = trial.suggest_float("subsample", 0.6, 1.0, step=0.05)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1)
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1])
    reg_alpha = trial.suggest_categorical("reg_alpha", [0, 0.1, 0.5, 1, 5, 10])
    reg_lambda = trial.suggest_categorical("reg_lambda", [0, 0.1, 0.5, 1, 5, 10])
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100)
    
    xgb_model = XGBRegressor(random_state=random_state,
                             max_depth=max_depth,
                             min_child_weight=min_child_weight,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             learning_rate=learning_rate,
                             reg_alpha=reg_alpha,
                             reg_lambda=reg_lambda,
                             n_estimators=n_estimators,
                             tree_method="hist",
                             n_jobs=-1)
    
    score = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    return score.mean() * (-1)
