

def finding_parameters(trial):
    hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 10, 20)
    max_iter = trial.suggest_int("max_iter", 10, 20)
    random_state = trial.suggest_int("random_state", 10, 20)

    model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter, random_state=random_state)
    model.fit(X_train, Y_train)
    v = model.predict(X_test)
    r2 = r2_score(Y_test, v)
    return r2


print ("finding parameters...")
study = optu.create_study(direction="maximize")
study.optimize(finding_parameters, n_trials=30)

print("Best parameters found:", study.best_params)