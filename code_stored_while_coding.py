def finding_parameters(trial):
    hidden_layer1 = trial.suggest_int("hidden_layer1", 10, 100)
    hidden_layer2 = trial.suggest_int("hidden_layer2", 10, 100)
    hiden_layer3 = trial.suggest_int("hidden_layer3", 10, 100)
    hidden_layer4 = trial.suggest_int("hidden_layer4", 10, 100)
    max_iter = trial.suggest_int("max_iter", 10, 200)
    random_state = trial.suggest_int("random_state", 10, 100)

    model = MLPRegressor(hidden_layer_sizes=(hidden_layer1, hidden_layer2, hiden_layer3, hidden_layer4), max_iter=max_iter, random_state=random_state)
    model.fit(X_train, Y_train)
    v = model.predict(X_test)
    r2 = r2_score(Y_test, v)
    return r2


print ("finding parameters...please wait")
study = optu.create_study(direction="maximize")
study.optimize(finding_parameters, n_trials=2)

print("Best parameters found:", study.best_params)