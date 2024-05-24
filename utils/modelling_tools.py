import os
import copy
from typing import (
    Dict, 
    List,
    Any
)
import shutil

import mlflow

import numpy as np
import pandas as pd

from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import StratifiedShuffleSplit

import shap

import matplotlib.pyplot as plt


def generate_params_dict_from_distributions(distributions: Dict[str, Any]) -> Dict[str, Any]:
    params = dict.fromkeys(distributions.keys())
    for kye_param in distributions.keys():
        if (type(distributions[kye_param]) == list) | (type(distributions[kye_param]) == np.ndarray):
            i = np.random.randint(len(distributions[kye_param]))
            params[kye_param] = distributions[kye_param][i]
        else:
            params[kye_param] = distributions[kye_param].rvs()
    return params


def update_solver_by_penalty(params: Dict[str, Any]):
    solvers_penalty = {
        'l2': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        'l1': ['liblinear', 'saga'],
        'elasticnet': ['saga']
    }
    n_solvers = len(solvers_penalty[params['penalty']])
    ind = np.random.choice(np.arange(n_solvers), 1)[0]
    params['solver'] = solvers_penalty[params['penalty']][ind]

    if params['penalty'] == 'elasticnet':
        params['l1_ratio'] = uniform(loc=0, scale=1).rvs()
    return params


def search_hyper_params_and_log(run_name: str, 
                                data_dict: Dict[str, pd.DataFrame],
                                cols_to_use: List[str],
                                target_name: str,
                                distributions: Dict[str, Any],
                                n_iter: int,
                                model: Any,
                                palette: Dict[str, Any],
                                custom_params_updates=None, 
                                tags: Dict[str, str]=None, 
                                test_name: str='df_test'
                               ) -> pd.DataFrame:
    pr_auc: List[float] = []
    gini_vals: List[float] = []
    df_metrics: pd.DataFrame = pd.DataFrame()
    
    np.random.seed(distributions['random_state'][0])
    for i in range(n_iter):
        print(f"# {i + 1} / {n_iter}")
       
        # generate parameters
        params = generate_params_dict_from_distributions(distributions)
        if custom_params_updates is not None:
            params = custom_params_updates(params)

        X_train = data_dict['df_train'][cols_to_use]
        Y_train = data_dict['df_train'][target_name]
        X_val = data_dict['df_val'][cols_to_use]
        Y_val = data_dict['df_val'][target_name]

        std_scaler = StandardScaler()
        std_scaler.fit(X_train)
        X_train = std_scaler.transform(X_train)
        X_val = std_scaler.transform(X_val)

        model_obj = model(**params)
        model_obj.fit(X_train, Y_train)

        y_train_pred = model_obj.predict_proba(X_train)[:, 1]
        y_val_pred = model_obj.predict_proba(X_val)[:, 1]

        roc_auc_train = roc_auc_score(Y_train.astype(int), y_train_pred)
        gini_train = 2 * roc_auc_train - 1

        roc_auc_val = roc_auc_score(Y_val.astype(int), y_val_pred)
        gini_val = 2 * roc_auc_val - 1

        precision, recall, _ = precision_recall_curve(Y_train.astype(int), y_train_pred)
        auc_train = auc(x=recall, y=precision)
        
        precision, recall, _ = precision_recall_curve(Y_val.astype(int), y_val_pred)
        auc_values = auc(x=recall, y=precision)
        
        pr_auc += [auc_values]
        gini_vals += [gini_val]

        df_metrics = pd.concat(
            (
                df_metrics, 
                pd.DataFrame(
                    {
                        'iter': [i],
                        'GINI_train': [gini_train],
                        'GINI_val': [gini_val],
                        'GINI_prcnt_chng': [(gini_val - gini_train) * 100 / gini_train],
                        'PR_AUC_train': [auc_train],
                        'PR_AUC_val': [auc_values],
                        'PR_AUC_prcnt_chng': [(auc_values - auc_train) * 100 / auc_train],
                        'params': [params]
                    }
                )
            )
        )
        

        with mlflow.start_run(run_name=f'{i}', nested=True, log_system_metrics=True) as child:
            mlflow.log_params(params)
            mlflow.log_metric(f"Val-PR_AUC", auc_values)
            mlflow.log_metric(f"Val-GINI", gini_val)
            if os.path.exists(f'{run_name}-{i}'):
                shutil.rmtree(f'{run_name}-{i}')  
            mlflow.sklearn.save_model(model_obj, f'{run_name}-{i}')
            mlflow.sklearn.log_model(model_obj, f'{run_name}-{i}')
            if tags is not None:
                for tag_key, tag_value in tags.items():
                    mlflow.set_tag(tag_key, tag_value)
    df_metrics = df_metrics.reset_index(drop=True)
    best_iter = df_metrics.PR_AUC_val.argmax()
    best_params = df_metrics.loc[best_iter, 'params']
    best_gini = df_metrics.loc[best_iter, 'GINI_val'] 
    best_pr_auc = df_metrics.loc[best_iter, 'PR_AUC_val'] 
    
    print("Best params:\n", best_params)
    mlflow.log_metric(f"Val-PR_AUC", best_pr_auc)
    mlflow.log_metric(f"Val-GINI", best_gini)
    mlflow.log_params(best_params)

    model_obj = mlflow.sklearn.load_model(f'{run_name}-{best_iter}')
    X_train = data_dict['df_train'][cols_to_use]
    Y_train = data_dict['df_train'][target_name]
    X_val = data_dict['df_val'][cols_to_use]
    Y_val = data_dict['df_val'][target_name]

    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_val = std_scaler.transform(X_val)

    y_val_pred = model_obj.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(Y_val.astype(int), y_val_pred)
    f, ax_pr_rec = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    pr_val = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_val.plot(ax=ax_pr_rec, name='Validation', color=palette['DataPart']['Val'])
    _ = plt.title('Precision Recall Curve')

    X_train_val = data_dict['df_train_val'][cols_to_use]
    Y_train_val = data_dict['df_train_val'][target_name]
    X_test = data_dict[test_name][cols_to_use]
    Y_test = data_dict[test_name][target_name]

    std_scaler = StandardScaler()
    std_scaler.fit(X_train_val)
    X_train_val = std_scaler.transform(X_train_val)
    X_test = std_scaler.transform(X_test)

    model_obj = model(**best_params)
    model_obj.fit(X_train_val, Y_train_val)
    y_test_pred = model_obj.predict_proba(X_test)[:, 1]

    # MODELS_PATH

    roc_auc_val = roc_auc_score(Y_test, y_test_pred)
    gini_val = 2 * roc_auc_val - 1

    precision, recall, _ = precision_recall_curve(Y_test, y_test_pred)
    auc_val = auc(x=recall, y=precision)

    pr_val = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_val.plot(ax=ax_pr_rec, name='Test', color=palette['DataPart']['Test'])
    mlflow.log_figure(f, "PR-Val-Test.png")    

    print("Test-PR_AUC", auc_val)
    print("Test-Gini", gini_val)
    mlflow.log_metric(f"Test-PR_AUC", auc_val)
    mlflow.log_metric(f"Test-GINI", gini_val)

    mlflow.log_metric(
        f"val_test-prcnt_chng-PR_AUC", 
        (auc_val - best_pr_auc) * 100 / best_pr_auc
    )
    mlflow.log_metric(
        f"val_test-prcnt_chng-GINI", 
        (gini_val - best_gini) * 100 / best_gini
    )
    
    
    if tags is not None:
        for tag_key, tag_value in tags.items():
            mlflow.set_tag(tag_key, tag_value)
    try:
        mlflow.sklearn.save_model(model_obj, os.path.join(model_path, run_name))
        mlflow.sklearn.log_model(model_obj, os.path.join(model_path, run_name))
    except:
        print('Not saved')

    explainer_smpl= shap.Explainer(
        model_obj, data_dict[test_name + '_sample'][cols_to_use], columns=cols_to_use, 
        feature_dependence="interventional", model_output="predict_proba"
    )
    explainer_obj = explainer_smpl(data_dict[test_name + '_sample'][cols_to_use])
    n_features = len(cols_to_use)
    
    f, ax = plt.subplots(nrows=n_features, ncols=1, figsize=(20, 100))
    for i, col in enumerate(cols_to_use):
        print(explainer_obj[:, col])
        _ = shap.plots.scatter(explainer_obj[:, col], ax=ax[i], show=False)
    
    mlflow.log_figure(f, "SHAP_test_sample_per_obs.png")
    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    shap.summary_plot(
        explainer_obj[:, :], data_dict[test_name + '_sample'][cols_to_use], 
        plot_type="bar", feature_names=cols_to_use, plot_size=(5, 10), show=False
        # max_daisplay=len(cols_to_use)
    )
    plt.show()
    mlflow.log_figure(f, "SHAP_test_sample_features_importance.png") 

    for i in range(n_iter):
        if i != best_iter:
            shutil.rmtree(f'{run_name}-{i}')
    return df_metrics


def train_pred(data_dict: Dict[str, pd.DataFrame], 
              cols_to_use: List[str], 
              target_name: str, 
              model, 
              params,
              alias: Any,
              test_name: str='df_test') -> Any:
    X_train = data_dict['df_train_val'][cols_to_use]
    Y_train = data_dict['df_train_val'][target_name]

    X_test = data_dict[test_name][cols_to_use]
    Y_test = data_dict[test_name][target_name]

    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train = std_scaler.transform(X_train)
    X_test = std_scaler.transform(X_test)

    model_obj = model(**params)
    model_obj.fit(X_train, Y_train)

    y_train_pred = model_obj.predict_proba(X_train)[:, 1]
    y_test_pred = model_obj.predict_proba(X_test)[:, 1]
    
    data_dict['df_train_val'][f'score{alias}'] = y_train_pred
    data_dict[test_name][f'score{alias}'] = y_test_pred
    
    return (data_dict, model_obj, std_scaler)

