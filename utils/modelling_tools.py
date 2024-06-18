import os
import copy
from typing import (
    Dict, 
    List,
    Any
)
import shutil

import mlflow
import warnings

import numpy as np
import pandas as pd

from scipy.stats import uniform

from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

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
                                model_path: str, 
                                custom_params_updates=None, 
                                tags: Dict[str, str]=None, 
                                test_name: str='df_test',
                                model_type='lr',
                                normalize=True,
                                shap_explainer=shap.Explainer
                               ) -> pd.DataFrame:
    if data_dict.get(test_name) is None:
        raise ValueError(f'Incorrect `test_name`. Available keys: {set(data_dict.keys())}')

    if os.path.exists(f"{run_name}.json"):
        raise ValueError(f'Incorrect `run_name`. Experiment "{run_name}" already exists.')
        
    df_metrics: pd.DataFrame = pd.DataFrame()

    # run searching for hyperparameters:
    np.random.seed(distributions['random_state'][0])
    for i in range(n_iter):
        print(f"# {i + 1} / {n_iter}")
       
        # generate parameters
        params = generate_params_dict_from_distributions(distributions)
        if custom_params_updates is not None:
            params = custom_params_updates(params)

        # exrtract train/val datasets
        X_train = data_dict['df_train'][cols_to_use]
        Y_train = data_dict['df_train'][target_name]
        X_val = data_dict['df_val'][cols_to_use]
        Y_val = data_dict['df_val'][target_name]

        # standardize them
        if normalize:
            std_scaler = StandardScaler()
            std_scaler.fit(X_train)
            X_train = std_scaler.transform(X_train)
            X_val = std_scaler.transform(X_val)

        # train a model
        with warnings.catch_warnings(record=True) as w:
            model_obj = model(**params)
            model_obj.fit(X_train, Y_train)
        
        comment: str = ''
        if (len(w) != 0):
            comment = w[-1].message
            if isinstance(comment, ConvergenceWarning):
                df_metrics = pd.concat(
                    (
                        df_metrics, 
                        pd.DataFrame(
                            {
                                'iter': [i],
                                'GINI_train': [np.nan],
                                'GINI_val': [np.nan],
                                'GINI_prcnt_chng': [np.nan],
                                'PR_AUC_train': [np.nan],
                                'PR_AUC_val': [np.nan],
                                'PR_AUC_prcnt_chng': [np.nan],
                                'params': [params],
                                'comment': [comment]
                            }
                        )
                    )
                )
                continue
    
        # generate predictions
        y_train_pred = model_obj.predict_proba(X_train)[:, 1]
        y_val_pred = model_obj.predict_proba(X_val)[:, 1]

        # compute GINI
        # - for train
        roc_auc_train = roc_auc_score(Y_train.astype(int), y_train_pred)
        gini_train = 2 * roc_auc_train - 1
        # - for validation 
        roc_auc_val = roc_auc_score(Y_val.astype(int), y_val_pred)
        gini_val = 2 * roc_auc_val - 1

        # compute Area Under Precision Recall Curve
        # - for train
        precision, recall, _ = precision_recall_curve(Y_train.astype(int), y_train_pred)
        auc_train = auc(x=recall, y=precision)
        # - for validation
        precision, recall, _ = precision_recall_curve(Y_val.astype(int), y_val_pred)
        auc_values = auc(x=recall, y=precision)
        
        # save metrics' values + compute percentage of changes of GINI, AU PRC
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
                        'params': [params], 
                        'comment': [comment]
                    }
                )
            )
        )
        
        # logging with MLFlow
        with mlflow.start_run(run_name=f'{i}', nested=True, log_system_metrics=True) as child:
            mlflow.log_params(params)
            mlflow.log_metric(f"Val-PR_AUC", auc_values)
            mlflow.log_metric(f"Val-GINI", gini_val)

            # save the model
            mlflow.sklearn.save_model(model_obj, f'{run_name}-{i}')
            mlflow.sklearn.log_model(model_obj, f'{run_name}-{i}')
            
            # tag the run
            if tags is not None:
                for tag_key, tag_value in tags.items():
                    mlflow.set_tag(tag_key, tag_value)
    if df_metrics.GINI_train.isna().all():
        return (df_metrics, data_dict, np.nan, np.nan)
    else:
        # extract the most optimal model: 
        # select one with the lowest absolute value of a percantage 
        # change (train <-> val) of AU PRC in top 20 largest AU PRC on the validation dataset 
        df_metrics = df_metrics.reset_index(drop=True)
        df_metrics_top = df_metrics.sort_values('PR_AUC_val', ascending=False, ignore_index=True).head(20)
        
        best_iter = df_metrics_top.loc[df_metrics_top.PR_AUC_prcnt_chng.abs().argmin(), 'iter']
        best_params = df_metrics.loc[df_metrics.iter == best_iter, 'params'].values[0]
        best_gini = df_metrics.loc[df_metrics.iter == best_iter, 'GINI_val'].values[0] 
        best_pr_auc = df_metrics.loc[df_metrics.iter == best_iter, 'PR_AUC_val'].values[0] 
    
        # logging with MLFlow params, AU PRC, GINI
        print("Best params:\n", best_params)
        print(f"Val-PR_AUC: ", best_pr_auc)
        print(f"Val-GINI: ", best_gini)
    
        mlflow.log_metric(f"Val-PR_AUC", best_pr_auc)
        mlflow.log_metric(f"Val-GINI", best_gini)
        mlflow.log_params(best_params)
    
        # load the best model, make predictions to plot ROC, PRC
        X_train = data_dict['df_train'][cols_to_use]
        Y_train = data_dict['df_train'][target_name]
        X_val = data_dict['df_val'][cols_to_use]
        Y_val = data_dict['df_val'][target_name]

        if normalize:
            std_scaler = StandardScaler()
            std_scaler.fit(X_train)
        
            X_train = std_scaler.transform(X_train)
            X_val = std_scaler.transform(X_val)
    
        model_obj = mlflow.sklearn.load_model(f'{run_name}-{best_iter}')
        y_train_pred = model_obj.predict_proba(X_train)[:, 1]
        y_val_pred = model_obj.predict_proba(X_val)[:, 1]
        
        data_dict['df_train']['score'] = y_train_pred
        data_dict['df_val']['score'] = y_val_pred
    
        # - ROC
        #   train-validation
        #   -- train
        roc_curve_d = RocCurveDisplay.from_estimator(
            model_obj, X_train, Y_train, name='Train', color=palette['Train'], plot_chance_level=True
        )
        ax = plt.gca()
        #   -- validation
        _ = RocCurveDisplay.from_estimator(model_obj, X_val, Y_val, ax=ax, name='Validation', color=palette['Val'])
        # roc_curve_d.plot(ax=ax, alpha=0.8)
        f = roc_curve_d.figure_
        mlflow.log_figure(f, "ROC-Train-Val.png") 
    
        # - PRC
        #   train-validation
        #   -- train
        precision_train, recall_train, _ = precision_recall_curve(Y_train.astype(int), y_train_pred)
        f, ax_pr_rec_train = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
        pr_train = PrecisionRecallDisplay(precision=precision_train, recall=recall_train)
        pr_train.plot(ax=ax_pr_rec_train, name='Train', color=palette['Train'])
        #   -- validation
        precision_val, recall_val, _ = precision_recall_curve(Y_val.astype(int), y_val_pred)
        pr_val = PrecisionRecallDisplay(precision=precision_val, recall=recall_val)
        pr_val.plot(ax=ax_pr_rec_train, name='Validation', color=palette['Val'])
        _ = plt.title('Precision Recall Curve')
        mlflow.log_figure(f, "PRC-Train-Val.png")    
    
        # Train on the full 'Train + Validation'
        X_train_val = data_dict['df_train_val'][cols_to_use]
        Y_train_val = data_dict['df_train_val'][target_name]
        X_test = data_dict[test_name][cols_to_use]
        Y_test = data_dict[test_name][target_name]
        
        std_scaler = None
        if normalize:
            std_scaler = StandardScaler()
            std_scaler.fit(X_train_val)
            X_train_val = std_scaler.transform(X_train_val)
            X_test = std_scaler.transform(X_test)
    
        model_obj = model(**best_params)
        model_obj.fit(X_train_val, Y_train_val)
        
        y_train_val_pred = model_obj.predict_proba(X_train_val)[:, 1]
        y_test_pred = model_obj.predict_proba(X_test)[:, 1]
    
        data_dict['df_train_val']['score'] = y_train_val_pred
        data_dict[test_name]['score'] = y_test_pred
    
        try:
            mlflow.sklearn.save_model(model_obj, os.path.join(model_path, run_name))
            mlflow.sklearn.log_model(model_obj, os.path.join(model_path, run_name))
        except:
            print('Not saved')
    
        # GINI
        roc_auc_test = roc_auc_score(Y_test, y_test_pred)
        gini_test = 2 * roc_auc_test - 1
    
        # AU PRC
        precision_test, recall_test, _ = precision_recall_curve(Y_test, y_test_pred)
        auc_test = auc(x=recall_test, y=precision_test)
    
        print("Test-PR_AUC", auc_test)
        print("Test-Gini", gini_test)
        mlflow.log_metric(f"Test-PR_AUC", auc_test)
        mlflow.log_metric(f"Test-GINI", gini_test)
    
        mlflow.log_metric(
            f"val_test-prcnt_chng-PR_AUC", 
            (auc_test - best_pr_auc) * 100 / best_pr_auc
        )
        mlflow.log_metric(
            f"val_test-prcnt_chng-GINI", 
            (gini_test - best_gini) * 100 / best_gini
        )

        # ROC
        #   validation-test
        #   -- validation
        roc_curve_d = RocCurveDisplay.from_predictions(
            y_true=data_dict['df_val'][target_name], 
            y_pred=data_dict['df_val']['score'],
            name='Validation', 
            color=palette['Val'],
            plot_chance_level=True
        )
        ax = plt.gca()
        #   -- test
        _ = RocCurveDisplay.from_predictions(
            y_true=data_dict[test_name][target_name], 
            y_pred=data_dict[test_name]['score'],
            ax=ax, name='Test', color=palette['Test'])
        # roc_curve_d.plot(ax=ax, alpha=0.8)
        f = roc_curve_d.figure_
        mlflow.log_figure(f, "ROC-Val-Test.png")
        
        # PRC
        #   validation-test
        #   -- validation
        f, ax_pr_rec = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
        pr_val = PrecisionRecallDisplay(precision=precision_val, recall=recall_val)
        pr_val.plot(ax=ax_pr_rec, name='Validation', color=palette['Val'])
        _ = plt.title('Precision Recall Curve')
        
        pr_val = PrecisionRecallDisplay(precision=precision_test, recall=recall_test)
        pr_val.plot(ax=ax_pr_rec, name='Test', color=palette['Test'])
        mlflow.log_figure(f, "PRC-Val-Test.png")    
        
        if tags is not None:
            for tag_key, tag_value in tags.items():
                mlflow.set_tag(tag_key, tag_value)
        
        # SHAP
        if model_type =='lr':
            explainer_smpl= shap_explainer(
                model_obj, data_dict[test_name + '_sample'][cols_to_use], columns=cols_to_use, 
                feature_dependence="interventional", model_output="predict_proba"
            )
        else:
            explainer_smpl= shap_explainer(
                model_obj, feature_perturbation="tree_path_dependent", model_output="raw"
            )

        explainer_obj = explainer_smpl(data_dict[test_name + '_sample'][cols_to_use])
        n_features = len(cols_to_use)
        if model_type =='lr':
            try:
                f, ax = plt.subplots(nrows=n_features, ncols=1, figsize=(20, 100))
                for i, col in enumerate(cols_to_use):
                    _ = shap.plots.scatter(explainer_obj[:, col], ax=ax[i], show=False)
                
                mlflow.log_figure(f, "SHAP_test_sample_per_obs.png")
                plt.close()
            except Exception as e:
                warnings.warn(f'Error while SHAP scatter plot generation: {e}')
        else:
            f, ax = plt.subplots(nrows=n_features, ncols=1, figsize=(20, 100))
            for i, col in enumerate(cols_to_use):
                _ = shap.plots.scatter(explainer_obj[:, col, 1], ax=ax[i], show=False)
            
            mlflow.log_figure(f, "SHAP_test_sample_per_obs.png")
            plt.close()
        
        if model_type =='lr':
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            shap.summary_plot(
                explainer_obj, data_dict[test_name + '_sample'][cols_to_use], 
                plot_type="bar", feature_names=cols_to_use, plot_size=(5, 10), show=False
                # max_daisplay=len(cols_to_use)
            )
            plt.show()
            mlflow.log_figure(f, "SHAP_test_sample_features_importance.png") 
        else:
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            shap.summary_plot(
                explainer_obj[:, :, 1], data_dict[test_name + '_sample'][cols_to_use], 
                plot_type="bar", feature_names=cols_to_use, plot_size=(5, 10), show=False
                # max_daisplay=len(cols_to_use)
            )
            plt.show()
            mlflow.log_figure(f, "SHAP_test_sample_features_importance.png") 

        # mean absolute SHAP value as feature importance
        if model_type =='lr':
            vals = np.abs(explainer_obj.values).mean(0)
            df_shap_imp = pd.DataFrame(list(zip(cols_to_use, vals)), columns=['Feature', 'Importance(SHAP)'])
            df_shap_imp['Importance(SHAP, %)'] = df_shap_imp['Importance(SHAP)'] * 100 / (df_shap_imp['Importance(SHAP)'].sum())
        else:
            vals = np.abs(explainer_obj[:, :, 1].values).mean(0)
            df_shap_imp = pd.DataFrame(list(zip(cols_to_use, vals)), columns=['Feature', 'Importance(SHAP)'])
            df_shap_imp['Importance(SHAP, %)'] = df_shap_imp['Importance(SHAP)'] * 100 / (df_shap_imp['Importance(SHAP)'].sum())


        # coefficients of Logistic Regression as importance
        
        if model_type == 'lr':
            df_coeff = coeff_stats(X_train_val, cols_to_use, model_obj)
            df_coeff = pd.merge(df_coeff, df_shap_imp, on='Feature', how='outer')
            mlflow.log_table(data=df_coeff, artifact_file=f"{run_name}-best_coeff.json")
        else:
            df_coeff = df_shap_imp
            mlflow.log_table(data=df_shap_imp, artifact_file=f"{run_name}-best_model-shap-imp.json")

        mlflow.log_table(data=df_metrics, artifact_file=f"{run_name}-hypertunning.json")


        # remove directory to save the model of the current run (if not the first run of the experiment)
        for i in range(n_iter):
            if os.path.exists(f'{run_name}-{i}'):
                shutil.rmtree(f'{run_name}-{i}')  
        return (df_metrics, df_coeff, data_dict, model_obj, std_scaler)


def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.

 
    For a vector of maximum likelihood estimates theta, its variance-covariance matrix can be estimated as inverse(H), where H is the Hessian matrix of log-likelihood at theta.

 
    """
    from scipy.stats import norm
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p


def coeff_stats(train_data, cols_to_use: List[str], model) -> pd.DataFrame:
    """`train_data` - Standardized """
    df_coeff = pd.DataFrame(
        {
            'Feature': cols_to_use + ['Intercept'],
            'Coeff': np.append(model.coef_[0], model.intercept_)
        }
    )
    df_coeff['Importance(%)'] = df_coeff['Coeff'].abs() * 100 / df_coeff.Coeff.abs().sum()
    df_p = pd.DataFrame(
        {
            'p-value': logit_pvalue(model, train_data), 
            'Feature': ['Intercept'] + cols_to_use,
        }
    ) 
    df_coeff = pd.merge(df_coeff, df_p, on='Feature', how='outer')
    return df_coeff


def train_pred(data_dict: Dict[str, pd.DataFrame], 
              cols_to_use: List[str], 
              target_name: str, 
              model, 
              params,
              alias: str='',
              train_name='df_train_val',
              test_name: str='df_test', 
              normalize=True) -> Dict[str, Any]:
    X_train = data_dict[train_name][cols_to_use]
    Y_train = data_dict[train_name][target_name]

    X_test = data_dict[test_name][cols_to_use]
    Y_test = data_dict[test_name][target_name]

    std_scaler = None
    if normalize:
        std_scaler = StandardScaler()
        std_scaler.fit(X_train)
        X_train = std_scaler.transform(X_train)
        X_test = std_scaler.transform(X_test)

    model_obj = model(**params)
    model_obj.fit(X_train, Y_train)

    y_train_pred = model_obj.predict_proba(X_train)[:, 1]
    y_test_pred = model_obj.predict_proba(X_test)[:, 1]
    
    data_dict[train_name][f'score{alias}'] = y_train_pred
    data_dict[test_name][f'score{alias}'] = y_test_pred
    
    return (data_dict, model_obj, std_scaler)


def evaluate_model(data: pd.DataFrame, 
                   cols_to_use: List[str],
                   std_scaler,
                   target_name: str,
                   model_obj):
    if std_scaler is not None:
        X = std_scaler.transform(data[cols_to_use])
    else:
        X = data[cols_to_use]
    Y = data[target_name]
    y_pred = model_obj.predict_proba(X)[:, 1]

    # GINI
    roc_auc_value = roc_auc_score(Y, y_pred)
    gini_value = 2 * roc_auc_value - 1

    # AU PRC
    precision_value, recall_value, _ = precision_recall_curve(Y, y_pred)
    auc_value = auc(x=recall_value, y=precision_value)

    print("PR_AUC", auc_value)
    print("Gini", gini_value)   
    return auc_value, gini_value


def shap_analysis(data: pd.DataFrame, model_obj, cols_to_use: List[str]):
    # SHAP
    explainer_smpl= shap.Explainer(
        model_obj, data[cols_to_use], columns=cols_to_use, 
        feature_dependence="interventional", model_output="predict_proba"
    )
    explainer_obj = explainer_smpl(data[cols_to_use])
    n_features = len(cols_to_use)
    
    f, ax = plt.subplots(nrows=n_features, ncols=1, figsize=(20, 100))
    for i, col in enumerate(cols_to_use):
        _ = shap.plots.scatter(explainer_obj[:, col], ax=ax[i], show=True)
    plt.show()

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    shap.summary_plot(
        explainer_obj[:, :], data[cols_to_use], 
        plot_type="bar", feature_names=cols_to_use, plot_size=(5, 10), show=True,
        max_daisplay=len(cols_to_use)
    )
    plt.show()

    return {'explainer': explainer_smpl, 'shap_object': explainer_obj}
