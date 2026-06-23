"""
Code for HMGWX
Author: Fan Gao, gaofancj@gmail.com
Date: June 23, 2026
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from itertools import product
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from tqdm import tqdm
from sklearn.cluster import KMeans
import re
import math
from sklearn.neighbors import KDTree
from joblib import parallel_backend
import math
import time


class XGBoostTrainer_1:
    def __init__(self, n_splits=5, random_state=42, n_jobs=-1):
        self.best_model = None
        self.best_params = None
        self.oof_predictions = None
        self.cv_scores = []
        self.n_splits = n_splits
        self.random_state = random_state
        self.best_iterations_per_fold = []
        self.oof_shap_values = None
        self.n_jobs = n_jobs  # 控制并行线程数

    def _train_fold(self, params, X, y, train_idx, val_idx, fold_num):
        """训练单个fold的辅助函数，用于并行化"""
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid_fold = xgb.DMatrix(X_val_fold, label=y_val_fold)

        watchlist = [(dvalid_fold, 'eval')]

        model = xgb.train(params,
                         dtrain_fold,
                         num_boost_round=1000,
                         evals=watchlist,
                         early_stopping_rounds=30,
                         verbose_eval=False)

        return {
            'fold_num': fold_num,
            'score': model.best_score,
            'model': model,
            'val_idx': val_idx
        }

    def objective(self, params, X, y):
        """使用joblib并行化的目标函数"""
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])

        param_use = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'nthread': 1,
            **params
        }

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(X, y))

        # 使用joblib并行训练各个fold
        with parallel_backend('loky', n_jobs=self.n_jobs): 
            results = Parallel()(
                delayed(self._train_fold)(
                    param_use, X, y, train_idx, val_idx, fold_num
                )
                for fold_num, (train_idx, val_idx) in enumerate(folds)
            )

        # 收集结果
        cv_scores = [res['score'] for res in results]
        avg_cv_score = np.mean(cv_scores)
        
        return {'loss': avg_cv_score, 'status': STATUS_OK}

    def _cv_fold(self, X, y, train_idx, val_idx, fold_num):
        """用于最终CV的并行化辅助函数"""
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid_fold = xgb.DMatrix(X_val_fold, label=y_val_fold)

        watchlist = [(dtrain_fold, 'train'), (dvalid_fold, 'eval')]

        fold_model = xgb.train(self.best_params,
                             dtrain_fold,
                             num_boost_round=2000,
                             evals=watchlist,
                             early_stopping_rounds=50,
                             verbose_eval=100 if fold_num == 0 else False)  # 只打印第一个fold的日志

        best_iteration = fold_model.best_iteration
        fold_score = fold_model.best_score

        # 计算该fold的预测和SHAP值
        oof_preds_fold = fold_model.predict(dvalid_fold, iteration_range=(0, best_iteration))
        
        fold_explainer = shap.TreeExplainer(fold_model)
        fold_shap_value = fold_explainer.shap_values(dvalid_fold)

        return {
            'fold_num': fold_num,
            'best_iteration': best_iteration,
            'score': fold_score,
            'val_idx': val_idx,
            'oof_pred': oof_preds_fold,
            'shap_values': fold_shap_value,
            'model': fold_model
        }

    def calcu_oof_and_shap(self, X, y, tune=True, max_evals=50):
        """使用并行化计算OOF预测和SHAP值"""
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if tune or self.best_params is None:
            self.tune_params(X, y, max_evals=max_evals)
        elif not tune and self.best_params is None:
            raise ValueError("Cannot train without tuning if best_params are not already set.")

        print(f"\nStarting {self.n_splits}-Fold CV for OOF predictions (parallel)...")
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(X, y))

        # 初始化存储结构
        self.oof_predictions = np.zeros(X.shape[0])
        self.oof_shap_values = np.zeros(X.shape)
        self.cv_scores = []
        self.best_iterations_per_fold = []

        # 并行处理各个fold
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._cv_fold)(X, y, train_idx, val_idx, fold_num)
            for fold_num, (train_idx, val_idx) in enumerate(folds)
        )

        # 合并结果
        for res in results:
            self.best_iterations_per_fold.append(res['best_iteration'])
            self.cv_scores.append(res['score'])
            self.oof_predictions[res['val_idx']] = res['oof_pred']
            self.oof_shap_values[res['val_idx']] = res['shap_values']

        # 计算总体指标
        oof_rmse = np.sqrt(mean_squared_error(y, self.oof_predictions))
        oof_r2 = r2_score(y, self.oof_predictions)
        
        # 训练最终模型
        print("\n--- Training Final Model on Full Data ---")
        final_num_boost_round = int(np.median(self.best_iterations_per_fold))
        dfull = xgb.DMatrix(X, label=y)

        self.best_model = xgb.train(self.best_params,
                                  dfull,
                                  num_boost_round=final_num_boost_round,
                                  verbose_eval=100)

        print("Final model training complete.")
        return self.best_model, self.oof_predictions, oof_rmse, oof_r2, self.oof_shap_values

    # 保留原有的tune_params、predict和compute_shap_values方法
    def tune_params(self, X, y, max_evals=50):
        """与之前相同的参数调优方法"""
        search_space = {
            "max_depth": hp.quniform('max_depth', 3, 8, 1),
            "learning_rate": hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            "subsample": hp.uniform('subsample', 0.6, 1.0),
            "colsample_bytree": hp.uniform('colsample_bytree', 0.6, 1.0),
            "min_child_weight": hp.quniform('min_child_weight', 1, 6, 1),
            "gamma": hp.uniform('gamma', 0, 0.5),
            "reg_alpha": hp.loguniform('reg_alpha', np.log(0.001), np.log(1.0)),
            "reg_lambda": hp.loguniform('reg_lambda', np.log(0.1), np.log(10.0)),
        }

        trials = Trials()
        best = fmin(
            fn=lambda params: self.objective(params, X, y),
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state)
        )

        self.best_params = {
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'min_child_weight': int(best['min_child_weight']),
            'gamma': best['gamma'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda'],
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': self.random_state
        }
        print("\nBest parameters found:")
        print(self.best_params)
        return self.best_params

    def predict(self, X):
        """预测方法"""
        dmatrix = xgb.DMatrix(X)
        return self.best_model.predict(dmatrix)

    def compute_shap_values(self, X):
        """计算SHAP值"""
        explainer = shap.TreeExplainer(self.best_model)
        dmatrix = xgb.DMatrix(X)
        return explainer.shap_values(dmatrix)

def _parallel_process_region(
    # 大型数据数组，joblib会高效处理
    data, 
    target, 
    locations, 
    # 每个任务的特定数据
    local_indices, 
    center_loc, 
    region_id,
    # 控制参数
    mode_is_cv, 
    use_spatial_weights, 
    n_splits, 
    tune_evals
):
    """
    This function is designed to be called in parallel by joblib.
    It processes a single geographic region, tunes hyperparameters, and trains an XGBoost model.
    """
    # --- 数据准备 ---
    X_local, y_local = data[local_indices], target[local_indices]
    locs_local = locations[local_indices]
    
    # --- 权重计算 (如果需要) ---
    weights_local = None
    if use_spatial_weights:
        # 使用 Bi-square kernel 计算空间权重
        distances = np.linalg.norm(locs_local - center_loc, axis=1)
        bandwidth = np.max(distances)
        if bandwidth > 0:
            u = distances / bandwidth
            weights = (1 - u**2)**2
            weights[u > 1] = 0
            weights_local = weights
        else: # 如果所有点都在同一位置
            weights_local = np.ones(len(locs_local))

    # --- 超参数搜索空间 ---
    search_space = {
        "max_depth": hp.quniform('max_depth', 2, 4, 1),
        "learning_rate": hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        "subsample": hp.uniform('subsample', 0.6, 1.0),
        "colsample_bytree": hp.uniform('colsample_bytree', 0.6, 1.0),
        "min_child_weight": hp.quniform('min_child_weight', 1, 6, 1),
        "gamma": hp.uniform('gamma', 0, 1.0),
        "reg_alpha": hp.loguniform('reg_alpha', np.log(0.001), np.log(1.0)),
        "reg_lambda": hp.loguniform('reg_lambda', np.log(1.0), np.log(10.0))
    }
    
    # --- Hyperopt 目标函数 ---
    def _objective_for_tuning(params, X, y, weights):
        params.update({'max_depth': int(params['max_depth']), 'min_child_weight': int(params['min_child_weight'])})
        params['nthread'] = 2

        dmatrix = xgb.DMatrix(X, y, weight=weights) # 如果 weights is None, XGBoost 会忽略它
        
        if mode_is_cv:
            cv_results = xgb.cv(params, dmatrix, nfold=n_splits, num_boost_round=1000,
                                early_stopping_rounds=30, seed=42, verbose_eval=False)
            return {'loss': cv_results['test-rmse-mean'].min(), 'status': STATUS_OK}
        else:
            # 权重也需要一起切分
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
            w_train, w_val = (None, None)
            if weights is not None:
                w_train, w_val = train_test_split(weights, test_size=0.3, random_state=42)

            dtrain = xgb.DMatrix(X_train, y_train, weight=w_train)
            dval = xgb.DMatrix(X_val, y_val, weight=w_val)
            model = xgb.train(params, dtrain, num_boost_round=1000,
                              evals=[(dval, 'eval')], early_stopping_rounds=30, verbose_eval=False)
            return {'loss': model.best_score, 'status': STATUS_OK}

    # --- 数据划分与调优 ---
    if not mode_is_cv:
        indices_range = np.arange(len(X_local))
        train_val_idx, test_idx = train_test_split(indices_range, test_size=0.3, random_state=42)
        X_train_val, y_train_val = X_local[train_val_idx], y_local[train_val_idx]
        w_train_val = weights_local[train_val_idx] if weights_local is not None else None
        X_test, y_test = X_local[test_idx], y_local[test_idx]
        X_for_tuning, y_for_tuning, w_for_tuning = X_train_val, y_train_val, w_train_val
    else:
        X_for_tuning, y_for_tuning, w_for_tuning = X_local, y_local, weights_local

    best = fmin(fn=lambda p: _objective_for_tuning(p, X_for_tuning, y_for_tuning, w_for_tuning),
                space=search_space, algo=tpe.suggest, max_evals=tune_evals,
                trials=Trials(), rstate=np.random.default_rng(42), verbose=False)

    best_params_local = {**best, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 42, 'nthread': 2}
    best_params_local.update({'max_depth': int(best_params_local['max_depth']), 'min_child_weight': int(best_params_local['min_child_weight'])})
    
    # --- 模型训练与返回 ---
    if mode_is_cv: # OOF 模式
        oof_preds_local, best_iterations, fold_models, fold_val_indices = np.zeros(len(local_indices)), [], [], []
        for train_idx, val_idx in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_local):
            w_train = weights_local[train_idx] if weights_local is not None else None
            w_val = weights_local[val_idx] if weights_local is not None else None
            dtrain = xgb.DMatrix(X_local[train_idx], y_local[train_idx], weight=w_train)
            dvalid = xgb.DMatrix(X_local[val_idx], y_local[val_idx], weight=w_val)
            model = xgb.train(best_params_local, dtrain, num_boost_round=2000, evals=[(dvalid, 'eval')], early_stopping_rounds=50, verbose_eval=False)
            best_iterations.append(model.best_iteration)
            oof_preds_local[val_idx] = model.predict(dvalid)
            fold_models.append(model)
            fold_val_indices.append(val_idx)
        
        d_local_full = xgb.DMatrix(X_local, y_local, weight=weights_local)
        final_model = xgb.train(best_params_local, d_local_full, num_boost_round=int(np.median(best_iterations) if best_iterations else 100))
        return {'final_model': final_model, 'best_params': best_params_local, 'original_indices': local_indices,
                'oof_preds': oof_preds_local, 'fold_models': fold_models, 'fold_val_indices': fold_val_indices, 'mode': 'oof'}
    else: # Train-Test split 模式
        d_train_val = xgb.DMatrix(X_train_val, label=y_train_val, weight=w_train_val)
        final_model = xgb.train(best_params_local, d_train_val, num_boost_round=100)
        dtest = xgb.DMatrix(X_test)
        test_predictions = final_model.predict(dtest)
        return {'final_model': final_model, 'best_params': best_params_local,
                'test_indices': local_indices[test_idx], 'test_predictions': test_predictions, 'mode': 'test_split'}

class GeoXGBTrainer:
    
    def __init__(self, data, target, locations, k_nearest=100, n_clusters=250, n_splits=5, n_jobs=-1, 
                 tune_evals=30, use_full_sample=False, use_spatial_weights=False):

        """
        GeoXGBTrainer: Geographically weighted XGBoost training framework.

        This class implements a geographically weighted XGBoost model,
        where multiple local XGBoost models are trained on spatial neighborhoods.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Feature matrix of shape (n_samples, n_features).

        target : np.ndarray or pd.Series
            Target variable of shape (n_samples,).

        locations : array-like
            Spatial coordinates for each sample, shape (n_samples, n_dims).
            Example: [(latitude, longitude), ...]

        k_nearest : int, default=100
            Number of nearest neighbors used to train each local model.
            👉 Key parameter controlling spatial locality:
               - Smaller → more local (higher variance)
               - Larger → more global (higher bias)

        n_clusters : int, default=250
            Number of spatial clusters (via KMeans) used to define local regions.
            👉 Each cluster center corresponds to one local model.
            👉 Larger value → more models, higher spatial resolution, slower training

        n_splits : int, default=5
            Number of folds for K-Fold cross-validation (used in OOF mode only).

        n_jobs : int, default=-1
            Number of parallel jobs (joblib).
            👉 -1 means using all available CPU cores.

        tune_evals : int, default=30
            Number of hyperparameter tuning iterations per local model.
            👉 Higher → better performance, but slower.

        use_full_sample : bool, default=False
            Whether to use full-sample mode:
            - True  → every sample acts as a center (no CV)
            - False → use clustered centers + OOF evaluation (recommended)

        use_spatial_weights : bool, default=False
            Whether to use spatial weighting when training the local model and aggregating predictions.
            👉 True → apply distance-decay weighting (similar to GWR)
            👉 False → simple averaging
        """
        self.data = data
        self.target = target
        self.locations = np.array(locations)
        self.k_nearest = k_nearest
        self.n_clusters = n_clusters
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.tune_evals = tune_evals
        self.use_full_sample = use_full_sample
        self.use_spatial_weights = use_spatial_weights

        # 结果存储
        self.final_models, self.centers_used_for_fitting, self.best_params_per_region = [], None, []
        self.oof_predictions = np.full(self.data.shape[0], np.nan)
        self.fold_models_per_region, self.fold_val_indices_per_region = [], []
        self.test_indices_list, self.test_predictions_list = [], []
        self._selected_indices_list = []
        self.center_locations_for_regions = []
        self.search_history =[]
        self.local_indices_list = []     
        self.local_predictions_list =[] 

    def fit(self, lightweight_mode=False):

        """
        Train spatial local XGBoost models.

        Parameters
        ----------
        lightweight_mode : bool, default=False
            Whether to run in lightweight mode (used for hyperparameter search).
            👉 True:
                - Faster execution
                - Reduced logging and storage
            👉 False:
                - Full training pipeline
                - Stores models and predictions
        """
        # --- 初始化/重置状态 ---
        self.oof_predictions = np.full(self.data.shape[0], np.nan)
        self.test_indices_list, self.test_predictions_list = [], []
        self.fold_models_per_region, self.fold_val_indices_per_region = [], []
        self.center_locations_for_regions = []

        self.local_indices_list = []
        self.local_predictions_list =[]
        
        if not lightweight_mode:
            print(f"\n--- Starting Final Model Fitting with k_nearest = {self.k_nearest} ---")
            print(f"Spatial Weights Enabled: {self.use_spatial_weights}")
        
        # --- 邻域定义 ---
        self.centers_used_for_fitting = self._get_centers(lightweight_mode)
        
        print("Building KDTree for efficient neighbor search...")
        kdt = KDTree(self.locations, metric='euclidean')
        n_points = len(self.locations)
        k_to_query = min(self.k_nearest, n_points)
        if k_to_query < self.k_nearest:
            print(f"  - Warning: k_nearest ({self.k_nearest}) is > n_points ({n_points}). Capping k at {n_points}.")
            
        neighbor_indices_list = kdt.query(self.centers_used_for_fitting, k=k_to_query, return_distance=False)
        
        # --- 准备并行任务 ---
        process_data_list = []
        for i, center_loc in enumerate(self.centers_used_for_fitting):
            indices = neighbor_indices_list[i]
            # (local_indices, center_location)
            process_data_list.append((indices, center_loc))
            self.center_locations_for_regions.append(center_loc) # 保存中心点位置，用于OOF聚合
        
        self._selected_indices_list = [item[0] for item in process_data_list]
        
        mode_is_cv = not self.use_full_sample and not lightweight_mode
        if not lightweight_mode:
             print(f"Training strategy: {'K-Fold CV (OOF)' if mode_is_cv else 'Train-Validation-Test Split'}")
        
        # 注意这里我们传递了 self.data, self.target, self.locations 作为顶级参数
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_process_region)(
                self.data, self.target, self.locations, # 大数组作为直接参数
                indices, center, i,                     # 每个任务的特定数据
                mode_is_cv, self.use_spatial_weights,   # 控制参数
                self.n_splits, self.tune_evals
            ) 
            for i, (indices, center) in enumerate(process_data_list)
        )
        
        if not results:
            print("Warning: Parallel processing returned no results.")
            return

        # --- 聚合结果 ---
        self.final_models = [res['final_model'] for res in results]
        self.best_params_per_region = [res['best_params'] for res in results]
        
        if results and results[0]['mode'] == 'oof':
            
            for res in results:
                self.local_indices_list.append(res['original_indices'])
                self.local_predictions_list.append(res['oof_preds'])
            
            if self.use_spatial_weights:
                # OOF 结果的加权聚合
                oof_pred_sum = np.zeros_like(self.oof_predictions)
                oof_weight_sum = np.zeros_like(self.oof_predictions)
                for i, res in enumerate(results):
                    indices = res['original_indices']
                    center_loc = self.center_locations_for_regions[i]
                    local_locs = self.locations[indices]
                    
                    # 重新计算权重用于聚合（这步很快）
                    distances = np.linalg.norm(local_locs - center_loc, axis=1)
                    bandwidth = np.max(distances)
                    if bandwidth > 0:
                        u = distances / bandwidth
                        weights = (1 - u**2)**2
                        weights[u > 1] = 0
                    else:
                        weights = np.ones(len(local_locs))

                    oof_pred_sum[indices] += res['oof_preds'] * weights
                    oof_weight_sum[indices] += weights
                
                non_zero_mask = oof_weight_sum > 0
                self.oof_predictions[non_zero_mask] = oof_pred_sum[non_zero_mask] / oof_weight_sum[non_zero_mask]

            else:
                # OOF 结果的简单平均聚合
                oof_pred_sum, counts = np.zeros_like(self.oof_predictions), np.zeros(len(self.data))
                for res in results:
                    indices = res['original_indices']
                    oof_pred_sum[indices] += res['oof_preds']
                    counts[indices] += 1
                non_zero_counts = counts > 0
                self.oof_predictions[non_zero_counts] = oof_pred_sum[non_zero_counts] / counts[non_zero_counts]

            # 这部分逻辑对两种模式都通用
            for res in results:
                self.fold_models_per_region.append(res['fold_models'])
                self.fold_val_indices_per_region.append(res['fold_val_indices'])

        elif results:
            # Test-set 模式的结果聚合
            self.test_indices_list = [res['test_indices'] for res in results]
            self.test_predictions_list = [res['test_predictions'] for res in results]

    def _get_centers(self, lightweight_mode):
        """
        Determine spatial centers for local model training.

        Returns
        -------
        np.ndarray
            Coordinates of centers.

        Notes
        -----
        - If use_full_sample=True:
            → All samples are used as centers.

        - Otherwise:
            → KMeans clustering is used to define region centers.
        """
        if self.use_full_sample:
            print(f"Mode: Full sample. Using all {len(self.locations)} samples as centers.")
            return self.locations
        else:
            mode_str = "Lightweight search" if lightweight_mode else "OOF"
            n_clusters = min(self.n_clusters, len(self.locations))
            print(f"Mode: {mode_str} (Clustered). Using {n_clusters} cluster centers.")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return kmeans.fit(self.locations).cluster_centers_

    def search_optimal_k_golden(self, k_range=(100, 500), tolerance=25):

        """
        Optimize k_nearest using Golden Section Search.

        Parameters
        ----------
        k_range : tuple
            Search interval (min_k, max_k).

        tolerance : int
            Stopping criterion for interval width.

        Notes
        -----
        - More efficient than interval search for continuous parameters.
        - Uses cached results to avoid redundant computation.
        """

        print(f"\n--- Starting Golden Section Search (range={k_range}, tolerance={tolerance}) ---")
        self.search_history =[]  # 重置历史记录
        a, b = k_range[0], k_range[1]
        print(f"[DEBUG] use_full_sample={self.use_full_sample}, a={a}, b={b}, len(locations)={len(self.locations)}")
        phi = (math.sqrt(5) - 1) / 2
        evaluated_k = {}

        def get_mse_for_k(k_val):
            k_val = int(round(k_val))
            k_val = min(k_val, len(self.locations))
            
            if k_val in evaluated_k:
                # 记录缓存命中
                self.search_history.append({
                    'k': k_val, 'mse': evaluated_k[k_val]['mse'], 
                    'r2': evaluated_k[k_val]['r2'], 'cached': True
                })
                return evaluated_k[k_val]['mse']
                
            print(f"  [GSS] Evaluating k_nearest = {k_val}...")
            step_start_time = time.time()
            self.k_nearest = k_val
            self.fit(lightweight_mode=True) 
            mean_test_error, r2 = self.evaluate()
            step_time = time.time() - step_start_time
            
            if mean_test_error is None:
                mean_test_error, r2 = float('inf'), -float('inf')
                
            evaluated_k[k_val] = {'mse': mean_test_error, 'r2': r2}
            # 记录实际运算
            self.search_history.append({'k': k_val, 'mse': mean_test_error, 'r2': r2, 
                                        'time_sec': round(step_time, 4), 'cached': False})
            return mean_test_error

        if abs(b - a) <= tolerance:
            print(f"  [GSS] Range width ({abs(b-a)}) <= tolerance ({tolerance}). Evaluating midpoint directly.")
            mid_k = int(round((a + b) / 2))
            get_mse_for_k(mid_k)
        else:
            c = int(round(b - phi * (b - a)))
            d = int(round(a + phi * (b - a)))

            while abs(b - a) > tolerance:
                mse_c = get_mse_for_k(c)
                mse_d = get_mse_for_k(d)

                if mse_c < mse_d:
                    b = d
                    d = c
                    c = int(round(b - phi * (b - a)))
                else:
                    a = c
                    c = d
                    d = int(round(a + phi * (b - a)))
                    
                if c == d: break

        best_k = min(evaluated_k, key=lambda k: evaluated_k[k]['mse'])
        self.k_nearest = best_k
        print(f"--- GSS Finished. Optimal k: {best_k} (Best MSE: {evaluated_k[best_k]['mse']:.4f}) ---")
        return best_k

    def search_optimal_k_fixed(self, k_range=(100, 500, 50), early_stop_patience=2):

        """
        Optimize k_nearest using interval search

        Parameters
        ----------
        k_range : tuple
            Search interval (min_k, max_k, step_size).

        early_stop_patience : int
            When MSE increases for early_stop_patience rounds, terminate the bandwidth search process early.

        """
        self.search_history =[]  # 重置历史记录
        min_error = float('inf')
        optimal_k = self.k_nearest
        consecutive_increases = 0
        last_mse = float('inf')

        print(f"--- Starting Grid Search (k_range={k_range}, patience={early_stop_patience}) ---")
        for k in range(k_range[0], k_range[1] + 1, k_range[2]):
            print(f"  [Grid] Evaluating k_nearest = {k}...")
            step_start_time = time.time()
            self.k_nearest = k
            self.fit(lightweight_mode=True) 
            mean_test_error, r2 = self.evaluate()
            step_time = time.time() - step_start_time
            
            if mean_test_error is not None:
                self.search_history.append({'k': k, 'mse': mean_test_error, 'r2': r2, 'time_sec': round(step_time, 4), 'cached': False})
                if mean_test_error < min_error:
                    min_error = mean_test_error
                    optimal_k = k
                
                if mean_test_error > last_mse:
                    consecutive_increases += 1
                else:
                    consecutive_increases = 0
                last_mse = mean_test_error
            else:
                continue

            if consecutive_increases >= early_stop_patience:
                print(f"  - Early stopping at k={k}.")
                break
                
        self.k_nearest = optimal_k
        return optimal_k

    def evaluate(self, use_full_sample=None):

        """
        Evaluate model performance.

        Returns
        -------
        mse : float
            Mean Squared Error.

        r2 : float
            R-squared score.

        Notes
        -----
        Automatically selects evaluation mode:
        - Test predictions (if available)
        - Otherwise OOF predictions
        """

        y_true, y_pred = None, None
        if self.test_indices_list:
            print("Evaluating using aggregated test-set predictions...")
            try:
                flat_indices = np.concatenate(self.test_indices_list)
                flat_predictions = np.concatenate(self.test_predictions_list)
                if len(flat_indices) == 0: return None, None
                mean_preds = pd.DataFrame({'idx': flat_indices, 'pred': flat_predictions}).groupby('idx')['pred'].mean()
                y_true, y_pred = self.target[mean_preds.index], mean_preds.values
            except ValueError: return None, None
        elif np.any(~np.isnan(self.oof_predictions)):
            eval_type = "weighted aggregated" if self.use_spatial_weights else "aggregated"
            print(f"Evaluating using {eval_type} OOF predictions...")
            valid_indices = ~np.isnan(self.oof_predictions)
            y_true, y_pred = self.target[valid_indices], self.oof_predictions[valid_indices]
        else:
            print("Warning: No predictions available to evaluate."); return None, None
        mse, r2 = mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)
        print(f"Evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")
        return mse, r2

    def prediction(self, use_full_sample=True):
        if self.test_indices_list:
            print("Returning detailed and aggregated test-set predictions.")
            try:
                flat_indices = np.concatenate(self.test_indices_list)
                flat_predictions = np.concatenate(self.test_predictions_list)
                if len(flat_indices) == 0: return pd.DataFrame(), pd.DataFrame()
                detailed_df = pd.DataFrame({
                    'sub_model_index': np.repeat(np.arange(len(self.test_indices_list)),[len(arr) for arr in self.test_indices_list]),
                    'original_index': flat_indices,
                    'prediction': flat_predictions
                })
                aggregated_df = detailed_df.groupby('original_index')['prediction'].mean().to_frame()
                return detailed_df, aggregated_df
            except ValueError: return pd.DataFrame(), pd.DataFrame()
            
        elif hasattr(self, 'local_indices_list') and self.local_indices_list:
            pred_type = "weighted aggregated" if self.use_spatial_weights else "aggregated"
            print(f"Returning detailed and {pred_type} OOF predictions.")
            
            flat_indices = np.concatenate(self.local_indices_list)
            flat_predictions = np.concatenate(self.local_predictions_list)
            
            detailed_df = pd.DataFrame({
                'sub_model_index': np.repeat(np.arange(len(self.local_indices_list)),[len(arr) for arr in self.local_indices_list]),
                'original_index': flat_indices,
                'prediction': flat_predictions
            })
            oof_df = pd.DataFrame({'prediction': self.oof_predictions}, index=np.arange(len(self.oof_predictions)))
            return detailed_df, oof_df
        # ==============================================
        else:
            print("Warning: No predictions available to return."); return pd.DataFrame(), None

class GeoXGBPredictor:

    """
    Prediction module for GeoXGB.

    Strategy
    --------
    For each new sample:
        1. Find k nearest trained local models (based on spatial distance)
        2. Generate predictions from each model
        3. Average predictions
    """

    def __init__(self, models, locations, k_nearest):
        self.models = models
        self.locations = locations
        self.k_nearest = k_nearest

    def predict(self, X_new, new_locations):
        predictions = []
        for x_sample, loc in zip(X_new, new_locations):
            distances = np.linalg.norm(self.locations - loc, axis=1)
            nearest_model_idx = np.argsort(distances)[:self.k_nearest]
            model_predictions = [self.models[idx].predict(xgb.DMatrix(x_sample.reshape(1, -1))) for idx in nearest_model_idx]
            predictions.append(np.mean(model_predictions))
        return np.array(predictions)

class GeoXGBInterpreter:

    """
    Model interpretation module using SHAP.

    Supports two modes:
    -------------------
    1. Full-sample mode (use_full_sample=True)
        → SHAP values computed per local model on test data

    2. OOF mode (use_full_sample=False)
        → Cross-validated SHAP values aggregated globally (recommended)

    Attributes
    ----------
    k_explain : int
        Number of nearest models used for explaining new samples.
    """

    def __init__(self, trainer: GeoXGBTrainer, feature_names, k_explain=5, use_full_sample=None):
        self.trainer = trainer
        self.feature_names = feature_names
        self.k_explain = k_explain
        self._oof_shap_values = None  # 内部缓存
        self.use_full_sample = use_full_sample

    def calculate_training_shap(self, use_full_sample=None):
        """
        Fits the spatial proximity XGBoost models.

        Args:
            use_full_sample: If True, use all samples as centers.
                             If False, activate oof model, and calculate the oof shap values

        Return:
            if use_full_sample=True, generate both the local (for each local model) and global model
            if use_full_sample=False, generate oof shap values (only for global model)
        """
              
        if self._oof_shap_values is not None:
            print("Returning cached OOF SHAP values.")
            return pd.DataFrame(self._oof_shap_values, columns=self.feature_names)

        current_use_full_sample = self.use_full_sample if use_full_sample is None else use_full_sample

        if current_use_full_sample:
            print("calculate shap values when use_full_sample = True")
            per_model_results_list = []
            iterator = range(len(self.trainer.final_models))
            for i in iterator:
                # model_center_loc = self.trainer.centers_used_for_fitting[i]
                test_indices = self.trainer.test_indices_list[i]
                test_X = self.trainer.data[test_indices]
                
                explainer = shap.TreeExplainer(self.trainer.final_models[i])
                shap_values = explainer.shap_values(test_X)
                
                # 为了合并，增加一列来标识是哪个子模型产生的
                df = pd.DataFrame({
                    'sub_model_index': i,  # 子模型ID
                    'original_index': test_indices,  # point id
                }, index = test_indices)

                shap_df = pd.DataFrame(shap_values, 
                                       columns=[f'shap_{name}' for name in self.feature_names], 
                                       index=test_indices)
                df_combine = pd.concat([df, shap_df], axis=1)
                per_model_results_list.append(df_combine)
                
            detailed_results_df = pd.concat(per_model_results_list)
            
            ## generate the aggregrated results 
            aggregated_shaps = detailed_results_df.drop(columns=['sub_model_index']).groupby('original_index').mean()
            
            # 补上缺失的样本，用NaN填充
            full_index_df = pd.DataFrame(index=np.arange(len(self.trainer.data)))
            final_aggregated_df = full_index_df.join(aggregated_shaps)

            return detailed_results_df, final_aggregated_df

        else:
            print("Calculating OOF SHAP values for the training set. This may take a while...")
        
            # 初始化容器
            oof_shap_sum = np.zeros_like(self.trainer.data, dtype=float)
            counts = np.zeros(self.trainer.data.shape[0])
            
            # 新增：用于存储详细结果的列表
            per_fold_results_list = []

            # 遍历每个区域的CV结果
            for i, region_indices in enumerate(self.trainer._selected_indices_list):
                X_local = self.trainer.data[region_indices]
                fold_models = self.trainer.fold_models_per_region[i]
                fold_val_indices_local = self.trainer.fold_val_indices_per_region[i]
                
                for j, model in enumerate(fold_models):
                    val_indices_local = fold_val_indices_local[j]
                    X_val = X_local[val_indices_local]
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values_fold = explainer.shap_values(X_val)

                    global_indices = region_indices[val_indices_local]
                    oof_shap_sum[global_indices] += shap_values_fold
                    counts[global_indices] += 1
                    
                    # 新增：记录每个fold的详细结果
                    df = pd.DataFrame({
                        'sub_model_index': i,  # 区域索引
                        'fold_index': j,  # fold索引
                        'original_index': global_indices,  # 全局索引
                    })
                    
                    shap_df = pd.DataFrame(
                        shap_values_fold, 
                        columns=[f'shap_{name}' for name in self.feature_names], 
                        index=global_indices
                    )
                    df = pd.concat([df, shap_df], axis=1)
                    per_fold_results_list.append(df)
            
            # 计算加权平均
            oof_shap_values = np.full_like(self.trainer.data, np.nan, dtype=float)
            non_zero_counts = counts > 0
            oof_shap_values[non_zero_counts] = oof_shap_sum[non_zero_counts] / counts[non_zero_counts][:, np.newaxis]
            
            self._oof_shap_values = oof_shap_values  # 缓存结果
            
            # 新增：合并所有fold的详细结果
            detailed_results_df = pd.concat(per_fold_results_list)
            
            # 新增：创建最终聚合结果DataFrame
            full_index_df = pd.DataFrame(index=np.arange(len(self.trainer.data)))
            final_aggregated_df = full_index_df.join(
                pd.DataFrame(oof_shap_values, columns=[f'shap_{name}' for name in self.feature_names])
            )
            
            print("OOF SHAP values calculation complete.")
            return detailed_results_df, final_aggregated_df
            
    def explain_new_data(self, X_new, new_locations):
        # only use for the newly unseen data
        X_new = np.array(X_new)
        new_locations = np.array(new_locations)
        
        all_shap_values = []
        for x_sample, loc in zip(X_new, new_locations):
            distances = np.linalg.norm(self.trainer.centers_used_for_fitting - loc, axis=1)
            nearest_model_indices = np.argsort(distances)[:self.k_explain]
            d_sample = xgb.DMatrix(x_sample.reshape(1, -1), feature_names=self.feature_names)
            
            shap_values_list = []
            for idx in nearest_model_indices:
                explainer = shap.TreeExplainer(self.trainer.final_models[idx])
                shap_values_list.append(explainer.shap_values(d_sample))
            
            avg_shap_values = np.mean(np.array(shap_values_list), axis=0)
            all_shap_values.append(avg_shap_values.flatten())
            
        return pd.DataFrame(all_shap_values, columns=self.feature_names)

class GAM_MGWX:
    
    """
    GAM-MGWX: Generalized Additive Model with Multi-scale Geographically Weighted XGBoost

    This model extends the classical Generalized Additive Model (GAM) by replacing 
    each additive component f_i(x_i) with a spatially adaptive learner based on 
    geographically weighted XGBoost (GeoXGB / GeoWXGB).

    Key Features
    ------------
    1. Multi-scale modeling:
       Each feature is assigned its own optimal spatial bandwidth (k-nearest neighbors).

    2. Automatic model selection:
       For each feature, the model automatically selects between:
       - GeoXGB   (unweighted local model)
       - GeoWXGB  (spatially weighted local model)

    3. Backfitting optimization:
       Iteratively updates each component using partial residuals.

    4. Dynamic bandwidth shrinking:
       The search range of spatial bandwidth is adaptively reduced across iterations.

    5. Damping mechanism:
       Stabilizes convergence during backfitting updates.
       Will be removed soon

    Mathematical Form
    -----------------
        y = β₀ + Σ f_i(x_i) + ε

    where each f_i(x_i) is learned via a spatially localized XGBoost model.
    
    """   
    def __init__(self, data, target, locations,
                 selected_learner_types=None,
                 k_range=None,
                 max_iterations=20,
                 score_tol=0.00001,
                 init_component_predictions=None, # Optional initial f_i(X_i)
                 use_damping_factor = True,
                 damping_factor=0.5,
                 n_jobs=-1, # NEW: Parameters for dynamic bandwidth search
                 enable_bw_shrinking=True,
                 bw_width_reduction_factor=0.85,
                 bw_stop_search_width=20,
                 bw_min_k=20,
                 use_full_sample=False, 
                 search_strategy='golden',
                 n_cluster = 250): 
        
        """
        Parameters
        ----------
        data : np.ndarray
            Feature matrix (n_samples, n_features)

        target : np.ndarray
            Target variable (n_samples,)

        locations : np.ndarray
            Spatial coordinates for each sample

        selected_learner_types : list or None
            Predefined model type for each feature:
            - 'GeoXGB'
            - 'GeoWXGB'
            If None → automatically selected during early iterations

        k_range : list of tuples or None
            Bandwidth search range per feature:
            [(min_k, max_k, step), ...]
            If None → automatically initialized

        max_iterations : int
            Maximum number of backfitting iterations

        score_tol : float
            Convergence threshold based on prediction change

        init_component_predictions : np.ndarray or None
            Initial values for component functions f_i(x_i)

        use_damping_factor : bool
            Whether to apply damping in updates

        damping_factor : float (0,1]
            Controls update smoothness:
            - small → stable but slow
            - large → faster but may oscillate

        n_jobs : int
            Number of parallel jobs

        enable_bw_shrinking : bool
            Whether to dynamically shrink bandwidth search range

        bw_width_reduction_factor : float
            Shrinking factor for search range

        bw_stop_search_width : int
            Stop shrinking when range becomes small

        bw_min_k : int
            Minimum allowed bandwidth

        use_full_sample : bool
            Whether to use all samples as centers (no clustering)

        search_strategy : str
            Bandwidth optimization method:
            - 'fixed'  → interval search
            - 'golden' → golden section search

        n_cluster : int
            Number of spatial clusters (used when use_full_sample=False)
        """
        self.data = data
        self.target = target.ravel()
        self.locations = locations
        self.n_samples, self.n_features = data.shape
        self.use_full_sample = use_full_sample
        self.search_strategy = search_strategy
        self.n_cluster = n_cluster

        if selected_learner_types is not None:
            if len(selected_learner_types) != self.n_features:
                raise ValueError(f"Provided 'selected_learner_types' has length {len(selected_learner_types)}, but data has {self.n_features} features.")
            print("Using pre-specified learner types:", selected_learner_types)
            self.selected_learner_types_ = list(selected_learner_types)
            self._auto_select_models = False
        else:
            print("No learner types specified. Models will be selected automatically in the first iteration.")
            self.selected_learner_types_ = [None] * self.n_features
            self._auto_select_models = True
        
        # Default k_range if not provided
        if k_range is None:
            k_min_default = max(bw_min_k, int(self.n_samples * 0.05))
            k_max_default = int(self.n_samples * 0.8)
            k_step_default = max(10, int((k_max_default - k_min_default) / 10))
            self.k_range_orig_ = [(k_min_default, k_max_default, k_step_default)] * self.n_features
        else:
            self.k_range_orig_ = k_range
        self.current_k_ranges_ = list(self.k_range_orig_) # Dynamic range for each feature

        self.enable_bw_shrinking = enable_bw_shrinking
        self.bw_width_reduction_factor = bw_width_reduction_factor
        self.bw_stop_search_width = bw_stop_search_width
        self.bw_min_k = bw_min_k
        self.bw_search_converged_ = [False] * self.n_features
        
        # --- Standard Initialization ---
        self.max_iterations = max_iterations
        self.score_tol_ = score_tol
        self.n_jobs = n_jobs
        
        self.use_damping_factor = use_damping_factor
        if not (0 < damping_factor <= 1.0):
            raise ValueError("damping_factor must be in the interval (0, 1].")
        self.damping_factor_ = damping_factor

        # --- State and History Variables ---
        self.intercept_ = np.mean(self.target)
        if init_component_predictions is not None:
            if init_component_predictions.shape == (self.n_samples, self.n_features):
                self.component_predictions_ = init_component_predictions.copy()
            else:
                raise ValueError("init_component_predictions must have shape (n_samples, n_features)")
        else:
            self.component_predictions_ = np.zeros((self.n_samples, self.n_features))
        self.fitted_learners_ = [None] * self.n_features
        self.optimal_bandwidths_ = [-1] * self.n_features

        # New history tracking
        self.history_ = {
            'train_mse': [],
            'bandwidths': [],  # List of lists: history['bandwidths'][iter][feature]
            'predictions': [],  # List of arrays: history['predictions'][iter][feature]
            'component_change': [],
            'component_mse':[],  # 记录每轮每个特征的 MSE
            'component_time':[]  # 记录每轮每个特征的拟合耗时
        }
        self.best_iteration_ = 0
        self.best_mse_ = np.inf
        self.best_component_predictions_, self.best_optimal_bandwidths_, self.best_fitted_learners_ = None, None, None
        self.best_detailed_predictions_ = {}

    def _fit_component(self, X_feature, target_h, k_range_feature, use_weights):

        """
        Fit a single GAM component f_i(x_i) using GeoXGB/GeoWXGB.

        Parameters
        ----------
        X_feature : np.ndarray
            Single feature column (n_samples, 1)

        target_h : np.ndarray
            Partial residual:
                h_i = y - (β₀ + Σ_{j≠i} f_j(x_j))

        k_range_feature : tuple
            Bandwidth search range for this feature

        use_weights : bool
            Whether to use spatial weighting (GeoWXGB)

        Returns
        -------
        mse : float
            Mean squared error for this component

        h_mi : np.ndarray
            Predicted component values

        optimal_k : int
            Optimal bandwidth

        trainer : GeoXGBTrainer
            Trained local model object

        comp_time : float
            Computation time

        detailed_df : pd.DataFrame
            Raw local predictions for interpretability
        """
        
        start_time = time.time() # 开始计时
        
        trainer = GeoXGBTrainer(
            data=X_feature,
            target=target_h,
            locations=self.locations,
            use_full_sample=self.use_full_sample, # 使用外部指定的采样策略
            n_splits=5,
            n_jobs=self.n_jobs,
            n_clusters=self.n_cluster if not self.use_full_sample else len(self.locations),
            tune_evals=15,
            use_spatial_weights = use_weights
        )
        
        # 根据指定的策略执行带宽搜索
        if self.search_strategy == 'golden':
            # 黄金分割搜索只需要 (min, max) 两个值
            k_range_golden = (k_range_feature[0], k_range_feature[1])
            optimal_k = trainer.search_optimal_k_golden(k_range_golden, tolerance=25)
        else:
            optimal_k = trainer.search_optimal_k_fixed(k_range_feature, early_stop_patience=2)
            
        trainer.fit(lightweight_mode=False)
        detailed_df, agg_df = trainer.prediction()
        if agg_df is not None and 'prediction' in agg_df.columns:
            h_mi = agg_df['prediction'].values
        else:
            h_mi = trainer.oof_predictions
        h_mi = np.nan_to_num(h_mi, nan=np.nanmean(h_mi))
        # Handle cases where some predictions are NaN due to not being in any cluster's oof
        mse, _ = trainer.evaluate()
        
        comp_time = time.time() - start_time # 计算该组件拟合耗时
        return mse, h_mi, optimal_k, trainer, comp_time, detailed_df

    # NEW: The dynamic bandwidth range update function
    def _update_dynamic_k_range(self, feature_index, current_k_range, found_k):
        
        """
        Dynamically shrink bandwidth search range.

        Strategy
        --------
        - Center new range around current optimal k
        - Gradually reduce search width
        - Stop when:
            * range is sufficiently small
            * or convergence is detected

        Purpose
        -------
        Improves efficiency and stabilizes bandwidth estimation.
        """

        if not self.enable_bw_shrinking or self.bw_search_converged_[feature_index] or found_k < self.bw_min_k:
            return current_k_range

        current_min, current_max, current_step = current_k_range
        current_width = current_max - current_min
        reduction_factor = min(0.99, self.bw_width_reduction_factor)
        new_width = int(math.ceil(current_width * reduction_factor))

        if new_width < self.bw_stop_search_width or new_width <= 2 * current_step:
            print(f"  - BW search for feature {feature_index} has converged.")
            self.bw_search_converged_[feature_index] = True
            return current_k_range

        new_half_width = max(1, new_width // 2)
        new_min_k = max(self.bw_min_k, found_k - new_half_width)
        new_max_k = new_min_k + new_width
        new_max_k = min(new_max_k, self.n_samples)

        if new_min_k >= new_max_k:
            self.bw_search_converged_[feature_index] = True
            return current_k_range

        # Optional: reduce step size as range shrinks
        new_step = max(10, int(new_width / 10))
        next_range = (new_min_k, new_max_k, new_step)
        return next_range
        
    def fit(self):

        """
        Fit the GAM-MGWX model using backfitting.

        Algorithm Overview
        ------------------
        Repeat until convergence:

        1. Update intercept:
            β₀ = mean(y - Σ f_i)

        2. For each feature i:
            a. Compute partial residual
            b. Fit local model (GeoXGB / GeoWXGB)
            c. Update component f_i using damping

        3. Update prediction:
            ŷ = β₀ + Σ f_i

        4. Check convergence:
            ||y_new - y_old|| / ||y_old||

        Additional Mechanisms
        ---------------------
        - Automatic model selection (first few iterations)
        - Dynamic bandwidth shrinking
        - Best-model tracking (early stopping style)
        """
        print("--- Starting GAM-MGWX Fitting (with Component Score Convergence) ---")
        y_pred_old = self.intercept_ + np.sum(self.component_predictions_, axis=1)
        initial_mse = mean_squared_error(self.target.ravel(), y_pred_old)
        self.history_['train_mse'].append(initial_mse)
        
        # 初始化用于追踪最佳模型的变量
        self.best_mse_ = initial_mse
        self.best_component_predictions_ = self.component_predictions_.copy()
        self.best_optimal_bandwidths_ = list(self.optimal_bandwidths_)
        print(f" Initial State: Train MSE={initial_mse:.6f}")

        for m in tqdm(range(self.max_iterations), desc="Backfitting Cycles"):
            predictions_before_cycle = self.component_predictions_.copy()
            iter_bandwidths = [-1] * self.n_features
            iter_predictions = np.zeros((self.n_samples, self.n_features))
            iter_component_mse = [-1.0] * self.n_features   # 记录本轮各分量MSE
            iter_component_time =[0.0] * self.n_features # 记录本轮各分量耗时
            
            iter_detailed_preds = {}
            
            current_total_f = np.sum(self.component_predictions_, axis=1)
            self.intercept_ = np.mean(self.target - current_total_f)

            for i in range(self.n_features):
                X_feature = self.data[:, i].reshape(-1, 1)
                
                # --- Calculate Partial Residual ---
                current_F = self.intercept_ + np.sum(self.component_predictions_, axis=1)
                f_i_old = self.component_predictions_[:, i]
                target_h = self.target - (current_F - f_i_old)                
             
                # Use the current dynamic k-range for this feature
                variable_k_range = self.current_k_ranges_[i]
                
                # --- Automatic Model Selection (in first five iteration) ---
                h_mi = None
                if m <= 3 and self._auto_select_models:
                    print(f"\nCycle 0, Feature {i}: Selecting best model (k_range={variable_k_range})...")
                    # Fit GeoWXGB (weighted)
                    mse_w, h_w, k_w, trainer_w, time_w, df_w = self._fit_component(X_feature, target_h, variable_k_range, use_weights=True)
                    print(f"  - GeoWXGB (weighted) -> MSE: {mse_w:.4f}, k: {k_w}")
                    
                    # Fit GeoXGB (unweighted)
                    mse_uw, h_uw, k_uw, trainer_uw, time_uw, df_uw = self._fit_component(X_feature, target_h, variable_k_range, use_weights=False)
                    print(f"  - GeoXGB (unweighted) -> MSE: {mse_uw:.4f}, k: {k_uw}")

                    if mse_w < mse_uw:
                        self.selected_learner_types_[i], h_mi, self.optimal_bandwidths_[i], self.fitted_learners_[i] = 'GeoWXGB', h_w, k_w, trainer_w
                        iter_component_mse[i] = mse_w
                        iter_detailed_preds[f'x{i+1}'] = df_w
                        print(f"  --> Selected: GeoWXGB for feature {i}")
                    else:
                        self.selected_learner_types_[i], h_mi, self.optimal_bandwidths_[i], self.fitted_learners_[i] = 'GeoXGB', h_uw, k_uw, trainer_uw
                        iter_component_mse[i] = mse_uw
                        iter_detailed_preds[f'x{i+1}'] = df_uw
                        print(f"  --> Selected: GeoXGB for feature {i}")
                    iter_component_time[i] = time_w + time_uw
                
                # --- Use selected model for subsequent iterations ---
                else:
                    learner_type = self.selected_learner_types_[i]
                    use_weights = (learner_type == 'GeoWXGB')
                    mse_iter, h_mi, k_iter, trainer, time_iter, detailed_df = self._fit_component(X_feature, target_h, variable_k_range, use_weights=use_weights)
                    
                    self.optimal_bandwidths_[i] = k_iter
                    self.fitted_learners_[i] = trainer
                    iter_component_mse[i] = mse_iter
                    iter_component_time[i] = time_iter
                    iter_detailed_preds[f'x{i+1}'] = detailed_df
                
                h_mi_centered = h_mi - np.mean(h_mi)
                if self.use_damping_factor:
                    new_f_i = (1 - self.damping_factor_) * f_i_old + self.damping_factor_ * h_mi_centered
                else: 
                    new_f_i = h_mi_centered
                self.component_predictions_[:, i] = new_f_i
                
                self.current_k_ranges_[i] = self._update_dynamic_k_range(i, variable_k_range, self.optimal_bandwidths_[i])
                
                
                # Store history for this feature in this iteration
                iter_bandwidths[i] = self.optimal_bandwidths_[i]
                iter_predictions[:, i] = self.component_predictions_[:, i]

            # Recalculate overall prediction and MSE
            epsilon = 1e-8 # 防止除以零
            y_pred_new = self.intercept_ + np.sum(self.component_predictions_, axis=1)
            prediction_change = np.linalg.norm(y_pred_new - y_pred_old) / (np.linalg.norm(y_pred_old) + epsilon)
            y_pred_old = y_pred_new
            
            # c. 计算并记录当前状态的性能
            current_mse = mean_squared_error(self.target, y_pred_new)
            self.history_['train_mse'].append(current_mse)
            self.history_['component_change'].append(prediction_change) # 复用旧的history字段记录新指标
            self.history_['bandwidths'].append(iter_bandwidths)
            self.history_['predictions'].append(iter_predictions)
            self.history_['component_mse'].append(iter_component_mse)
            self.history_['component_time'].append(iter_component_time)
            print(f"\nCycle {m+1} End: Train MSE={current_mse:.6f}, Bandwidths: {iter_bandwidths}")

            # Check for convergence
            # --- MODIFIED: Check and store the best state ---
            if current_mse < self.best_mse_:
                self.best_mse_ = current_mse
                self.best_iteration_ = m + 1
                self.best_component_predictions_ = self.component_predictions_.copy()
                self.best_optimal_bandwidths_ = list(self.optimal_bandwidths_)
                self.best_fitted_learners_ = list(self.fitted_learners_)
                self.best_detailed_predictions_ = iter_detailed_preds.copy()
                print(f"\nCycle {m+1} End: New best MSE={current_mse:.6f}, Bandwidths: {iter_bandwidths}")
            else:
                print(f"\nCycle {m+1} End: Train MSE={current_mse:.6f}, Bandwidths: {iter_bandwidths}")

            if prediction_change < self.score_tol_ or m > self.max_iterations:
                print(f"\nOverall prediction convergence met: Change ({prediction_change:.8f}) < tolerance (1e-5). Stopping.")
                break

        
        # --- MODIFIED: Restore the best state at the end of fitting ---
        print(f"\n--- GAM-MGWX Fitting Finished ---")
        print(f"Restoring model state from best iteration: {self.best_iteration_}")
        self.component_predictions_ = self.best_component_predictions_
        self.optimal_bandwidths_ = self.best_optimal_bandwidths_
        self.fitted_learners_ = self.best_fitted_learners_
        
        print(f"Final Model Choice: {self.selected_learner_types_}")
        print(f"Final Best MSE: {self.best_mse_:.6f} (at cycle {self.best_iteration_})")
        print(f"Final Optimal Bandwidths: {self.optimal_bandwidths_}")
        
        
    def get_training_log(self):
        """
        Export training history.

        Returns
        -------
        pd.DataFrame
            Contains:
            - Iteration
            - Feature
            - Bandwidth (k)
            - Component MSE
            - Training time
            - Overall model MSE

        Useful for:
        - Debugging
        - Visualization
        - Paper figures
        """
        records = []
        n_iters = len(self.history_['bandwidths'])
        
        for m in range(n_iters):
            current_overall_mse = self.history_['train_mse'][m + 1]
            for i in range(self.n_features):
                records.append({
                    'Iteration': m + 1,
                    'Feature': f'x{i+1}',
                    'Bandwidth_K': self.history_['bandwidths'][m][i],
                    'Component_MSE': self.history_['component_mse'][m][i],
                    'Fit_Time_Sec': self.history_['component_time'][m][i],
                    'Overall_Train_MSE': current_overall_mse 
                })
        return pd.DataFrame(records)

    def get_best_local_detailed_predictions(self):
        """
        输出最佳迭代轮次下，所有特征、所有局部子模型产生的大量原始预测值。
        行数将等于 n_features * (n_clusters 或 n_samples) * bandwidth
        """
        if not self.best_detailed_predictions_:
            return pd.DataFrame()
            
        combined_list =[]
        for feature_name, df in self.best_detailed_predictions_.items():
            df_copy = df.copy()
            df_copy['feature_name'] = feature_name # 添加一列标识当前是哪个特征的贡献
            combined_list.append(df_copy)
            
        # 拼接在一起，方便一次性保存
        return pd.concat(combined_list, axis=0, ignore_index=True)

    def predict(self, new_data, new_locations):
        """
        Predicts target values for new data using the final fitted component functions f_i.
        """
        if new_data.shape[1] != self.n_features:
            raise ValueError(f"Input data has {new_data.shape[1]} features, but model was trained on {self.n_features}.")
        if len(new_locations) != new_data.shape[0]:
             raise ValueError("Number of new locations must match number of new data samples.")
        if not hasattr(self, 'fitted_learners_') or self.fitted_learners_[0] is None: # Check if fitted
             raise RuntimeError("The model has not been fitted yet or fitting failed. Call fit() first.")

        # Start with the intercept
        y_pred = np.full(new_data.shape[0], self.intercept_)

        # print("Predicting using final component learners (backfitting)...")
        for i in range(self.n_features):
            final_learner_instance = self.fitted_learners_[i] # The learner representing f_i

            if final_learner_instance is None:
                print(f"Warning: No final learner stored for feature {i}. Assuming zero contribution.")
                continue 

            X_feature_new = new_data[:, i].reshape(-1, 1)
            learner_type = self.selected_learner_types_[i]
            component_pred_i = np.zeros(new_data.shape[0]) # Default to zero

            try:
                geo_models_to_use = final_learner_instance # Models from last fit
                optimal_k = self.optimal_bandwidths_[i]
                if learner_type == 'GeoXGB':

                    if geo_models_to_use and optimal_k > 0:

                        predictor = GeoXGBPredictor(geo_models_to_use, new_locations, optimal_k)
                        component_pred_i = predictor.predict(X_feature_new, new_locations)
                else:
                    predictor = GeoXGBPredictor(geo_models_to_use, new_locations, optimal_k)
                    component_pred_i = predictor.predict(X_feature_new, new_locations)

                y_pred += component_pred_i

            except Exception as e:
                print(f"Error predicting component for feature {i} on new data: {e}. Skipping contribution.")

        return y_pred

    def get_fitting_history(self):
        """
        Returns the history of bandwidths and component predictions for each iteration.

        Returns:
            dict: A dictionary containing 'bandwidths' and 'predictions'.
                  history['bandwidths'][iteration][feature] = k_value
                  history['predictions'][iteration] = array of shape (n_samples, n_features)
        """
        return {
            'bandwidths': self.history_['bandwidths'],
            'predictions': self.history_['predictions']
        }

    def get_final_components(self):
        """
        Returns the final fitted component predictions after the last iteration.
        """
        return pd.DataFrame(self.component_predictions_, columns=[f'f_{i}' for i in range(self.n_features)])
        
    def get_final_predictions(self):
        return self.intercept_ + np.sum(self.component_predictions_, axis=1)
        
