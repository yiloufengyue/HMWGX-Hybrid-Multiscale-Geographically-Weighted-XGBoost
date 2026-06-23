import numpy as np
import os
from HMGWX_1 import GeoXGBTrainer, GAM_MGWX, GeoXGBInterpreter
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import math
import time
from tqdm import tqdm
import random

# =====================================================================
# 1. dataset generation function
# =====================================================================
def generate_dataset(seed, size=50):
    np.random.seed(seed)
    
    X1 = np.random.uniform(-1.5, 1.5, size*size)
    X2 = np.random.uniform(-1.5, 1.5, size*size)
    X3 = np.random.uniform(-1.5, 1.5, size*size)
    X4 = np.random.uniform(-1.5, 1.5, size*size)
    X5 = np.random.uniform(-1.5, 1.5, size*size)
    X6 = np.random.uniform(-1.5, 1.5, size*size)

    X = np.vstack([X1, X2, X3, X4, X5, X6]).T

    b6 = np.zeros((size, size))
    b3 = np.zeros((size, size))
    b4 = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            b6[i][j] = 2
            b4[i][j] = math.cos(math.pi*math.exp(i/50))*math.sin(math.pi*math.exp(j/50))*3
            b3[i][j] = 3*((i+j)/99)

    u = np.array([np.linspace(0, size-1, num=size)]*size).reshape(-1)
    v = np.array([np.linspace(0, size-1, num=size)]*size).T.reshape(-1)
    coords = np.array(list(zip(u, v)))

    f1 = np.zeros_like(X1)
    center_u = size / 2
    center_v = size / 2

    mask_top_left = (u < center_u) & (v < center_v)
    mask_top_right = (u >= center_u) & (v < center_v)
    mask_bottom_left = (u < center_u) & (v >= center_v)
    mask_bottom_right = (u >= center_u) & (v >= center_v)

    f1[mask_top_left] = np.tanh(X1[mask_top_left]*np.pi)
    f1[mask_bottom_right] = np.sin(X1[mask_bottom_right] * np.pi)
    f1[mask_top_right] = np.abs(X1[mask_top_right]) * X1[mask_top_right]
    f1[mask_bottom_left] = X1[mask_bottom_left]**3

    f2 = np.zeros_like(X2)
    mask_left = u < (size / 2)
    mask_right = ~mask_left
    f2[mask_left] = np.abs(X2[mask_left])*X2[mask_left]
    f2[mask_right] = np.tanh(X2[mask_right]*np.pi)

    f3 = b3.reshape(-1)*X3.reshape(-1)
    f4 = b4.reshape(-1)*X4.reshape(-1)
    f5 = X5**3
    f6 = 2*X6

    fs = np.vstack([f1, f2, f3, f4, f5, f6]).T
    err = np.random.uniform(-1.5, 1.5, size*size)
    y = (f1 + f2 + f3 + f4 + f5 + f6 + err).reshape(-1, 1)
    
    return X, y, coords, fs


# =====================================================================
# 2. extract prediction and shap from GWXGB/GXGB model
# =====================================================================

def get_full_sample_mode_details(trainer, intepreter):
 
    if not trainer.use_full_sample:
        local_shap_df, global_shap_df  = intepreter.calculate_training_shap()
        global_pred_df, global_pred_df = trainer.prediction()
        global_results_df = pd.concat([global_pred_df, global_shap_df], axis=1)
        return global_results_df, local_shap_df    

    else:
        local_shap_df, global_shap_df  = intepreter.calculate_training_shap()
        local_pred_df, global_pred_df = trainer.prediction()

        global_results_df = pd.concat([global_pred_df, global_shap_df], axis=1)
        local_results_df = pd.concat([local_shap_df, local_pred_df], axis=1)
            
        return global_results_df, local_results_df


# =====================================================================
# 3. define 95% CI Bootstrap 
# =====================================================================
def calculate_gam_ci(base_model, X, y, coords, save_dir, prefix, n_bootstraps=100, n_jobs=16):
    """
    基于残差自举法 (Residual Bootstrapping) 为 GAM 模型计算 95% 置信区间。
    由于重训代价大，我们冻结由 base_model 选出的带宽 (k) 和模型结构 (GeoWXGB/GeoXGB)。
    """
    print(f"\n[Bootstrap] 开始进行 {n_bootstraps} 次自举法计算 95% CI...")
    y_pred = base_model.get_final_predictions()
    residuals = y.ravel() - y_pred.ravel()
    n_samples = len(y)

    fixed_learners = base_model.selected_learner_types_
    fixed_k_ranges =[(int(k), int(k), 10) for k in base_model.optimal_bandwidths_]

    global_preds_list = []
    local_preds_list =[]

    for b in tqdm(range(n_bootstraps), desc="Bootstrapping CI Iterations"):
  
        resample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_pred.ravel() + residuals[resample_idx]

        boot_model = GAM_MGWX(
            data=X, target=y_boot.reshape(-1, 1), locations=coords,
            selected_learner_types=fixed_learners,
            k_range=fixed_k_ranges,
            max_iterations=20,             
            use_damping_factor=True, damping_factor=1,
            enable_bw_shrinking=False,   
            n_jobs=n_jobs,
            use_full_sample=base_model.use_full_sample,
            search_strategy='golden'
        )
        boot_model.fit()

        global_preds_list.append(boot_model.get_final_components().values)
        local_df = boot_model.get_best_local_detailed_predictions()
        local_preds_list.append(local_df['prediction'].values)



    global_stack = np.stack(global_preds_list, axis=-1) # shape: (2500, 6, n_bootstraps)
    global_lower = np.percentile(global_stack, 2.5, axis=-1)
    global_upper = np.percentile(global_stack, 97.5, axis=-1)

    base_global_df = base_model.get_final_components().rename(columns={f'f_{i}': f'f{i+1}_pred' for i in range(X.shape[1])})
    for i in range(X.shape[1]):
        base_global_df[f'f{i+1}_lower_95'] = global_lower[:, i]
        base_global_df[f'f{i+1}_upper_95'] = global_upper[:, i]
    
    global_save_path = os.path.join(save_dir, f"{prefix}_global_CI.csv")
    base_global_df.to_csv(global_save_path, index=False)

    base_local_df = base_model.get_best_local_detailed_predictions()
    local_stack = np.stack(local_preds_list, axis=-1) # shape: (n_rows, n_bootstraps)
    base_local_df['lower_95'] = np.percentile(local_stack, 2.5, axis=-1)
    base_local_df['upper_95'] = np.percentile(local_stack, 97.5, axis=-1)
    
    local_save_path = os.path.join(save_dir, f"{prefix}_local_CI.csv")
    base_local_df.to_csv(local_save_path, index=False)


# =====================================================================
# 主执行入口
# =====================================================================
if __name__ == "__main__":
    
    save_dir = r'C:\\' 
    ## Replace with your results storage path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    seed = random.randint(0, 123456789) 
    X, y, coords, fs = generate_dataset(seed=seed)
    k_range_for_all_features = (125, 2500, 125) 
    k_ranges = [k_range_for_all_features] * X.shape[1]

    feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    n_jobs = 16 
    # -----------------------------------------------------------------
    # Task 1: Fitting GWXGB/GXGB models
    # -----------------------------------------------------------------

    configs =[
        {'use_full_sample': True, 'n_clusters': len(coords), 'use_spatial_weights': False}, ## GXGB using full sample 
        {'use_full_sample': True, 'n_clusters': len(coords), 'use_spatial_weights': True},  ## GWXGB using full sample 
        {'use_full_sample': False, 'n_clusters': 250, 'use_spatial_weights': False},  ## GXGB using 250 clusters
        {'use_full_sample': False, 'n_clusters': 250, 'use_spatial_weights': True},  ## GWXGB using 250 clusters
    ]

    file_name = ['gx_full', 'gwx_full', 'gx_cluster', 'gwx_cluster']

    for config_idx, config in enumerate(configs):

        print(f"\n=======================================================")
        print(f"Testing Config: Full Sample: {config['use_full_sample']} | "
              f"Clusters: {config['n_clusters']} | "
              f"Spatial Weights: {config['use_spatial_weights']}")
        print(f"=======================================================")

        start_time = time.time()

        trainer = GeoXGBTrainer(
            data=X,
            target=y,
            locations=coords,
            n_jobs=n_jobs,
            n_clusters=config['n_clusters'],
            use_full_sample=config['use_full_sample'],
            use_spatial_weights=config['use_spatial_weights']
            )
        
        optimal_k_golden = trainer.search_optimal_k_golden(k_range=(25, 2500), )

        trainer.fit(lightweight_mode=False)

        end_time = time.time()
        g_duration = end_time - start_time
        print(f"Running time: {end_time - start_time:.2f} 秒")
        print("-" * 50)

        print(f'fixed duration: {g_duration}')
        print(f'golden duration: {g_duration}')

        gwxgb_interpreter = GeoXGBInterpreter(trainer, feature_names)
        pred_shap, detailed_pred_shap = get_full_sample_mode_details(trainer, gwxgb_interpreter)

        analysis_df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(X.shape[1])])
        true_f_df = pd.DataFrame(fs, columns=[f'f{i+1}_true' for i in range(fs.shape[1])])
        analysis_df['y_true'] = y.flatten()
        analysis_df['bw'] = optimal_k_golden

        # combine data
        analysis_df = pd.concat([analysis_df, true_f_df, pred_shap], axis=1)
        detailed_pred_shap.to_csv(save_dir + '\\' + f'{file_name[config_idx]}_analysis_results_local.csv', index=False)
        analysis_df.to_csv(save_dir + '\\' + f'{file_name[config_idx]}_analysis_results.csv', index=False)

    # -----------------------------------------------------------------
    # Task 2: Fitting HMGWX, MGWX, and MGX models 
    # -----------------------------------------------------------------
        
    experiments = {
        'MGWX': {
            'learner': ['GeoWXGB'] * X.shape[1],
            'use_full_sample': False,
            'enable_bw_shrinking': True,
            'search_strategy': 'golden'
        },
        'MGX': {
            'learner': ['GeoXGB'] * X.shape[1],
            'use_full_sample': False,
            'enable_bw_shrinking': True,
            'search_strategy': 'golden'
        },
        'HMGWX': {
            'learner': None, #
            'use_full_sample': False,
            'enable_bw_shrinking': True,
            'search_strategy': 'golden'
        }
    }

    for model_name, config in experiments.items():
        start_time = time.time()
            
        model = GAM_MGWX(
            data=X, target=y, locations=coords,
            selected_learner_types=config['learner'], 
            use_damping_factor=True, damping_factor=1,
            enable_bw_shrinking=config['enable_bw_shrinking'],
            k_range=k_ranges,
            max_iterations=20,
            n_jobs=n_jobs, 
            use_full_sample=config['use_full_sample'],
            search_strategy=config['search_strategy']
        )
        model.fit()
            
        print(f"{model_name} (Seed: {seed}) 训练完毕，耗时: {time.time() - start_time:.2f} 秒")
            
        # save log
        model.get_training_log().to_csv(os.path.join(save_dir, f'task2_{model_name}_log.csv'), index=False)
            
        # save global results
        analysis_df_s = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(X.shape[1])])
        true_f_df_s = pd.DataFrame(fs, columns=[f'f{i+1}_true' for i in range(fs.shape[1])])
        predicted_f_df_s = model.get_final_components().rename(columns={f'f_{i}': f'f{i+1}_pred' for i in range(X.shape[1])})
            
        analysis_df_s['y_true'] = y.flatten()
        analysis_df_s['y_hat'] = model.get_final_predictions()
        analysis_df_s = pd.concat([analysis_df_s, true_f_df_s, predicted_f_df_s], axis=1)
        analysis_df_s.to_csv(os.path.join(save_dir, f'task2_{model_name}_global_results.csv'), index=False)
            
        # save local results
        local_df_s = model.get_best_local_detailed_predictions()
        local_df_s.to_csv(os.path.join(save_dir, f'task2_{model_name}_local_predictions.csv'), index=False)


        calculate_gam_ci(model, X, y, coords, save_dir, prefix='task2_{model_name}', n_bootstraps=100, n_jobs=16)