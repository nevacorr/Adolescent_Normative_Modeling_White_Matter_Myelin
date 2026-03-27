def evaluate_splines_cv(X_train,y_train, roi_ids, agemin,agemax,cv_dir,spline_orders=[1, 2],spline_knots_list=[2, 3],n_folds=5):
    import os
    import shutil
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from pcntoolkit.normative import estimate
    from Utility_Functions import create_design_matrix, makenewdir

    makenewdir(cv_dir)

    allfoldsdf = pd.DataFrame()  # store all folds’ metrics

    for roi in roi_ids:
        print(f"Evaluating splines for ROI: {roi}")

        for order in spline_orders:
            for knots in spline_knots_list:

                # Combine numeric age and sex into a string label for stratification
                strat_feature = X_train['age'].astype(str) + "_" + X_train['sex'].astype(str)

                # For stratification, here we just shuffle
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

                fold_idx = 0
                for train_idx, val_idx in kf.split(X_train, strat_feature):  # stratify by sex for example
                    X_tr = X_train.iloc[train_idx].reset_index(drop=True)
                    X_val = X_train.iloc[val_idx].reset_index(drop=True)
                    y_tr = y_train.iloc[train_idx][[roi]].reset_index(drop=True)
                    y_val = y_train.iloc[val_idx][[roi]].reset_index(drop=True)

                    # drop the age column from the train and val data set because we want to use agedays and sex as predictors
                    X_tr.drop(columns=['age'], inplace=True)
                    X_val.drop(columns=['age'], inplace=True)

                    order_dir = os.path.join(cv_dir, f"order{order}_knots{knots}_findex{fold_idx}")
                    makenewdir(order_dir)
                    fold_dir = os.path.join(cv_dir, order_dir, roi)
                    makenewdir(fold_dir)

                    X_tr.to_csv(f"{fold_dir}/cov_tr.txt", sep='\t', header=False, index=False)
                    X_val.to_csv(f"{fold_dir}/cov_te.txt", sep='\t', header=False, index=False)
                    y_tr.to_csv(f"{fold_dir}/resp_tr.txt", sep='\t', header=False, index=False)
                    y_val.to_csv(f"{fold_dir}/resp_te.txt", sep='\t', header=False, index=False)

                    create_design_matrix('train', agemin, agemax, order, knots, [roi], order_dir)
                    create_design_matrix('test', agemin, agemax, order, knots, [roi], order_dir)

                    cov_file_tr = os.path.join(fold_dir, 'cov_bspline_tr.txt')
                    cov_file_te = os.path.join(fold_dir, 'cov_bspline_te.txt')
                    resp_file_tr = os.path.join(fold_dir, 'resp_tr.txt')
                    resp_file_te = os.path.join(fold_dir, 'resp_te.txt')

                    try:
                        _, _, _, _, metrics = estimate(
                            cov_file_tr,
                            resp_file_tr,
                            testresp=resp_file_te,
                            testcov=cov_file_te,
                            alg='blr',
                            optimizer='powell',
                            savemodel=False,
                            saveoutput=False,
                            standardize=False
                        )

                        metrics_clean = {k: float(v) if isinstance(v, (np.ndarray, list)) else v for k, v in
                                         metrics.items()}

                        metrics_clean['roi'] = roi
                        metrics_clean['spline_order'] = order
                        metrics_clean['spline_knots'] = knots
                        metrics_clean['fold'] = fold_idx + 1

                        allfoldsdf = pd.concat([allfoldsdf, pd.DataFrame([metrics_clean])], ignore_index=True)

                    except Exception as e:
                        print(f"Error in ROI {roi}, order {order}, knots {knots}, fold {fold_idx}: {e}")

                    fold_idx += 1

    # Now compute group mean safely, ignoring NaNs
    mean_perf_metrics = allfoldsdf.groupby(
        ['roi', 'spline_order', 'spline_knots'],
        as_index=False
    ).mean()

    # Save full metrics and mean
    allfoldsdf.to_csv(os.path.join(cv_dir, 'all_folds_metrics.csv'), index=False)
    mean_perf_metrics.to_csv(os.path.join(cv_dir, 'mean_metrics_per_spline.csv'), index=False)

    return mean_perf_metrics