import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', -1)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost
import shap
shap.initjs()

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def preprocessing(data):
    target_name = data.columns[-1]
    X = data.drop([target_name], axis=1).fillna(data.mean())
    X_cols = X.columns
    X_list = X.values.tolist()
    scaler = MinMaxScaler()
    scaler.fit(X_list)
    X_scaled_list = scaler.transform(X_list)

    X = pd.DataFrame(X_scaled_list, columns=X_cols)
    y = data[target_name]
    return X, y


def model_training(X, y, X_train, y_train):
    xgb_train = xgboost.DMatrix(X_train, label=y_train)
    xgb_full = xgboost.DMatrix(X, label=y)

    model = xgboost.train({"learning_rate": 0.01}, xgb_train, 100)
    y_pred = model.predict(xgb_full)
    y_pred = [1 if x > 0.5 else 0 for x in y_pred]
    print("accuracy score (full data set):", accuracy_score(y, y_pred))

    data_conc = pd.concat([X, y], axis=1)
    idxs = [i for i in range(len(y)) if y[i] != y_pred[i]]
    data_conc.drop(data_conc.index[idxs], inplace=True)
    df_conc = data_conc.reset_index(drop=True)
    target_name = df_conc.columns[-1]

    y = df_conc[target_name]
    X_idxs = df_conc.drop([target_name], axis=1)
    X = X_idxs
    return model, X, y


def SHAP(model, X, y):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X)

    # concatenating SHAP values and target column
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    df = pd.concat([shap_df, y], axis=1)
    return df


def SHAP_discretization(df):
    # list of X and y values
    list_of_lists = df.values.tolist()
    list_y = [x[-1] for x in list_of_lists]
    list_X = [x[:-1] for x in list_of_lists]

    # list of absolute SHAP values > 0
    threshold_cut = 0
    list_abs_final = []
    for x in list_X:
        for xx in x:
            if xx != threshold_cut:
                list_abs_final.append(np.abs(xx))
    threshold_low = np.percentile(list_abs_final, 25)
    threshold_mid = np.percentile(list_abs_final, 75)

    # SHAP values discretization
    X_final = list(X.columns)
    y_final = []
    for i in range(X.shape[0]):
        y_temp = [str(int(list_y[i]))]
        y1 = [x for x in list_X[i]]
        y1_abs = [abs(x) for x in y1]
        for ii, y in enumerate(y1_abs):
            if y > threshold_cut:
                if y <= threshold_low:
                    y_temp.append("low" + str(X_final[ii]))
                elif y <= threshold_mid:
                    y_temp.append("mid" + str(X_final[ii]))
                else:
                    y_temp.append("high" + str(X_final[ii]))
        y_final.append(y_temp)
    df_final = y_final
    return df_final


def association_rules_generator(df_final, target_class):
    te = TransactionEncoder()
    te_ary = te.fit(df_final).transform(df_final)
    df_apriori = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_apriori, min_support=0.05, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.1)

    df1 = rules.drop(columns=["leverage", "conviction", "antecedent support", "consequent support"])[
        rules['antecedents'].apply(lambda x: not any(y in x for y in [str(x) for x in target_class])) &
        rules['antecedents'].apply(lambda x: not any("low" in y for y in list(x))) &
        rules['antecedents'].apply(lambda x: 1 < len(x) <= 3) &
        rules['consequents'].apply(lambda x: x in [frozenset(str(x)) for x in target_class]) &
        rules['lift'].apply(lambda x: x > 1)
        ]
    df1 = df1.sort_values(by=["lift", "support"], ascending=False).reset_index(drop=True)
    return df1


if __name__=='__main__':
    file = "grapevines_template.csv"
    data = pd.read_csv(file)
    X, y = preprocessing(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    model, X, y = model_training(X, y, X_train, y_train)
    target_class = list(set(list(y)))

    shap_df = SHAP(model, X, y)
    df_final = SHAP_discretization(shap_df)
    results = association_rules_generator(df_final, target_class)
    print(results)
