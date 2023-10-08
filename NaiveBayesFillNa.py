import pandas as pd
class NaiveBayesFillna():
    def __init__(self):
        self.__p_dict = {}
        self.__cond_p_dict = {}

    def fit(self, X : pd.DataFrame):
        for col in X.columns:
            unique_values = X[col].dropna().unique()
            self.__p_dict.update({col : {}})
            for val in unique_values:
                p = X[col].value_counts(normalize=True)[val]
                self.__p_dict[col].update({val : p})

        for col in X.columns:
            unique_values = X[col].dropna().unique()
            self.__cond_p_dict.update({col : {}})
            for val in unique_values:
                self.__cond_p_dict[col].update({val : {}})
                for other_col in X.columns.drop(col):
                    self.__cond_p_dict[col][val].update({other_col : {}})
                    other_unique_values = X[other_col].dropna().unique()
                    for other_val in other_unique_values:
                        sub_df = X[X[other_col] == other_val][col]
                        if val in sub_df.values:
                            cond_p  = sub_df.value_counts(normalize=True)[val]
                        else:
                            cond_p = 0
                        self.__cond_p_dict[col][val][other_col].update({other_val : cond_p})
        return self.__p_dict, self.__cond_p_dict

    def __compute(self,x : pd.Series, missing_col_name):
        unique_values = list(self.__p_dict[missing_col_name].keys())
        max = -1
        result_label = None
        for val in unique_values:
            p = self.__p_dict[missing_col_name][val]
            for col in x.index:
                p *= self.__cond_p_dict[col][x[col]][missing_col_name][val]
            if p > max:
                max = p
                result_label = val
        return result_label

    def transform(self,df):
        X = df.copy()
        for index, row in X.iterrows():
            if not row.isna().any():
                continue
            for col_name, val in row.items():
                if pd.isna(val):
                    sub_series = row.dropna()
                    res = self.__compute(sub_series,col_name)
                    row[col_name] = res
                    X.loc[index,col_name] = res
        return X