import numpy as np
import pandas as pd


class MultiprocFeatures():

    @staticmethod
    def num_of_units_since_last_payment(data: pd.DataFrame, units: str, **kwargs) -> pd.DataFrame:
        """
        Compute number of days/hours since last payment;
        :param data:  pd.DataFrame, with columns "ACCNO", "PaymentDateTime"
        :param units: str, units of time delta {"h", "D"}, where "h" - hours, "D" - days;
        :return:      pd.DataFrame,
        """
        data_chunck = data.groupby("ACCNO").apply(
            lambda x: (x.PaymentDateTime - x.PaymentDateTime.shift(1)).astype(f'timedelta64[{units}]')
        ).reset_index(name=f"NumOf{units.upper()}SinceLastPayment")
        data_chunck[f"NumOf{units.upper()}SinceLastPayment"].fillna(0, inplace=True)
        data_chunck[f"NumOf{units.upper()}SinceLastPayment"] = data_chunck[
            f"NumOf{units.upper()}SinceLastPayment"
        ].astype(int)
        if "level_1" in data_chunck.columns:
            del data_chunck["level_1"]
        if "index" in data_chunck.columns:
            del data_chunck["index"]
        return data_chunck
    
    @staticmethod
    def num_of_units_since_last_payment_by_k_days(data: pd.DataFrame, units: str, k_days: int, **kwargs) -> pd.DataFrame:
        """
        Compute number of days/hours since last payment;
        :param data:  pd.DataFrame, with columns "ACCNO", "PaymentDateTime"
        :param units: str, units of time delta {"h", "D"}, where "h" - hours, "D" - days;
        :return:      pd.DataFrame,
        """
        feature_name = f"NumOf{units.upper()}SinceLastPaymentBy{k_days}Days"
        data_cp = data[["ACCNO", "PaymentDateTime", "SaleDate"]].copy().reset_index(drop=True)
        data_cp["PaymentDateTime"] = pd.to_datetime(data_cp["PaymentDateTime"])
        data_cp["SaleDate"] = pd.to_datetime(data_cp["SaleDate"])
        
        data_chunck = data.groupby("ACCNO").apply(
            lambda x: (x.PaymentDateTime - x.PaymentDateTime.shift(1)).astype(f'timedelta64[{units}]')
        ).reset_index(name=feature_name)
        
        data_chunck = pd.concat((data[["SaleDate", "PaymentDateTime"]], data_chunck), axis=1)
        data_chunck["PaymentDateTime"] = pd.to_datetime(data_chunck["PaymentDateTime"])
        data_chunck["SaleDate"] = pd.to_datetime(data_chunck["SaleDate"])
        
        mask = data_chunck[feature_name].isnull()
        data_chunck.loc[
            mask & ((data_chunck.PaymentDateTime - data_chunck.SaleDate).dt.days < k_days), 
            feature_name
        ] = -1
        data_chunck.loc[
            mask & ((data_chunck.PaymentDateTime - data_chunck.SaleDate).dt.days >= k_days), 
            feature_name
        ] = -100
        data_chunck.loc[
            ~mask & ((data_chunck.PaymentDateTime - data_chunck.SaleDate).dt.days >= k_days), 
            feature_name
        ] = pd.Series(pd.Timedelta(days=k_days + 1)).astype(f'timedelta64[{units}]').values[0]
        
        n_nulls = data_chunck[feature_name].isnull().sum()
        if n_nulls > 0:
            raise ValueError(f"# rows with nulls in `{feature_name}`: {n_nulls}")

        data_chunck[feature_name] = data_chunck[feature_name].astype(int)
        if "level_1" in data_chunck.columns:
            del data_chunck["level_1"]
        if "index" in data_chunck.columns:
            del data_chunck["index"]
        return data_chunck
    
    @staticmethod
    def cum_sum(data: pd.DataFrame, cols_to_group_by: list, col_to_cumsum: str, col_alias: int, **kwargs) -> pd.DataFrame:
        returnc_cols = cols_to_group_by + [col_alias]
        cols_to_cp = list(set(cols_to_group_by).union({'ACCNO', 'PaymentDateTime', 'TotalAmount'})) + [col_to_cumsum]
        data_cp = data[cols_to_cp].copy().reset_index(drop=True)
        # data_cp.sort_values(['ACCNO', 'PaymentDateTime', 'TotalAmount'], inplace=True, ignore_index=True)
        
        data_cp[col_alias] = data_cp.groupby(cols_to_group_by)[col_to_cumsum].cumsum()

        return data_cp[returnc_cols]
    
    @staticmethod
    def cum_sum_by_k_days(data: pd.DataFrame, col_to_cumsum: str, col_alias: int, k_days: int, **kwargs) -> pd.DataFrame:
        col_alias += f"By{k_days}Days"
        data_cp = data[['ACCNO', 'PaymentDateTime', 'TotalAmount', 'OriginFileRecordID'] + [col_to_cumsum]].copy().reset_index(drop=True)
        # data_cp.sort_values(['ACCNO', 'PaymentDateTime', 'TotalAmount'], inplace=True, ignore_index=True)
        data_cp["PaymentDateTime"] = pd.to_datetime(data_cp["PaymentDateTime"])
        
        del data_cp['TotalAmount']
        data_cp = pd.concat(
            (
                data_cp[["OriginFileRecordID"]],
                data_cp.groupby('ACCNO', group_keys=False)[["PaymentDateTime", col_to_cumsum]]\
                    .rolling(f"{k_days}D", on="PaymentDateTime")\
                    .sum()\
                    .reset_index().rename(columns={col_to_cumsum: col_alias})
            ), axis=1
        )
        data_cp[col_alias].fillna(0, inplace=True)
        if "level_1" in data_cp.columns:
            del data_cp['level_1']
        return data_cp
    
#     def cum_amount_100g(data: pd.DataFrame, col_alias: int, n_days: int) -> pd.DataFrame:
#         data_cp = data[["ACCNO", 'PaymentDateTime', "TotalAmount", "Amount"]].copy()
        
#         data_cp.sort_values(['ACCNO', 'PaymentDateTime', 'TotalAmount'], inplace=True, ignore_index=True)
        
#         data_chunck = data_cp[~data_cp.SaleDate.isnull()]\
#             .groupby(["ACCNO"], group_keys=True)\
#             .apply(
#                 lambda x: x((pd.to_datetime(x["PaymentDateTime"]) < pd.to_datetime(x["PaymentDateTime"]) - pd.to_timedelta(30, unit="D")).astype(int).cumsum() > 0).astype(int)
#                 )\
#             .reset_index(name=col_alias)
        
#         if "level4_" in data_chunck_1.columns:
#             del data_chunck_1["level_4"]
#         if "level_4" in data_chunck.columns:
#             del data_chunck["level_4"]
    
#         data_chunck[col_alias] = data_chunck.groupby("ACCNO")[col_alias].cumsum() - 1
#         data_chunck = pd.concat((data_chunck, data_chunck_1)).reset_index(drop=True)
#         data_chunck.loc[data_chunck[col_alias] < 0, col_alias] = 0
        
#         # data_chunck.to_csv(f"temp_{np.random.randint(0, 10, 1)[0]}.csv")
#         print("Not duplicsted shape of `data_chunck`: ", data_chunck[~data_chunck[["ACCNO", "PaymentDateTime", 'OriginFileRecordID', 'PaymentReference']].duplicated()].shape)
#         print("Duplicates' shape of `data_cp`: ", data_cp[~data_cp[["ACCNO", "PaymentDateTime", 'OriginFileRecordID', 'PaymentReference']].duplicated()].shape)
#         print("Duplicates in `data_chunck`:", data_chunck[data_chunck[["ACCNO", "PaymentDateTime", 'OriginFileRecordID', 'PaymentReference']].duplicated()])
        
#         assert data_chunck.shape[0] == data_cp.shape[0], f"Expected shape of a dataframe with the feature: {data_cp.shape[0]}; Got: {data_chunck.shape[0]}"
#         return data_chunck[["ACCNO", "PaymentDateTime", "OriginFileRecordID", "PaymentReference", col_alias]]
    
    @staticmethod
    def num_of_units_between_n_last_payments(data: pd.DataFrame,
                                             n_payments: int,
                                             date_units: str, **kwargs) -> pd.DataFrame:
        feature_name = f"NumOf{date_units.upper()}Between{n_payments}LastPayments"
        # data.sort_values(["ACCNO", "PaymentDateTime", "TotalAmount"], inplace=True, ignore_index=True)
        data['PaymentDateTimePrev'] = data.groupby('ACCNO')['PaymentDateTime'].shift(n_payments)
        data[feature_name] = (data['PaymentDateTime'] - data['PaymentDateTimePrev']).astype(
            f'timedelta64[{date_units}]'
        )
        data[feature_name].fillna(0, inplace=True)
        return data[["ACCNO", feature_name]]
    
    @staticmethod
    def num_of_units_between_n_last_payments_by_k_days(data: pd.DataFrame,
                                                       n_payments: int,
                                                       date_units: str, 
                                                       k_days: int, **kwargs) -> pd.DataFrame:
        if (n_payments < 1) | (k_days < 1):
            raise ValueError(f"`{n_payments}` and {k_days} should be >= 1")
        feature_name = f"NumOf{date_units.upper()}Between{n_payments}LastPaymentsBy{k_days}Days"
        data_cp = data[["ACCNO", "PaymentDateTime", "TotalAmount", "SaleDate"]].copy().reset_index(drop=True)
        
        # data_cp.sort_values(["ACCNO", "PaymentDateTime", "TotalAmount"], inplace=True, ignore_index=True)
        # get n-th previous payment
        data_cp['PaymentDateTimePrev'] = data_cp.groupby('ACCNO')['PaymentDateTime'].shift(n_payments)
        
        not_nan_mask = ~data_cp['PaymentDateTimePrev'].isnull()
        
        data_cp['PaymentDateTime'] = pd.to_datetime(data_cp['PaymentDateTime'])
        data_cp['SaleDate'] = pd.to_datetime(data_cp['SaleDate'])
        
        data_cp["PaymentDateTime_k_days"] = data_cp['PaymentDateTime'] - pd.offsets.Day(k_days)
        
        # if an n-th previous payment was longer than k_days ago (and a user is registered longer than k_days ago) 
        # set k_days+1 date as a date of the n-th payment
        data_cp.loc[
            (not_nan_mask & (data_cp['PaymentDateTimePrev'] < data_cp["PaymentDateTime_k_days"])) | 
            (~not_nan_mask & ((data_cp["PaymentDateTime"] - data_cp["SaleDate"]).dt.days > k_days)), 
            "PaymentDateTimePrev"
        ] = data_cp.loc[
            (not_nan_mask & (data_cp['PaymentDateTimePrev'] < data_cp["PaymentDateTime_k_days"])) | 
            (~not_nan_mask & ((data_cp["PaymentDateTime"] - data_cp["SaleDate"]).dt.days > k_days)),
            "PaymentDateTime"
        ] - pd.offsets.Day(k_days + 1)
        
        # compute datetime units between n-th and current payment in case
        # -> PaymentDateTimePrev is given
        data_cp[feature_name] = None
        data_cp.loc[not_nan_mask, feature_name] = (data_cp.loc[not_nan_mask, 'PaymentDateTime'] - data_cp.loc[not_nan_mask, 'PaymentDateTimePrev']).astype(
            f'timedelta64[{date_units}]'
        )
        # ->  PaymentDateTimePrev is null, and a user has been using account less than k_days, it's set to 0
        data_cp.loc[
            data_cp[feature_name].isna() & ((data_cp.PaymentDateTime - data_cp.SaleDate).dt.days > k_days), feature_name
        ] = -100
        data_cp.loc[
            (~not_nan_mask & ((data_cp["PaymentDateTime"] - data_cp["SaleDate"]).dt.days <= k_days)), 
            feature_name
        ] = -1

        n_nulls = data_cp[feature_name].isnull().sum()
        if n_nulls > 0:
            raise ValueError(f"# of zeros in `{feature_name}`: {n_nulls}")
        if (data_cp[feature_name].max() > (k_days + 1)) & (date_units=="D"):
            raise ValueError(f"Max in `{feature_name}`: {data_cp[feature_name].max()} > {k_days + 1}")
        if (data_cp[feature_name].max() > ((k_days + 1)*24)) & (date_units=="h"):
            raise ValueError(f"Max in `{feature_name}`: {data_cp[feature_name].max()} > {(k_days + 1)*24}")
        if data_cp[feature_name].min() < -100:
            raise ValueError(f"Max in `{feature_name}`: {data_cp[feature_name].min()} < -100")
            
        return data_cp[["ACCNO", feature_name]]

    @staticmethod
    def compute_average_time_between_n_last_payments(data: pd.DataFrame,
                                                     n_payments: int,
                                                     date_units: str, **kwargs) -> pd.DataFrame:
        feature_name = f"Average{date_units.upper()}Between{n_payments}LastPayments"
        # data.sort_values(["ACCNO", "PaymentDateTime", "TotalAmount"], inplace=True, ignore_index=True)
        data['PaymentDateTimePrev'] = data.groupby('ACCNO')['PaymentDateTime'].shift(1)
        data['PaymentDateTimeDiff'] = (data['PaymentDateTime'] - data['PaymentDateTimePrev']).astype(
            f'timedelta64[{date_units}]'
        )
        data_chunk = data.groupby(["ACCNO"]).PaymentDateTimeDiff.transform(
            lambda x: x.rolling(n_payments).mean()
        ).reset_index(name=feature_name)
        data_chunk[feature_name].fillna(0, inplace=True)
        del data_chunk["index"]

        data_chunk = pd.concat((data_chunk, data[["ACCNO"]]), axis=1)
        return data_chunk
    
    @staticmethod
    def compute_average_time_between_n_last_payments_by_k_days(data: pd.DataFrame,
                                                               n_payments: int,
                                                               date_units: str,
                                                               k_days: int, date_from: str, **kwargs) -> pd.DataFrame:
        if (n_payments == 0) | (k_days == 0):
            raise ValueError("Parameters `n_payments` and `k_days` should be > 0.")
        feature_name = f"Average{date_units.upper()}Between{n_payments}LastPaymentsBy{k_days}Days"
        data_cp = data[["ACCNO", "PaymentDateTime", "TotalAmount", "SaleDate"]].copy().reset_index(drop=True)
        
        data_cp["PaymentDateTime"] = pd.to_datetime(data_cp["PaymentDateTime"])
        data_cp["SaleDate"] = pd.to_datetime(data_cp["SaleDate"])
        
        # data_cp.sort_values(["ACCNO", "PaymentDateTime", "TotalAmount"], inplace=True, ignore_index=True)
        data_cp['PaymentDateTimePrev'] = data_cp.groupby('ACCNO')['PaymentDateTime'].shift(1)
        data_cp['PaymentDateTimeDiff'] = (data_cp['PaymentDateTime'] - data_cp['PaymentDateTimePrev']).astype(
            f'timedelta64[{date_units}]'
        )
        accno_unique = data_cp.ACCNO.unique()
        res = []
        for accno in accno_unique:
            subsample = data_cp[data_cp.ACCNO == accno].reset_index(drop=True).copy()
            unique_dates = subsample.PaymentDateTime.unique()
            unique_dates_rel = unique_dates[unique_dates >= pd.to_datetime(date_from)]
            for date in unique_dates_rel:
                date_subsample = subsample[
                    (subsample.PaymentDateTime <= pd.to_datetime(date))
                ].copy().reset_index(drop=True)
                date_subsample["PaymentDateTime"] = pd.to_datetime(date_subsample["PaymentDateTime"])
                date_subsample =  date_subsample[
                    (pd.to_datetime(date) - date_subsample.PaymentDateTime).dt.days <= k_days
                ]
                date_subsample = date_subsample[
                    date_subsample.PaymentDateTimeDiff <= pd.Series(pd.Timedelta(days=k_days )).astype(f'timedelta64[{date_units}]').values[0]
                ]
                m = date_subsample.reset_index(drop=True).loc[:, "PaymentDateTimeDiff"].tail(n_payments).mean()
                res += [(accno, date, m)]

        data_chunk = pd.DataFrame(res, columns=["ACCNO", "PaymentDateTime", feature_name])
        
        data_cp["PaymentDateTime"] = pd.to_datetime(data_cp.PaymentDateTime)
        data_chunk["PaymentDateTime"] = pd.to_datetime(data_chunk.PaymentDateTime)
        
        data_chunk = pd.merge(data_cp[["ACCNO", "PaymentDateTime", "SaleDate"]], data_chunk, how="left", on=["ACCNO", "PaymentDateTime"])
        data_chunk["PaymentDateTime"] = pd.to_datetime(data_chunk.PaymentDateTime)
        data_chunk["SaleDate"] = pd.to_datetime(data_chunk.SaleDate)
        data_chunk.loc[
            data_chunk[feature_name].isnull() & ((data_chunk.PaymentDateTime - data_chunk.SaleDate).dt.days < k_days), feature_name
        ] = -1
        data_chunk.loc[
            data_chunk[feature_name].isnull() & ((data_chunk.PaymentDateTime - data_chunk.SaleDate).dt.days > k_days), feature_name
        ] = -100
        return data_chunk
    
    @staticmethod
    def num_of_payments_before_date_col(data: pd.DataFrame, col_alias: int, date_col_name: str, **kwargs) -> pd.DataFrame:
        returnc_cols = ["ACCNO", "PaymentDateTime", "OriginFileRecordID", col_alias]
        data_cp = data[["ACCNO", 'PaymentDateTime', "TotalAmount", date_col_name, 'DummyVar', 'OriginFileRecordID']].copy().reset_index(drop=True)
        # data_cp.sort_values(['ACCNO', 'PaymentDateTime', 'TotalAmount'], inplace=True, ignore_index=True)
        data_cp[date_col_name] = data_cp.groupby('ACCNO', group_keys=False)[date_col_name]\
            .apply(lambda x: x.ffill().bfill())
        
        null_mask = data_cp[date_col_name].isnull()
        if null_mask.sum() == 0:
            data_chunck_1 = pd.DataFrame(columns=returnc_cols)
        else:
            data_chunck_1 = data_cp[null_mask]\
                .groupby(["ACCNO", 'OriginFileRecordID', 'PaymentDateTime'], group_keys=True)\
                .DummyVar.apply(lambda x: x.cumsum())\
                .reset_index(name=col_alias)
            data_chunck_1[col_alias] = data_chunck_1.groupby("ACCNO")[col_alias].cumsum()
            data_chunck_1[col_alias] = data_chunck_1[col_alias] - 1
        
        if null_mask.sum() == data_cp.shape[0]:
            data_chunck = pd.DataFrame(columns=returnc_cols)
        else:
            data_chunck = data_cp[~null_mask]\
                .groupby(["ACCNO", 'OriginFileRecordID', 'PaymentDateTime'], group_keys=True)\
                .apply(
                    lambda x: ((pd.to_datetime(x[date_col_name]) > pd.to_datetime(x["PaymentDateTime"])).astype(int).cumsum() > 0).astype(int)
            )
            if type(data_chunck) == pd.Series:
                data_chunck = data_chunck.reset_index(name=col_alias)
            elif type(data_chunck) == pd.DataFrame:
                data_chunck = data_chunck.reset_index().rename(columns={0: col_alias})
            data_chunck[col_alias] = data_chunck.groupby("ACCNO")[col_alias].cumsum() - 1
        
        data_chunck = pd.concat((data_chunck, data_chunck_1)).reset_index(drop=True)
        data_chunck.loc[data_chunck[col_alias] < 0, col_alias] = 0
        
        assert data_chunck.shape[0] == data_cp.shape[0], f"Expected shape of a dataframe with the feature: {data_cp.shape[0]}; Got: {data_chunck.shape[0]}"
        return data_chunck[returnc_cols]
    
    @staticmethod
    def num_of_payments_within_n_days_by_user(df_group: pd.DataFrame, n_days: int, date_from: str, **kwargs) -> pd.Series:
        df_group_copy = df_group[["ACCNO", "PaymentDateTime", "TotalAmount"]].copy().reset_index(drop=True)
        # df_group_copy.sort_values(["ACCNO", "PaymentDateTime", "TotalAmount"], inplace=True)
        unique_dates = df_group.PaymentDateTime.unique()
        unique_dates_rel = unique_dates[unique_dates >= pd.to_datetime(date_from)]
        n_payments_dict = dict()
        for d in unique_dates_rel:
            d_left = pd.to_datetime(pd.to_datetime(d) - pd.offsets.DateOffset(days=n_days)).date()
            n_payments = df_group[
                (pd.to_datetime(df_group.PaymentDateTime) >= pd.to_datetime(d_left)) & 
                (pd.to_datetime(df_group.PaymentDateTime) < pd.to_datetime(d))
            ].shape[0]
            n_payments_dict[str(d)] = n_payments
        return n_payments_dict

    @staticmethod
    def num_of_payments_within_n_hours_by_user(df_group: pd.DataFrame, n_hours: int, date_from: str, **kwargs) -> pd.Series:
        df_group_copy = df_group[["ACCNO", "PaymentDateTime", "TotalAmount"]].copy().reset_index(drop=True)
        # df_group_copy.sort_values(["ACCNO", "PaymentDateTime", "TotalAmount"], inplace=True, ignore_index=True)
        n_payments_dict = dict()
        for i in range(df_group_copy.shape[0]):
            current_time = pd.to_datetime(df_group_copy.loc[i, "PaymentDateTime"])
            if current_time < pd.to_datetime(date_from):
                continue
            d_left = pd.to_datetime(
                current_time - pd.offsets.DateOffset(hours=n_hours)
            )
            n_payments = df_group[
                (pd.to_datetime(df_group.PaymentDateTime) >= d_left) &
                (pd.to_datetime(df_group.PaymentDateTime) < current_time)
            ].shape[0]
            n_payments_dict[str(current_time)] = n_payments
        return n_payments_dict

    def compute_num_of_payments_within_n_days(self, data: pd.DataFrame, n_days: int, date_from: str, **kwargs) -> pd.DataFrame:
        feature_name = f"NumOfPaymentBy{n_days}Days"
        data_chunck = data[
            data.PaymentDateTime >= (pd.to_datetime(date_from) - pd.Timedelta(days=n_days))
        ].groupby(["ACCNO"]).apply(
            lambda x: self.num_of_payments_within_n_days_by_user(x, n_days, date_from)
        ).reset_index(name=feature_name)
        
        if "level_1" in data_chunck.columns:
            del data_chunck["level_1"]
        if "index" in data_chunck.columns:
            del data_chunck["index"]
        
        df_transformed = pd.concat(
            (
                pd.concat(
                    (
                        pd.DataFrame.from_dict(
                            values[1][feature_name], orient="index"
                        ).reset_index().rename(columns={"index": "PaymentDateTime", 0: feature_name}), 
                        pd.Series([values[1]["ACCNO"]] * len(values[1][feature_name]), name="ACCNO")
                    ), axis=1
                ) 
                for values in data_chunck.iterrows()
            )
        )
        df_transformed["PaymentDateTime"] = pd.to_datetime(df_transformed["PaymentDateTime"])
        df_transformed.sort_values(["ACCNO", "PaymentDateTime"])
        return df_transformed.reset_index(drop=True)

    def compute_num_of_payments_within_n_hours(self, data: pd.DataFrame, n_hours: int, date_from: str, **kwargs) -> pd.DataFrame:
        feature_name = f"NumOfPaymentBy{n_hours}Hours"
        data_chunck = data[
            data.PaymentDateTime >= (pd.to_datetime(date_from) - pd.Timedelta(hours=n_hours))
        ].groupby(["ACCNO"]).apply(
            lambda x: self.num_of_payments_within_n_hours_by_user(x, n_hours, date_from)
        ).reset_index(name=feature_name)

        if "level_1" in data_chunck.columns:
            del data_chunck["level_1"]
        if "index" in data_chunck.columns:
            del data_chunck["index"]

        df_transformed = pd.concat(
            (
                pd.concat(
                    (
                        pd.DataFrame.from_dict(
                            values[1][feature_name], orient="index"
                        ).reset_index().rename(columns={"index": "PaymentDateTime", 0: feature_name}),
                        pd.Series([values[1]["ACCNO"]] * len(values[1][feature_name]), name="ACCNO")
                    ), axis=1
                )
                for values in data_chunck.iterrows()
            )
        )
        df_transformed["PaymentDateTime"] = pd.to_datetime(df_transformed["PaymentDateTime"])
        df_transformed.sort_values(["ACCNO", "PaymentDateTime"])
        return df_transformed.reset_index(drop=True)

