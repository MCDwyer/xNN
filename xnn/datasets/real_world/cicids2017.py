import numpy as np
import pandas as pd
import os

from xnn.common.bandit_class import BanditDataset

FILEPATH = "dataset_files/CICIDS2017/MachineLearningCVE"

BENIGN = "BENIGN"
BOT = "Bot"
BRUTE_FORCE = "BRUTE_FORCE"
DOS_DDOS = "DoS/DDoS"
INFILTRATION = 'Infiltration'
PORTSCAN = "PortScan"
WEB_ATTACK = "WEB_ATTACK"

FTP_PATATOR = 'FTP-Patator'
SSH_PATATOR = 'SSH-Patator'
SLOWLORIS = 'DoS slowloris'
SLOWHTTPTEST = 'DoS Slowhttptest'
HULK = 'DoS Hulk'
GOLDENEYE = 'DoS GoldenEye'
HEARTBLEED = 'Heartbleed'
WEB_ATTACK_BRUTE_FORCE = 'Web Attack � Brute Force'
WEB_ATTACK_XSS = 'Web Attack � XSS'
WEB_ATTACK_SQL_INJ = 'Web Attack � Sql Injection'
DDOS = 'DDoS'

KURNIABUDI_FEATURE_ORDER = ["Packet Length Std", "Total Length of Bwd Packets", "Subflow Bwd Bytes",
                           "Destination Port", "Packet Length Variance", "Bwd Packet Length Mean",
                           "Avg Bwd Segment Size", "Bwd Packet Length Max", "Init_Win_bytes_backward",
                           "Total Length of Fwd Packets", "Subflow Fwd Bytes", "Init_Win_bytes_forward",
                           "Average Packet Size", "Packet Length Mean", "Max Packet Length",
                           "Fwd Packet Length Max", "Flow IAT Max", "Bwd Header Length",
                           "Flow Duration", "Fwd IAT Max", "Fwd Header Length", "Fwd IAT Total",
                           "Fwd IAT Mean", "Flow IAT Mean", "Flow Bytes/s", "Bwd Packet Length Std",
                           "Subflow Bwd Packets", "Total Backward Packets", "Fwd Packet Length Mean",
                           "Avg Fwd Segment Size", "Bwd Packet Length Min", "Flow Packets/s",
                           "Fwd Packets/s", "Bwd IAT Max", "Bwd Packets/s"]

# basically all continuous data except for Destination Port? I think? 

class CICIDS2017(BanditDataset):
    def __init__(self, dataset_size=-1, num_features=77, relabel=False, Kurniabudi=False, use_weighting=False) -> None:
        super().__init__([None], [None], shuffle=False)
        self.column_names = None
        self.weightings = np.ones(num_features)/num_features
        self.use_standard_action_function = True
        # self.use_noise = False

        if dataset_size == 0:
            return

        if Kurniabudi:
            relabel = True
            dataset_size = -1

        dataset, actions, class_splits = self.load_in_data(num_features, dataset_size, relabel)

        if use_weighting:
            self.set_weightings(num_features)
        else:
            self.weightings = np.ones(num_features)/num_features

        self.class_splits = class_splits

        if Kurniabudi:
            dataset = self.set_up_for_Kurniabudi(dataset)

        num_dims = dataset.shape[1] - 1 # to remove label column
        dims_to_reduce = range(1, num_dims) # as 0 index is destination port, so don't want to change that as it's categorical?

        dataset = self.reduce_data_range(dataset, dims_to_reduce=dims_to_reduce)
        # dataset = self.normalise_data(dataset)

        super().__init__(dataset, actions, shuffle=False)

    def load_in_data(self, num_features, dataset_size, relabel=False, get_df=False):
        # load in the data and remove redundant column, make sure action label is final column too
        # redundant column is duplicate of Fwd Header Length
        # df = pd.read_csv('data.csv')
        csv_files = [file for file in os.listdir(FILEPATH) if file.endswith('.csv')]

        labels = None

        dataframes = {"Monday": [], "Tuesday": [], "Wednesday": [], "Thursday": [], "Friday": []}

        for file in csv_files:
            file_path = os.path.join(FILEPATH, file)
            df = pd.read_csv(file_path)

            # remove leading and trailing whitespaces from the column names
            df.columns = df.columns.str.strip()

            # remove the duplicate fwd header length column
            df = df.drop(["Fwd Header Length.1"], axis=1)

            # reduce the feature set down
            if num_features < 77:
                df = self.feature_selection(df, num_features)

                assert df.shape[1] == (num_features + 1) # +1 due to label column

            df.replace([np.inf, -np.inf], np.nan, inplace=True) # replacing infs with NaNs so can drop them

            # drop rows containing NaN values from the dataframe
            df = df.dropna()

            if self.column_names is None:
                self.column_names = df.columns.to_list()

            # find which day this file section belongs to
            file_string_sections = file.split('-')
            day = file_string_sections[0]

            # append to the relevant day entry in the dictionary
            dataframes[day].append(df)

        # remove empty lists from dictionary (if not all days are being loaded in?)
        dataframes = {k: v for k, v in dataframes.items() if v}

        if get_df:
            return dataframes

        num_days = len(dataframes)

        if dataset_size < 0:
            dataset_size_per_day = -1
        else:
            dataset_size_per_day = int(dataset_size/num_days)

        day_keys = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        all_days = []

        dataset_size = 0

        actions = []
        class_splits = []

        all_day_dfs = []

        # this is to ensure the data is kept in the correct day order (i.e. Monday first)
        for key in day_keys:

            if key not in dataframes:
                print(f"Key: {key} not in dataframes")
                continue

            day_dfs = dataframes[key]

            if len(day_dfs) > 1:
                # need to combine the dataframes
                day_df = pd.concat(day_dfs, axis=0, ignore_index=True)
            else:
                day_df = day_dfs[0]

            labels = day_df["Label"].unique()

            day_df_size = len(day_df)
            day_arrays = []

            # find proportions of the data labels for this dataset and get that proportion of the reduced size out
            for label in labels:

                df_entries = len(day_df[day_df['Label'] == label])

                if dataset_size_per_day < 0:
                    # use the full dataset
                    num_entries = df_entries
                else:
                    proportion = df_entries/day_df_size
                    num_entries = int(proportion*dataset_size_per_day)

                dataset_size += num_entries

                assert num_entries <= df_entries

                if relabel:
                    new_label = self.get_new_label(label)
                    day_df.loc[day_df['Label'] == label, 'Label'] = new_label

                    label = new_label

                if label not in actions:
                    actions.append(label)
                    class_splits.append(0)
                    index = -1
                else:
                    index = actions.index(label)

                class_splits[index] += num_entries

                reduced_df = day_df[day_df['Label'] == label].sample(n=num_entries)

                if get_df:
                    all_day_dfs.append(reduced_df)

                day_arrays.append(reduced_df.to_numpy())

            day_array = np.vstack(day_arrays)

            all_days.append(day_array)

        if get_df:
            return pd.concat(all_day_dfs, axis=0, ignore_index=True)

        full_dataset = np.vstack(all_days)

        # assert len(full_dataset) < dataset_size

        assert full_dataset.shape == (dataset_size, num_features + 1), f"Dataset shape: {full_dataset.shape} but expected shape is ({dataset_size}, {num_features + 1})"

        class_splits = np.array(class_splits)
        class_splits = class_splits/dataset_size

        return full_dataset, actions, class_splits

    def get_new_label(self, label):
        # relabel data as per Kurniabudi et al.: CICIDS-2017 Dataset Feature Analysis With Information Gain for Anomaly Detection

        if label in {BENIGN, BOT, BRUTE_FORCE, DOS_DDOS, INFILTRATION, PORTSCAN, WEB_ATTACK}:
            return label

        if label in {FTP_PATATOR, SSH_PATATOR}:
            return BRUTE_FORCE

        if label in {DDOS, GOLDENEYE, HULK, SLOWHTTPTEST, SLOWLORIS, HEARTBLEED}:
            return DOS_DDOS

        if label in {WEB_ATTACK_BRUTE_FORCE, WEB_ATTACK_SQL_INJ, WEB_ATTACK_XSS}:
            return WEB_ATTACK

        print(f"Relabelling hasn't worked on: {label}.")

        return label

    def feature_selection(self, df, num_features):
        # reduce number of features according to what Kurniabudi et al. found - 4, 15, 22, 35, 52, 57, 77
        # to begin with just gonna do 4 - 35 and 77 (maybe)?

        # 4 features = 41, 13, 65, 8
        # 15 features += 42, 20, 54, 18, 67, 12, 63, 66, 52, 40, 39
        # 22 features += 14, 22, 36, 9, 26, 55, 24
        # 35 features += 25, 21, 2, 1, 64, 11, 16, 53, 19, 3, 37, 30, 7
        # else 77 (all features)

        columns_list = df.columns.to_list()

        columns_to_drop = [item for item in columns_list if item not in KURNIABUDI_FEATURE_ORDER[:num_features]]

        # keep label in always
        columns_to_drop.remove("Label")

        df = df.drop(columns_to_drop, axis=1)

        # print(f"{len(df.columns.to_list())} columns kept:")
        # for column_name in df.columns.to_list():
        #     print(f"\t{column_name}")

        return df

    def set_up_for_Kurniabudi(self, dataset):
        # they used a 396,304 size training dataset, consisting of:
        separate_data = {}
        for entry in dataset:
            label = entry[-1]

            if label not in separate_data:
                separate_data[label] = []

            separate_data[label].append(entry) # append the data to relevant label

        for label in separate_data:
            np.random.shuffle(separate_data[label])

        # 318,087 benign
        benign_data = separate_data[BENIGN][:318087]
        # 265 bot
        bot_data = separate_data[BOT][:265]
        # 1904 brute force
        brute_data = separate_data[BRUTE_FORCE][:1904]
        # 53427 ddos/dos
        dos_data = separate_data[DOS_DDOS][:53427]
        # 5 infiltration
        infiltration_data = separate_data[INFILTRATION][:5]
        # 22,324 port scan
        portscan_data = separate_data[PORTSCAN][:22324]
        # 292 web attack
        webattack_data = separate_data[WEB_ATTACK][:292]

        training_data = np.vstack((benign_data, bot_data, brute_data, dos_data, infiltration_data, portscan_data, webattack_data))
        np.random.shuffle(training_data)

        # then they used 136,219 size test dataset consisting of:
        # 136,219 benign
        benign_data = separate_data[BENIGN][318087:454306]
        # 102 bot
        bot_data = separate_data[BOT][265:367]
        # 813 brute force
        brute_data = separate_data[BRUTE_FORCE][1904:2717]
        # 23,018 ddos/dos
        dos_data = separate_data[DOS_DDOS][53427:76445]
        # 1 infiltration
        infiltration_data = separate_data[INFILTRATION][5:6]
        # 9,558 port scan
        portscan_data = separate_data[PORTSCAN][22324:31882]
        # 134 web attack
        webattack_data = separate_data[WEB_ATTACK][292:426]

        test_data = np.vstack((benign_data, bot_data, brute_data, dos_data, infiltration_data, portscan_data, webattack_data))
        np.random.shuffle(test_data)

        dataset = np.vstack((training_data, test_data))

        # ['BENIGN', 'BRUTE_FORCE', 'DoS/DDoS', 'Infiltration', 'WEB_ATTACK', 'Bot', 'PortScan']
        class_splits = [454306, 2717, 76445, 6, 426, 367, 31882]
        self.class_splits = np.array(class_splits)/len(dataset)

        self.dataset_size = len(dataset)

        return dataset

    def set_weightings(self, num_features):
        # TODO: set it up for 22 and 35 features
        assert num_features <= 15

        importance = np.array([0.638, 0.612, 0.612, 0.609, 0.577, 0.567, 0.567, 0.560, 0.554, 0.546, 0.546, 0.542, 0.535, 0.526, 0.512])
        weightings = np.zeros(num_features)

        for i in range(num_features):
            feature_name = self.column_names[i]
            index = KURNIABUDI_FEATURE_ORDER.index(feature_name)
            weightings[i] = importance[index]

        weightings = weightings/np.sum(weightings) # make it proportional

        self.weightings = weightings

    def find_diff(self, x, y):
        processed_diff = x - y

        if x[0] == y[0]:
            # if port number is same
            processed_diff[0] = 0
        else:
            processed_diff[0] = 1

        processed_diff = processed_diff*self.weightings

        diff = np.linalg.norm(processed_diff)

        return diff

    def action_function(self, x, a, label):

        loss = self.classification_action(x, a, label)

        # if not self.use_standard_action_function:
        #     if a != BENIGN: # i.e., not allowing through
        #         loss = 0.5 # get no info for this

        if self.use_standard_action_function:
            return loss
        else:
            # print('non standard')
            if a != BENIGN: # i.e., not allowing through
                return 0.5 # get no info for this
            else:
                # print(a)
                return loss # get info
