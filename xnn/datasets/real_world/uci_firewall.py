from scipy.io import arff
import numpy as np

from xnn.common.bandit_class import BanditDataset

FILEPATH = "/Users/mdwyer/Documents/Code/clean_up/NN-BF/dataset_files/firewall_dataset.arff"

class FirewallClassificationData(BanditDataset):

    def __init__(self, dataset=None, actions=[0, 1, 2]) -> None:

        self.columns_to_remove = ["BytesReceived", "pkts_received", "ElapsedTime", "Packets", "Bytes"]

        if dataset is None:
            dataset = self.load_dataset()

        self.actions = actions
        # self.class_splits = class_splits
        # self.dataset_size = dataset_size

        # self.dataset_size = len(dataset_size)

        super().__init__(dataset, actions)

    @staticmethod
    def find_diff(value_1, value_2):
        # process the context data from the firewall dataset
        # [0] SourcePort
        # [1] DestinationPort
        # [2] NATSourcePort
        # [3] NATDestinationPort
        # [4] Bytes
        # [5] BytesSent
        # [6] BytesReceived
        # [7] Packets
        # [8] ElapsedTime
        # [9] pkts_sent
        # [10] pkts_received

        processed_diff = value_1 - value_2

        # compare the port values:
        port_indices = [0, 1, 2, 3]

        for index in port_indices:
            # if the ports are exactly the same then the difference is zero, else it's one
            if value_1[index] == value_2[index]:
                processed_diff[index] = 0
            else:
                processed_diff[index] = 1

        diff = np.linalg.norm(processed_diff)/len(processed_diff)

        return diff

    def action_function(self, x, a, label=None):
        return self.classification_action(x, a, label)

    def load_dataset(self):
        # Load your ARFF file
        data, meta = arff.loadarff(FILEPATH)

        # Convert to a NumPy structured array
        structured_array = np.array(data.tolist(), dtype=data.dtype)

        # firewall dataset structure:
        # @ATTRIBUTE SourcePort INTEGER
        # @ATTRIBUTE DestinationPort INTEGER
        # @ATTRIBUTE NATSourcePort INTEGER
        # @ATTRIBUTE NATDestinationPort INTEGER
        # @ATTRIBUTE Action INTEGER
        # @ATTRIBUTE Bytes INTEGER
        # @ATTRIBUTE BytesSent INTEGER
        # @ATTRIBUTE BytesReceived INTEGER
        # @ATTRIBUTE Packets INTEGER
        # @ATTRIBUTE ElapsedTime INTEGER
        # @ATTRIBUTE pkts_sent INTEGER
        # @ATTRIBUTE pkts_received INTEGER
        action_column = structured_array["Action"]

        columns_list = [(name, typ) for name, typ in structured_array.dtype.descr if name != "Action"]
        columns_list.append(("Action", action_column.dtype.str))

        new_dtype = np.dtype(columns_list)

        # Create a new array with the new dtype
        new_structured_array = np.empty(structured_array.shape, dtype=new_dtype)

        # Fill the new array with data from the old array
        for name in new_structured_array.dtype.names:
            new_structured_array[name] = structured_array[name]

        train_data = self.normalise_data(new_structured_array)

        for column_name in self.columns_to_remove:
            new_structured_array = self.remove_field_name(new_structured_array, column_name)

        # print("post removal\n", train_data[:10])

        # train_data = np.unique(train_data)

        # print(train_data[:10])

        return train_data
 
    @staticmethod
    def remove_field_name(a, name):
        names = list(a.dtype.names)
        
        if name in names:
            names.remove(name)
        
        b = a[names]

        return b

    @staticmethod
    def normalise_data(new_structured_array):
        train_data = []
        names = list(new_structured_array.dtype.names)

        columns_to_norm = ["Bytes", "BytesSent", "BytesReceived", "Packets", "ElapsedTime", "pkts_sent", "pkts_received"]
        max_values = {}

        for name in names:
            max_values[name] = np.max(new_structured_array[name])

        for entry in new_structured_array:
            size = len(entry)
            new_entry = np.zeros(size)
            
            for i in range(size):
                new_entry[i] = int(entry[i])
                if names[i] in columns_to_norm:
                    new_entry[i] = new_entry[i]/max_values[names[i]]

            train_data.append(new_entry)

        return train_data

    def split_up_dataset(self, dataset_size, all_actions=False, num_copies=3, tuning_size=None):
        allow_data = []
        deny_data = []
        drop_data = []
        reset_all = []

        for entry in self.dataset:
        # Allow = 0, Drop = 1, Deny = 2, Reset-both = 3
            if entry[-1] == 0:
                allow_data.append(entry)
            elif entry[-1] == 1:
                drop_data.append(entry)
            elif entry[-1] == 2:
                deny_data.append(entry)
            elif entry[-1] == 3:
                reset_all.append(entry)

        np.random.shuffle(allow_data)
        np.random.shuffle(drop_data)
        np.random.shuffle(deny_data)
        np.random.shuffle(reset_all)

        datasets = []

        if all_actions:
            allow_percentage = len(allow_data)/len(self.dataset)
            deny_percentage = len(deny_data)/len(self.dataset)
            drop_percentage = len(drop_data)/len(self.dataset)
            reset_percentage = len(reset_all)/len(self.dataset)

            allow_prop = int(allow_percentage*dataset_size)
            deny_prop = int(deny_percentage*dataset_size)
            drop_prop = int(drop_percentage*dataset_size)
            reset_prop = int(reset_percentage*dataset_size)

            class_splits = [allow_percentage, deny_percentage, drop_percentage, reset_percentage]

            for i in range(num_copies):
                start_index = i*dataset_size

                allow_index = allow_prop + start_index
                deny_index = deny_prop + start_index
                drop_index = drop_prop + start_index
                reset_index = reset_prop + start_index

                dataset = allow_data[start_index:allow_index] + deny_data[start_index:deny_index] + drop_data[start_index:drop_index] + reset_all[start_index:reset_index]
                dataset = np.array(dataset)
                datasets.append(dataset)

        else:
            total_length = len(self.dataset) - len(reset_all)

            allow_percentage = len(allow_data)/total_length
            deny_percentage = len(deny_data)/total_length
            drop_percentage = len(drop_data)/total_length

            print(f"allow: {allow_percentage}, deny_percentage: {deny_percentage}, drop_percentage: {drop_percentage}")

            allow_prop = int(allow_percentage*dataset_size)
            deny_prop = int(deny_percentage*dataset_size)
            drop_prop = int(drop_percentage*dataset_size)
            reset_prop = 0

            class_splits = [allow_percentage, deny_percentage, drop_percentage]

            start = 0

            for i in range(num_copies):
                start_index = start + (i*dataset_size)

                allow_index = allow_prop + start_index
                deny_index = deny_prop + start_index
                drop_index = drop_prop + start_index

                dataset = allow_data[start_index:allow_index] + deny_data[start_index:deny_index] + drop_data[start_index:drop_index]
                dataset = np.array(dataset)
                datasets.append(dataset)

        print(f"allow: {allow_prop}, deny: {deny_prop}, drop: {drop_prop}, reset: {reset_prop}")

        return datasets, class_splits