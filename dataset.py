from utils.general_functions import shuffled_copy, rgb2hsv

def read_data(files):
    data = []
    labels = []

    label_offset = 0

    for file in files:
        with open(file, "r") as f:
            for line in f:
                rgb = list(map(float, line.strip().split(" ")))
                rgb = rgb2hsv(rgb)
                rgb = [int(rgb[0]), int(rgb[1]), int(rgb[2])]# scale back to [0,1023] for HSV
                data.append(rgb)
                labels.append(label_offset)
        label_offset += 1

    return data, labels


# Split the data into train, validation, and test sets
def split_data(all_data, all_labels, TRAIN_SIZE=0.4, VALID_SIZE=0.1, test=False, seed=2):

    train_size = int(TRAIN_SIZE * len(all_data))
    valid_size = int(VALID_SIZE * len(all_data))

    indexes = [i for i in range(0, len(all_data))]
    indexes = shuffled_copy(indexes, seed=seed)

    train_indexes = indexes[:train_size]
    valid_indexes = indexes[train_size:train_size + valid_size]
    

    X_train = [all_data[i]   for i in train_indexes]
    y_train = [all_labels[i] for i in train_indexes]
    X_val = [all_data[i]   for i in valid_indexes]
    y_val = [all_labels[i] for i in valid_indexes]

    if test:
        test_indexes  = indexes[train_size + valid_size:]
        X_test = [all_data[i]   for i in test_indexes]
        y_test = [all_labels[i] for i in test_indexes]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    return (X_train, y_train), (X_val, y_val)

# Split the data into train, validation, and test sets
def split_data_test(all_data, all_labels, TRAIN_SIZE=0.4, VALID_SIZE=0.1, seed=2):

    train_size = int(TRAIN_SIZE * len(all_data))
    valid_size = int(VALID_SIZE * len(all_data))

    indexes = [i for i in range(0, len(all_data))]
    indexes = shuffled_copy(indexes, seed=seed)

    test_indexes  = indexes[train_size + valid_size:]

    X_test = [all_data[i]   for i in test_indexes]
    y_test = [all_labels[i] for i in test_indexes]

    return (X_test, y_test)


def normalize_data(data):
    for i in range(len(data)):
        data[i] = [d/1023 for d in data[i]]
    return data

# def augment_sample(x, noise_range=10):
#     # sample noise
#     noise = [random.randint(-noise_range, noise_range) for _ in range(len(x))]
    
#     # augment
#     augm_X = [x[i] + noise[i] for i in range(len(x))]

#     # clip
#     augm_X = [1023 if a > 1023 else 0 if a < 0 else a for a in augm_X]

#     return augm_X