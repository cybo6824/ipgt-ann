import numpy as np
import pickle

DAY_SECONDS = 86400
HOUR_SECONDS = 3600
SAMPLING_RATES_MIN = {"mean": 0.7, "std": 0.35, "min": 0.2}
SAMPLING_RATES_MAX = {"mean": 8.41, "std": 2.98, "min": 1.95}
SAMPLING_MAX_MULTI = 10.0
TIME_FREQS = [
    0.0572252,
    0.05846027,
    0.04322767,
    0.06998765,
    0.09468917,
    0.22725401,
    0.29971182,
    0.39069576,
    0.49485385,
    0.51132153,
    0.52449568,
    0.53561136,
    0.61259778,
    0.74310416,
    0.92095513,
    1.0,
    0.9691231,
    0.74063401,
    0.45244957,
    0.19596542,
    0.09798271,
    0.09880609,
    0.10209963,
    0.08892548,
]


class PriorityQueue:
    def __init__(self, num_groups):
        self.groups = list()
        self.group_ids = list()
        for i in range(num_groups):
            self.groups.append(list())
            self.group_ids.append(list())
        self.counts = [0] * num_groups
        self.sums = [0] * num_groups

    def _Sort(self):
        sorted_groups = zip(
            *sorted(
                zip(self.groups, self.group_ids, self.counts, self.sums),
                key=lambda x: (x[2], -x[3]),
            )
        )
        self.groups, self.group_ids, self.counts, self.sums_s = map(list, sorted_groups)

    def Insert(self, length, id):
        self.groups[0].append(length)
        self.group_ids[0].append(id)
        self.counts[0] += 1
        self.sums[0] += length
        self._Sort()


def GetFolds(pig_series, max_folds=10):
    lengths_dict = dict()
    for animal_id, animal_data in pig_series.items():
        lengths_dict[animal_id] = np.amax(animal_data[:, 3]) / DAY_SECONDS
    fold_sets = dict()
    fold_sets_lengths = dict()
    sorted_ids, sorted_lengths = zip(
        *sorted(zip(lengths_dict.keys(), lengths_dict.values()), key=lambda x: x[1])
    )
    sorted_ids, sorted_lengths = list(sorted_ids), list(sorted_lengths)

    for bin_count in range(2, max_folds + 1):
        priority_queue = PriorityQueue(bin_count)
        for i in range(len(sorted_ids)):
            priority_queue.Insert(sorted_lengths[i], sorted_ids[i])
        fold_sets[bin_count] = priority_queue.group_ids
        fold_sets_lengths[bin_count] = priority_queue.groups

    with open("data_folds.dat", "wb") as fh:
        pickle.dump(fold_sets, fh)


def DataSampling(
    animal_id, rng, pig_data, sample_rates, series_length, time_interp=False
):
    trim_index = np.argmin(
        np.abs((pig_data[animal_id][:, 3] / DAY_SECONDS) - series_length)
    )
    weights = pig_data[animal_id][:trim_index, 5]
    times = pig_data[animal_id][:trim_index, 7]
    feed_quan = pig_data[animal_id][:trim_index, 6]
    feed_dur = pig_data[animal_id][:trim_index, 4]
    days_length = (times[-1] - times[0]) / DAY_SECONDS

    if time_interp:
        # Interpolation
        interp_times = np.arange(times[0], times[-1], step=HOUR_SECONDS)
        interp_weights = np.interp(interp_times, times, weights)
        interp_feed_quan = np.interp(interp_times, times, np.cumsum(feed_quan))
        interp_feed_dur = np.interp(interp_times, times, np.cumsum(feed_dur))

        # Select pig weights such that time frequency distribution matches the target
        start_hour = int(np.around((times[0] % DAY_SECONDS) / HOUR_SECONDS))
        time_freq_t = TIME_FREQS[start_hour:] + TIME_FREQS[:start_hour]
        time_freq_t = np.tile(time_freq_t, int(np.ceil(days_length)))[
            : interp_times.shape[0]
        ]
        time_freq_r = rng.uniform(size=interp_times.shape)

        sample_indices = time_freq_t > time_freq_r
        sample_times = interp_times[sample_indices] - times[0]
        sample_weights = interp_weights[sample_indices]
        sample_feed_quan = interp_feed_quan[sample_indices]
        sample_feed_dur = interp_feed_dur[sample_indices]
    else:
        sample_times = pig_data[animal_id][:trim_index, 3]
        sample_weights = weights
        sample_feed_quan = feed_quan
        sample_feed_dur = feed_dur

    # Match target sample frequency
    sampling_rate = rng.normal(loc=sample_rates["mean"], scale=sample_rates["std"])
    sampling_rate *= days_length
    sampling_rate = min(sampling_rate, sample_times.shape[0])
    sample_indices = np.arange(sample_times.shape[0])
    sample_count = max(sampling_rate, sample_rates["min"] * days_length, 3.0)
    sample_indices = rng.choice(sample_indices, size=round(sample_count), replace=False)
    sample_indices = np.sort(sample_indices)

    sample_id = np.repeat([animal_id], sample_times.shape[0])
    sample_data = np.stack(
        (sample_weights, sample_times, sample_id, sample_feed_quan, sample_feed_dur),
        axis=1,
    )
    sample_data = sample_data[sample_indices, :]
    return sample_data


def DataSamplingUniform(rng, animal_id, pig_data, inc_mul=None, red_mul=0.25):
    weights = pig_data[animal_id][:, 5]
    times = pig_data[animal_id][:, 3]
    if inc_mul != None:
        interp_times = np.linspace(
            times[0], times[-1], num=int(np.ceil(weights.shape[0] * inc_mul))
        )
        weights = np.interp(interp_times, times, weights)
    sample_indices = rng.choice(
        weights.shape[0], size=int(weights.shape[0] * red_mul), replace=False
    )
    sample_times = times[sample_indices]
    sample_weights = weights[sample_indices]
    return sample_times, sample_weights


def UpdateSampleRates(sample_rate):
    sample_rate = min(sample_rate, SAMPLING_MAX_MULTI)
    stat_names = ["mean", "min", "std"]
    new_rates = dict()
    for stat in stat_names:
        new_rates[stat] = SAMPLING_RATES_MIN[stat] * sample_rate
    return new_rates


def GroupSampling(
    pig_data, group_id=0, pig_count=55, series_length=35, sample_rate=1.0, fold_set=None
):
    rng = np.random.Generator(np.random.PCG64(group_id))
    if fold_set != None:
        all_ids = fold_set
    else:
        all_ids = pig_data.keys()
    id_set = list()
    for animal_id in all_ids:
        if pig_data[animal_id][-1, 3] / DAY_SECONDS > series_length:
            id_set.append(animal_id)
    animal_ids = rng.choice(id_set, size=pig_count, replace=False)
    sample_rates = UpdateSampleRates(sample_rate)

    group_data = list()
    for i in animal_ids:
        data_i = DataSampling(i, rng, pig_data, sample_rates, series_length)
        group_data.append(data_i)
    group_data = np.concatenate(group_data, axis=0)
    group_mat = group_data[np.argsort(group_data[:, 1])]  # sort by time
    return group_mat
