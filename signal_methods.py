"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a number of signal processing methods for use with deap
"""
import numpy as np
import random
import time
from GPFramework.data import GTMOEPDataInstance, GTMOEPData, GTMOEPDataPair, FeatureData, StreamData, GTMOEPImagePair
from scipy.optimize import curve_fit, nnls
from scipy.stats import multivariate_normal as mvn
from scipy.ndimage import filters
from lmfit import Model, Parameters, minimize
from lmfit.models import GaussianModel
import copy as cp
import scipy.signal
import sklearn.gaussian_process
import sklearn.decomposition
import sklearn.cluster
from sklearn.manifold import SpectralEmbedding
from skimage import restoration
# import hmmlearn.hmm
import pywt
import gc
import sys
from PIL import Image
import PIL
from scipy import interpolate
import pdb

from functools import partial

from GPFramework.constants import FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES, TRI_STATE
from GPFramework.registry_helper import RegistryWrapper, primitive_template
from GPFramework.data import GTMOEPDataPair
from GPFramework.constants import TriState

smw = RegistryWrapper([GTMOEPDataPair, TriState])

# TODO: More intuitive docstring insertion with intellisense support
def template(f, f_setup, f_name, supported_modes=[FEATURES_TO_FEATURES, STREAM_TO_STREAM, STREAM_TO_FEATURES], input_types=[]):
    """ Helper function (for signal_methods!) to interface with primitive_template

    Also registers function in SignalMethodsWrapper (for eventual migration). By default f's docstring is used.
    
    Args:
        f: Function that operates on instance data
        f_setup: Function that generates auxiliary data input parameters for f, like convolution kernels. 
            If None is used, input params will forward to f. Otherwise, only kwarg return from f_setup is passed to f.
            Additionally, input validity testing (proper ranges) should be done in f_setup!
        f_name: Name for function in debug and registration
        supported_modes: List of supported tristates
        
    Returns:
        Function to register in gp_framework_helper.
    """
    # Assumption about primitive registration formatting here
    kwpos = {
        'data_pair': 0,
        'mode': 1
    }
    wrapped = partial(primitive_template, kwpos, f, f_setup, f_name, supported_modes)
    return smw.register(f_name, input_types)(wrapped)

def makeFeatureFromClass(data_pair, name=''):
    """Makes class labels into a feature

    Adds new feature to existing data by appending results from a machine
    learning algorithm

    Args:
        data_pair: given datapair
        name: name of labels

    Returns:
        Data pair with class feature appended
    """
    data_list = []
    # For debugging purposes let's print out method name
    print('makeFeatureFromClass') ; sys.stdout.flush()
    # Only need to perform this on testData, machine learning methods handle training data
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        instances = cp.deepcopy(data_set.get_instances())

        for instance in instances:
            feature_set = instance.get_features()
            # Testing a copy here as I believe modifying this feature later could be causing a change in the target
            target = cp.deepcopy(instance.get_target())
            new_data = np.hstack((feature_set.get_data(), np.array([target])))
            data_labels = np.hstack((feature_set.get_labels(), name+str(len(new_data))))

            feature_set.set_data(new_data, labels=data_labels)
        new_data_set = GTMOEPData(instances)
        data_list.append(new_data_set)

    data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])
    gc.collect(); return data_pair

def copy_stream_to_target(data_pair):
    """Copies the data in the stream objects to the target value

    Args:
        data_pair: given datapair

    Returns:
        Data Pair
    """
    # For debugging purposes let's print out method name
    print('copy_stream_to_target') ; sys.stdout.flush()
    if isinstance(data_pair, GTMOEPImagePair):
        # do nothing if data is image data
        return data_pair
    # Initialize where the data will be temporarily stored
    data_list = []
    # Iterate through train data then test data
    for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
        # Copy the dataSet so as not to destroy original data
        instances = cp.deepcopy(data_set.get_instances())
        # Iterate over all points in the dataSet
        for instance in instances:
            # Capture the data
            data = instance.get_stream().get_data()
            # Set the data over to the taget
            instance.set_target(cp.deepcopy(data))


        new_data_set = GTMOEPData(instances)
        data_list.append(new_data_set)
    # Build deapDataPair
    data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])
    gc.collect(); return data_pair

# def remove_func(signal, feat_num):
#     return np.delete(signal, feat_num, 1)
# remove_feature = template(remove_func, None, "myRemoveFeature", input_types=[int])
# remove_feature.__doc__ = "Unimplemented"
# Template does not support non-tristate (probably legacy code)
def remove_feature(data_pair, feat_num):
    """Removes a feature from the data

    Args:
        data_pair: given datapair
        feat_num: id of the feature to remove

    Returns:
        Data Pair with feature removed
    """
    # For debugging purposes let's print out method name
    print('remove_feature')
    sys.stdout.flush()
    if isinstance(data_pair, GTMOEPImagePair):
        try:
            train_list = data_pair.get_train_batch()
            test_list = data_pair.get_test_batch()
            new_train_list = []
            new_test_list = []
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.delete(train, feat_num, 1))
                new_test_list.append(np.delete(test, feat_num, 1))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        # Only need to perform this on testData, machine learning methods handle training data
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                feature_set = instance.get_features()
                # Remove the feat_num column
                new_data = np.delete(feature_set.get_data(), feat_num, 1)

                data_labels = np.delete(feature_set.get_labels(), feat_num)

                feature_set.set_data(new_data, labels=data_labels)
            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)

        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])
    gc.collect(); return data_pair


# def select_range_func(signal, start, stop):
#     signal = np.uint8(signal) # Just trying not to regress
#     return signal[:,start:stop]
# select_range = template(select_range_func, None, "mySelectRange", input_types=[int,int])
# select_range.__doc__ = "Unimplemented" # Warning! In the old version of select_range, image_pairs were cast to uint8! This has been regressed
# Cannot currently collapse select_range, doesn't take a mode safely yet
def select_range(data_pair, feat_start, feat_stop):
    """Selects a range of data starting from feature feat_start and
    stopping before feature feat_stop

    Args:
        data_pair: given datapair
        feat_start: id of feature to start at
        feat_stop: id of feature to stop at

    Returns:
        Data Pair from feature start to feature stop
    """
    # For debugging purposes let's print out method name
    print('select_range') ; sys.stdout.flush()
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train in train_list:
                train = np.uint8(train)
                train_images = []
                for i in range(train.shape[0]):
                    img = train[i]
                    new_img = img[:, feat_start:feat_stop, :]
                    train_images.append(new_img)
                train = np.vstack(train_images)
                new_train_list.append(train)

            for test in test_list:
                test = np.uint8(test)
                test_images = []
                for i in range(test.shape[0]):
                    img = test[i]
                    new_img = img[:, feat_start:feat_stop, :]
                    test_images.append(new_img)
                test = np.vstack(test_images)
                new_test_list.append(test)

            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                feature_set = instance.get_features()
                # select the correct range
                new_data = feature_set.get_data()[:, feat_start:feat_stop]
                data_labels = feature_set.get_labels()[feat_start:feat_stop]

                feature_set.set_data(new_data, labels=data_labels)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)

        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])

    gc.collect(); return data_pair

def my_concatenate(data_pair_1, data_pair_2, mode=FEATURES_TO_FEATURES):
    """Concatenate twos dataframes together

    Args:
        data_pair_1: given datapair
        data_pair_2: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Concatenated Data Pair

    Raises:
        ValueError
    """
    # Initialize where the data will be temporarily stored
    # For debugging purposes let's print out method name
    print('concatenate') ; sys.stdout.flush()
    if isinstance(data_pair_1, GTMOEPImagePair):
        raise ValueError(
            'Cannot concatenate batched image data'
            )
    else:
        data_list = []
        # Iterate through train data then test data
        for data_set_1, data_set_2 in [(data_pair_1.get_train_data(), data_pair_2.get_train_data()),
                                       (data_pair_1.get_test_data(), data_pair_2.get_test_data())]:
            # Copy the dataSet so as not to destroy original data
            instances_1 = cp.deepcopy(data_set_1.get_instances())
            instances_2 = cp.deepcopy(data_set_2.get_instances())
            # Iterate over all points in the dataSet
            for instance_1, instance_2 in zip(instances_1, instances_2):
                if mode is FEATURES_TO_FEATURES:
                    signal_1 = instance_1.get_features().get_data()
                    signal_2 = instance_2.get_features().get_data()
                    labels_1 = instance_1.get_features().get_labels()
                    labels_2 = instance_2.get_features().get_labels()
                elif mode is STREAM_TO_STREAM:
                    signal_1 = instance_1.get_stream().get_data()
                    signal_2 = instance_2.get_stream().get_data()
                    labels_1 = instance_1.get_stream().get_labels()
                    labels_2 = instance_2.get_stream().get_labels()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for concatenation'
                        )
                # Concatenate the two signals together
                data = np.hstack( (signal_1, signal_2) )
                labels = np.hstack( (labels_1, labels_2) )

                # Rebuild the data in instance 1
                if mode is FEATURES_TO_FEATURES:
                    instance_1.get_features().set_data(data, labels=labels)
                elif mode is STREAM_TO_STREAM:
                    instance_1.get_stream().set_data(data, labels=labels)

            new_data_set = GTMOEPData(instances_1)
            data_list.append(new_data_set)
        # Build deapDataPair
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_if_then_else(condition_pair, column, data_pair_1, data_pair_2):
    """Consumes three data_pairs. Based upon the truth of the target feature in
    condition_pair, choose correct rows from data_pair_1 and data_pair_2.

    Args:
        condition_pair: given condition pair
        column: column number
        data_pair_1: given Data Pair
        data_pair_2: given Data Pair

    Returns:
        Data Pair with added if then else feature
    """
    # Convert column to an int
    column = int(column)
    # Check that lengths match
    if isinstance(data_pair_1, GTMOEPImagePair):
        raise ValueError(
            'Cannot if then else batched image data'
            )
    else:
        # Initialize output list of instances to empty
        train_output_instances = []
        test_output_instances = []
        # First process train data
        for condition_pair in condition_link.get_data_pairs():
            for condition_instance, instance_1, instance_2 in zip(
                    condition_pair.get_train_data().get_instances(),
                    data_pair_1.get_train_data().get_instances(),
                    data_pair_2.get_train_data().get_instances()):
                # Select the proper test column from the condition instance
                # And choose the proper row for output
                if condition_instance.get_features().get_data()[0, column]:
                    train_output_instances.append(instance_1)
                else:
                    train_output_instances.append(instance_2)

            # Next process process test data
            for condition_instance, instance_1, instance_2 in zip(
                    condition_pair.get_test_data().get_instances(),
                    data_pair_1.get_test_data().get_instances(),
                    data_pair_2.get_test_data().get_instances()):
                if condition_instance.get_features().get_data()[0, column]:
                    test_output_instances.append(instance_1)
                else:
                    test_output_instances.append(instance_2)

        output_pair = GTMOEPDataPair(train_data=GTMOEPData(train_output_instances),
                                     test_data=GTMOEPData(test_output_instances))

    gc.collect(); return data_pair

def convolve_func(signal, kern=None):
    if signal.ndim == 1: # 1D input, 1D output
        return np.convolve(signal, kern, mode='same')
    else:
        return np.array([np.convolve(sample, kern) for sample in signal])

def average_setup(signal, window=3):
    if window <= 0:
        window = 3
    kern = np.ones(window)
    kern = kern / float((window))
    return { "kern": kern }            

my_averager = template(convolve_func, average_setup, 'myAverager', supported_modes=[FEATURES_TO_FEATURES, STREAM_TO_STREAM], input_types=[int])
my_averager.__doc__ = \
"""Runs an averager of length window (Low Pass Filter) on data

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES
        window: length of window

    Returns:
        Data Pair with averager feature added

    Raises:
        ValueError
"""
def diff_setup(signal):
    return { "kern": [1., -1.] }

my_diff = template(convolve_func, diff_setup, 'myDiff', supported_modes=[FEATURES_TO_FEATURES, STREAM_TO_STREAM] )
my_diff.__doc__ = \
"""Runs a diff (High Pass Filter) on data

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with diff feature added

    Raises:
        ValueError
"""

def auto_corr_func(signal):
    # FFT defaults to last axis, we don't need a loop here
    F = np.fft.fft(signal)
    S = np.conjugate(F)*F
    R = np.fft.ifft(S)
    return np.real(R)

my_auto_corr = template(auto_corr_func, None, 'myAutoCorr')
my_auto_corr.__doc__ = \
"""Computes the autocorrelation using the Wiener-Khinchin Theorem

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with autocorrelation feature added

    Raises:
        ValueError
"""

def my_cross_corr(data_pair_1, data_pair_2, mode=FEATURES_TO_FEATURES):
    """Computes the cross correlation of two signals

    Args:
        data_pair_1: given datapair
        data_pair_2: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with cross correlation feature added

    Raises:
        ValueError
    """
    # For debugging purposes let's print out method name
    print('cross_corr') ; sys.stdout.flush()
    data_pair_list = []
    if isinstance(data_pair_1, GTMOEPImagePair):
        raise ValueError(
            'Cannot cross correlate batched image data'
            )
    else:
        data_list = []
        # Iterate through train data then test data
        for data_set_1, data_set_2 in [(data_pair_1.get_train_data(), data_pair_2.get_train_data()),
                                       (data_pair_1.get_test_data(), data_pair_2.get_test_data())]:
            # Copy the dataSet so as not to destroy original data
            instances_1 = cp.deepcopy(data_set_1.get_instances())
            instances_2 = cp.deepcopy(data_set_2.get_instances())
            # Iterate over all points in the dataSet
            for instance_1, instance_2 in zip(instances_1, instances_2):
                if mode is FEATURES_TO_FEATURES:
                    signal_1 = instance_1.get_features().get_data()
                    signal_2 = instance_2.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    signal_1 = instance_1.get_stream().get_data()
                    signal_2 = instance_2.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Cross Corr'
                        )
                F = np.fft.fft(signal_1)
                G = np.fft.fft(signal_2)
                S = np.conjugate(F)*G
                R = np.fft.ifft(S)
                data = np.real(R)
                if mode is FEATURES_TO_FEATURES:
                    instance_1.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance_1.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances_1)
            data_list.append(new_data_set)
        # Build deapDataPair
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_dct(data_pair, transform=2, norm=1, mode=FEATURES_TO_FEATURES):
    """Performs the discrete cosine transform of a signal x
    transform can be 1,2, or 3. norm can be None or ortho.

    http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.fftpack.dct.html

    Args:
        data_pair: given datapair
        transform: type of dct
        norm: normalization mode
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with dct feature added
    """
    # For debugging purposes let's print out method name
    print('dct') ; sys.stdout.flush()
    # There are 3 DCT modes: 1, 2, and 3, so let's mod by 3 and add 1
    transform = transform%3 + 1
    # There are two norms ortho or none, so mod them
    norms = [None, 'ortho']
    norm = norms[norm%len(norms)]

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(scipy.fftpack.dct(train, type=transform, norm=norm))
                new_test_list.append(scipy.fftpack.dct(test, type=transform, norm=norm))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                data = scipy.fftpack.dct(data, type=transform, norm=norm)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    #data_labels = np.arange(0, 2*np.pi, 2*np.pi/len(data))
                    data_labels = np.linspace(0, 2*np.pi, len(data.flatten()), endpoint=False)

                    data_labels = np.array(data_labels, dtype='str')
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             data_labels
                                            ))
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1,-1)),
                                                     labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

# Kernel for hanning, hamming
def hanning_kernel(n):
    nVec = np.arange(n)
    return np.cos((2*np.pi*nVec)/(n-1))

# Hanning, hamming
def lag_window_setup(kernel, alpha, beta, signal, lag=True):
    N = signal.shape[1]
    kern = beta * kernel(N)
    w = alpha - kern if lag else alpha + kern
    return {"default": w, "n": N, "lag": lag, "alpha": alpha, "beta": beta}

# Generic window
def window_setup(kernel, signal):
    N = signal.shape[1]
    return {"default": kernel(N), "n": N, "kernel": kernel}

hann_setup = partial(lag_window_setup, hanning_kernel, 0.5, 0.5)

# Hanning, hamming signal operation
def cos_window_func(signal, default, n, lag, alpha, beta): # Could maybe fold this in too
    N = signal.shape[1]
    if N == n: # Use default
        return signal * default
    # print("Non default") - most instances will be default (only use case is differently shaped image batches)
    nVec = np.arange(N)
    if lag:
        w = alpha - beta * np.cos((2*np.pi*nVec)/(N-1))
    else:
        w = alpha + beta * np.cos((2*np.pi*nVec)/(N-1))
    return signal * w

def window_func(signal, default, n, kernel):
    N = signal.shape[1]
    if N == n:
        return signal * default
    nVec = np.arange(N)
    w = kernel(N)
    return signal * w

# Note, we cannot currently extract data-dependent preprocessing, would regress ImagePair if data is inconsistently shaped
window_hann = template(cos_window_func, hann_setup, "myHann", supported_modes=[FEATURES_TO_FEATURES, STREAM_TO_STREAM], input_types=[bool])
window_hann.__doc__ = \
"""Performs a Hann (Hanning) window on a signal with or without "lag"

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES
        lag: whether to include lag

    Returns:
        Data Pair with Hann window feature added

    Raises:
        ValueError
"""

hamming_setup = partial(lag_window_setup, hanning_kernel, .53836, .46164)
window_hamming = template(cos_window_func, hamming_setup, "myHamming", supported_modes=[FEATURES_TO_FEATURES, STREAM_TO_STREAM], input_types=[bool])
window_hamming.__doc__ = \
"""Performs a Hamming window on a signal with or without "lag"
    Args:
        data_pair: given datapair
        lag: whether to include lag
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Hamming window feature added

    Raises:
        ValueError
"""

def window_tukey(data_pair, alpha=0.5, mode=FEATURES_TO_FEATURES):
    """Performs a Tukey window on a signal using a parameter alpha

    An alpha of 0 is a rectangular window
    An alpha of 1 is a Hann window
    Alpha should be between 0 and 1

    Args:
        data_pair: given datapair
        alpha: type of window
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Tukey window feature added

    Raises:
        ValueError
    """
    if alpha > 1:
        alpha = 1
    elif alpha < 0:
        alpha = 0

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                alpha = float(alpha)
                N = train.shape[1]
                w = np.zeros(N)
                for i in np.arange(N):
                    if i < alpha*(N-1)/2:
                        w[i] = 0.5*(1 + np.cos( np.pi * ( (2*i)/(alpha*(N-1)) - 1 ) ) )
                    elif i <= (N-1)*(1-alpha/2):
                        w[i] = 1
                    else:
                        w[i] = 0.5*(1 + np.cos( np.pi * ( (2*i)/(alpha*(N-1)) - 2/alpha + 1 ) ) )
                new_train_list.append(train * w)
                N = train.shape[1]
                w = np.zeros(N)
                for i in np.arange(N):
                    if i < alpha*(N-1)/2:
                        w[i] = 0.5*(1 + np.cos( np.pi * ( (2*i)/(alpha*(N-1)) - 1 ) ) )
                    elif i <= (N-1)*(1-alpha/2):
                        w[i] = 1
                    else:
                        w[i] = 0.5*(1 + np.cos( np.pi * ( (2*i)/(alpha*(N-1)) - 2/alpha + 1 ) ) )
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Tukey window'
                        )
                alpha = float(alpha)
                N = data.shape[1]
                w = np.zeros(N)
                for i in np.arange(N):
                    if i < alpha*(N-1)/2:
                        w[i] = 0.5*(1 + np.cos( np.pi * ( (2*i)/(alpha*(N-1)) - 1 ) ) )
                    elif i <= (N-1)*(1-alpha/2):
                        w[i] = 1
                    else:
                        w[i] = 0.5*(1 + np.cos( np.pi * ( (2*i)/(alpha*(N-1)) - 2/alpha + 1 ) ) )
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

# Kernel for cosine window (aka sine window)
def cosine_kernel(n):
    return np.sin( np.pi*np.arange(n) / (n-1))
cosine_setup = partial(window_setup, cosine_kernel)
window_cosine = template(window_func, cosine_setup, "myCosine", supported_modes=[FEATURES_TO_FEATURES, STREAM_TO_STREAM])
window_cosine.__doc__ = \
"""Performs a Cosine (Sine) window on a signal

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Cosine window feature added

    Raises:
        ValueError
"""

def lanczos_kernel(n):
    return np.sinc(2.0 * np.arange(n) / (n-1) - 1)
lanczos_setup = partial(window_setup, lanczos_kernel)
window_lanczos = template(window_func, lanczos_setup, "myLanczos", supported_modes=[FEATURES_TO_FEATURES, STREAM_TO_STREAM])
window_lanczos.__doc__ = \
"""Performs a Lanczos window on a signal

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Lanczos window feature added

    Raises:
        ValueError
"""
  
def window_triangular(data_pair, mode=FEATURES_TO_FEATURES):
    """Performs the triangular window on a signal

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with triangular window feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        try:
            train_list = data_pair.get_train_batch()
            test_list = data_pair.get_test_batch()
            new_train_list = []
            new_test_list = []
            for train, test in zip(train_list, test_list):
                N = train.shape[1]
                nVec = np.arange(N)
                w = 2.0/(N+1)*( (N+1)/2.0 - np.abs( nVec - (N-1)/2.0 ) )
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = 2.0/(N+1)*( (N+1)/2.0 - np.abs( nVec - (N-1)/2.0 ) )
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Triangular window'
                        )
                N = data.shape[1]
                nVec = np.arange(N)
                w = 2.0/(N+1)*( (N+1)/2.0 - np.abs( nVec - (N-1)/2.0 ) )
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_bartlett(data_pair, mode=FEATURES_TO_FEATURES):
    """Perfoms the Bartlet window on a signal

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Bartlet window feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                N = train.shape[1]
                nVec = np.arange(N)
                w = 2.0/(N-1)*( (N-1)/2.0 - np.abs(nVec - (N-1)/2.0 ) )
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = 2.0/(N-1)*( (N-1)/2.0 - np.abs(nVec - (N-1)/2.0 ) )
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Bartlett window'
                        )
                N = data.shape[1]
                nVec = np.arange(N)
                w = 2.0/(N-1)*( (N-1)/2.0 - np.abs(nVec - (N-1)/2.0 ) )
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_gaussian(data_pair, sigma=1.0, mode=FEATURES_TO_FEATURES):
    """Performs a Gaussian window on a signal given a sigma

    Args:
        data_pair: given datapair
        sigma: sigma value used by gaussian
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Gaussian window feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                N = train.shape[1]
                nVec = np.arange(N)
                w = np.exp( -0.5*( (nVec - (N-1)/2.0 )/( sigma*(N-1)/2.0 ) )**2)
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = np.exp( -0.5*( (nVec - (N-1)/2.0 )/( sigma*(N-1)/2.0 ) )**2)
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Gaussian window'
                        )
                N = data.shape[1]
                nVec = np.arange(N)
                w = np.exp( -0.5*( (nVec - (N-1)/2.0 )/( sigma*(N-1)/2.0 ) )**2 )
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_bartlett_hann(data_pair, mode=FEATURES_TO_FEATURES):
    """Performs a Bartlett-Hann window on a signal

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Bartlett-Hann feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                a0 = 0.62
                a1 = 0.48
                a2 = 0.38
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1 * np.abs( nVec/(N-1) - 0.5 ) - a2 * np.cos( (2*np.pi*nVec)/(N-1) )
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1 * np.abs( nVec/(N-1) - 0.5 ) - a2 * np.cos( (2*np.pi*nVec)/(N-1) )
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Bartlett-Hann window'
                        )
                a0 = 0.62
                a1 = 0.48
                a2 = 0.38
                N = data.shape[1]
                nVec = np.arange(N)
                w = a0 - a1 * np.abs( nVec/(N-1) - 0.5 ) - a2 * np.cos( (2*np.pi*nVec)/(N-1) )
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_blackman(data_pair, alpha=0.16, mode=FEATURES_TO_FEATURES):
    """Performs a Blackman window on a signal given a parameter alpha

    Conventional Blackman sets alpha at 0.16

    Args:
        data_pair: given datapair
        alpha: equation parameter
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Blackman window feature added

    Raises:
        ValueError
    """
    # Not sure on alpha range for this one
    if alpha < 0:
        alpha = 0
    elif alpha > 1:
        alpha = 1

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                a0 = (1-alpha)/2.0
                a1 = 0.5
                a2 = alpha/2.0
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1 * np.cos( (2*np.pi*nVec)/( N-1 ) ) + a2 * np.cos( (4*np.pi*nVec)/( N-1 ) )
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1 * np.cos( (2*np.pi*nVec)/( N-1 ) ) + a2 * np.cos( (4*np.pi*nVec)/( N-1 ) )
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Blackman window'
                        )
                a0 = (1-alpha)/2.0
                a1 = 0.5
                a2 = alpha/2.0
                N = data.shape[1]
                nVec = np.arange(N)
                w = a0 - a1 * np.cos( (2*np.pi*nVec)/( N-1 ) ) + a2 * np.cos( (4*np.pi*nVec)/( N-1 ) )
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_kaiser(data_pair, alpha=3.0, lag=True, mode=FEATURES_TO_FEATURES):
    """Performs a Kaiser window on a signal with or without lag,
    given a parameter alpha. Alpha is usually 3

    Args:
        data_pair: given datapair
        alpha: equation parameter
        lag: whether to include lag
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Kaiser window feature added

    Raises:
        ValueError
    """
    # Not sure on alpha range on this one
    if alpha < 0:
        alpha = 0

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                alpha = float(alpha)
                N = train.shape[1]
                nVec = np.arange(N)
                if lag:
                    w = np.i0(np.pi*alpha*np.sqrt(1 - ( 2.0*nVec/(N-1) - 1)**2) ) / np.i0(np.pi*alpha)
                else:
                    w = np.i0(np.pi*alpha*np.sqrt(1 - ( 2.0*nVec/(N-1) )**2) ) / np.i0(np.pi*alpha)
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                if lag:
                    w = np.i0(np.pi*alpha*np.sqrt(1 - ( 2.0*nVec/(N-1) - 1)**2) ) / np.i0(np.pi*alpha)
                else:
                    w = np.i0(np.pi*alpha*np.sqrt(1 - ( 2.0*nVec/(N-1) )**2) ) / np.i0(np.pi*alpha)
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Kaiser window'
                        )
                alpha = float(alpha)
                N = data.shape[1]
                nVec = np.arange(N)
                if lag:
                    w = np.i0(np.pi*alpha*np.sqrt(1 - ( 2.0*nVec/(N-1) - 1)**2) ) / np.i0(np.pi*alpha)
                else:
                    w = np.i0(np.pi*alpha*np.sqrt(1 - ( 2.0*nVec/(N-1) )**2) ) / np.i0(np.pi*alpha)
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_planck_taper(data_pair, epsilon=0.1, mode=FEATURES_TO_FEATURES):
    """Performs a Planck-taper window on a signal, given a parameter epsilon
    Epsilon is 0.1 on wikipedia

    Args:
        data_pair: given datapair
        epsilon: equation parameter
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Planck-taper window feature added

    Raises:
        ValueError
    """
    # Not sure if this needs to be bounded either
    if epsilon < 0:
        epsilon = 0

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                N = train.shape[1]
                nVec = np.arange(N)
                Zplus = 2.0*epsilon*(1.0/(1+2.0*nVec/(N-1.0)) + 1.0/(1.0-2.0*epsilon+2.0*nVec/(N-1.0)))
                Zminus = 2.0*epsilon*(1.0/(1-2.0*nVec/(N-1.0)) + 1.0/(1.0-2.0*epsilon-2.0*nVec/(N-1.0)))
                w = np.zeros(N)
                for i in np.arange(N):
                    if i < epsilon*(N-1):
                        w[i] = 1.0/(np.exp(Zplus[i]) + 1.0)
                    elif i < (1.0-epsilon)*(N-1.0):
                        w[i] = 1.0
                    elif i <= (N-1.0):
                        w[i] = 1.0/(np.exp(Zminus[i]) + 1.0)
                    else:
                        w[i] = 0.0
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                Zplus = 2.0*epsilon*(1.0/(1+2.0*nVec/(N-1.0)) + 1.0/(1.0-2.0*epsilon+2.0*nVec/(N-1.0)))
                Zminus = 2.0*epsilon*(1.0/(1-2.0*nVec/(N-1.0)) + 1.0/(1.0-2.0*epsilon-2.0*nVec/(N-1.0)))
                w = np.zeros(N)
                for i in np.arange(N):
                    if i < epsilon*(N-1):
                        w[i] = 1.0/(np.exp(Zplus[i]) + 1.0)
                    elif i < (1.0-epsilon)*(N-1.0):
                        w[i] = 1.0
                    elif i <= (N-1.0):
                        w[i] = 1.0/(np.exp(Zminus[i]) + 1.0)
                    else:
                        w[i] = 0.0
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Planck-taper window'
                        )
                N = data.shape[1]
                nVec = np.arange(N)
                Zplus = 2.0*epsilon*(1.0/(1+2.0*nVec/(N-1.0)) + 1.0/(1.0-2.0*epsilon+2.0*nVec/(N-1.0)))
                Zminus = 2.0*epsilon*(1.0/(1-2.0*nVec/(N-1.0)) + 1.0/(1.0-2.0*epsilon-2.0*nVec/(N-1.0)))
                w = np.zeros(N)
                for i in np.arange(N):
                    if i < epsilon*(N-1):
                        w[i] = 1.0/(np.exp(Zplus[i]) + 1.0)
                    elif i < (1.0-epsilon)*(N-1.0):
                        w[i] = 1.0
                    elif i <= (N-1.0):
                        w[i] = 1.0/(np.exp(Zminus[i]) + 1.0)
                    else:
                        w[i] = 0.0
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_nuttall(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a Nuttall window on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Nuttall window feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                a0 = 0.355768
                a1 = 0.487396
                a2 = 0.144232
                a3 = 0.012604
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Nuttall Window'
                        )
                a0 = 0.355768
                a1 = 0.487396
                a2 = 0.144232
                a3 = 0.012604
                N = data.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_blackman_harris(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a Blackman-Harris window on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Blackman-Harris window feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                a0 = 0.35875
                a1 = 0.48829
                a2 = 0.14128
                a3 = 0.01168
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Blackman-Harris window'
                        )
                a0 = 0.35875
                a1 = 0.48829
                a2 = 0.14128
                a3 = 0.01168
                N = data.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_blackman_nuttall(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a Blackman-Nuttall window on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with Blackman-Nuttall window feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        try:
            train_list = data_pair.get_train_batch()
            test_list = data_pair.get_test_batch()
            new_train_list = []
            new_test_list = []
            for train, test in zip(train_list, test_list):
                a0 = 0.3635819
                a1 = 0.4891775
                a2 = 0.1365995
                a3 = 0.0106411
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Blackman-Nuttall Window'
                        )
                a0 = 0.3635819
                a1 = 0.4891775
                a2 = 0.1365995
                a3 = 0.0106411
                N = data.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0))
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)


            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def window_flat_top(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a flat top window on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with flat top window feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                a0 = 1.0
                a1 = 1.93
                a2 = 1.29
                a3 = 0.388
                a4 = 0.028
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0)) + a4*np.cos(8.0*np.pi*nVec/(N-1.0))
                new_train_list.append(train * w)
                N = train.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0)) + a4*np.cos(8.0*np.pi*nVec/(N-1.0))
                new_test_list.append(test * w)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Window Flat Top'
                        )
                a0 = 1.0
                a1 = 1.93
                a2 = 1.29
                a3 = 0.388
                a4 = 0.028
                N = data.shape[1]
                nVec = np.arange(N)
                w = a0 - a1*np.cos(2.0*np.pi*nVec/(N-1.0)) + a2*np.cos(4.0*np.pi*nVec/(N-1.0)) - a3*np.cos(6.0*np.pi*nVec/(N-1.0)) + a4*np.cos(8.0*np.pi*nVec/(N-1.0))
                data = data * w

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)


            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def sub_sample_data(data_pair, window_size, step_size, mode=FEATURES_TO_FEATURES):
    """Sub-sample the data

    Args:
        data_pair: given datapair
        window_size: size of the window
        step_size: size of the steps
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Sub sample Data Pair

    Raises:
        ValueError
    """
    # Just a couple of checks on the inputs
    window_size = abs(window_size)
    step_size = abs(step_size)
    if window_size < 1:
        window_size = 1
    if step_size < 1:
        step_size = 1
    if isinstance(data_pair, GTMOEPImagePair):
        raise ValueError(
            'No Features to Features available for sub sampling'
            )
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())
            new_instances = []
            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Features to Features available for sub sampling'
                        )
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                    data_labels = instance.get_stream().get_labels()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for sub sampling'
                        )

                N = data.shape[1]
                if window_size > N:
                    window_size = N
                if step_size > N:
                    step_size = N
                i = 0
                while i <= N - window_size:
                    new_stream_data = StreamData()
                    new_stream_data.set_data(data[:,i:i+window_size],
                                      labels=data_labels[i:i+window_size])
                    #new_stream_data.set_target(instance.get_stream().get_target())

                    new_feature_data = FeatureData()
                    #new_feature_data.set_target(instance.get_features().get_target())

                    new_instance = GTMOEPDataInstance(features=new_feature_data,
                                            stream=new_stream_data,
                                            target=instance.get_target())
                    new_instances.append(new_instance)
                    i += step_size

            new_data_set = GTMOEPData(new_instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def cut_data_lead(data_pair, samples_to_cut=100, mode=FEATURES_TO_FEATURES):
    """Cut samples_to_cut from the front of the data

    Args:
        data_pair: given datapair
        samples_to_cut: number of samples to cut
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Cut Data Pair

    Raises:
        ValueError
    """
    # Convert to an int to prevent deprication warnings
    samples_to_cut = int(samples_to_cut)
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(train[:, samples_to_cut:])
                new_test_list.append(test[:, samples_to_cut:])


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                    data_labels = instance.get_features().get_labels()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                    data_labels = instance.get_stream().get_labels()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Cut Data Lead'
                        )

                data = data[:, samples_to_cut:]
                data_labels = data_labels[samples_to_cut:]

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data, labels=data_labels)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data, labels=data_labels)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


# def split_data_lead(data_pair, samples_to_cut=100, mode=FEATURES_TO_FEATURES):
#     """Cut samples_to_cut from the front of the data
#
#     Args:
#         data_pair: given datapair
#         samples_to_cut: number of samples to cut
#         mode: iteration mode, options: FEATURES_TO_FEATURES,
#         STREAM_TO_STREAM, or STREAM_TO_FEATURES
#
#     Returns:
#         Cut Data Pair
#
#     Raises:
#         ValueError
#     """
#     # Convert to an int to prevent deprication warnings
#     samples_to_cut = int(samples_to_cut)
#     if isinstance(data_pair, GTMOEPImagePair):
#         train_list = data_pair.get_train_batch()
#         test_list = data_pair.get_test_batch()
#         new_train_list = []
#         new_test_list = []
#         try:
#             for train, test in zip(train_list, test_list):
#                 new_train_list.append(train[:, samples_to_cut:])
#                 new_test_list.append(test[:, samples_to_cut:])
#
#
#             data_pair.set_train_batch(new_train_list)
#             data_pair.set_test_batch(new_test_list)
#         except:
#             data_pair.clear()
#             raise
#     else:
#         data_list = []
#         for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
#             instances = cp.deepcopy(data_set.get_instances())
#
#             for instance in instances:
#                 if mode is FEATURES_TO_FEATURES:
#                     data = instance.get_features().get_data()
#                     data_labels = instance.get_features().get_labels()
#                 elif mode is STREAM_TO_STREAM:
#                     data = instance.get_stream().get_data()
#                     data_labels = instance.get_stream().get_labels()
#                 elif mode is STREAM_TO_FEATURES:
#                     raise ValueError(
#                         'No Stream to Features available for Cut Data Lead'
#                         )
#
#                 data = data[:, samples_to_cut:]
#                 data_labels = data_labels[samples_to_cut:]
#
#                 if mode is FEATURES_TO_FEATURES:
#                     instance.get_features().set_data(data, labels=data_labels)
#                 elif mode is STREAM_TO_STREAM:
#                     instance.get_stream().set_data(data, labels=data_labels)
#
#             new_data_set = GTMOEPData(instances)
#             data_list.append(new_data_set)
#         data_pair = GTMOEPDataPair(train_data=data_list[0],
#                                        test_data=data_list[1])
#
#     gc.collect(); return data_pair


def my_dwt(data_pair, mode=FEATURES_TO_FEATURES, axis=-1):
    """Perform a dwt on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Modified Data Pair
    """
    if isinstance(data_pair, GTMOEPImagePair):
        try:
            train_list = data_pair.get_train_batch()
            test_list = data_pair.get_test_batch()
            new_train_list = []
            new_test_list = []
            for train, test in zip(train_list, test_list):
                coeff_list = pywt.wavedec(train, 'db3')
                vec = []
                [vec.extend(level) for level in coeff_list]
                new_train_list.append(np.array(vec))
                coeff_list = pywt.wavedec(test, 'db3')
                vec = []
                [vec.extend(level) for level in coeff_list]
                new_test_list.append(np.array(vec))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                coeff_list = pywt.wavedec(data, 'db3', axis=axis)
                vec = []
                [vec.extend(level) for level in coeff_list]
                data = np.array(vec)
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    #data_labels = np.arange(0, 2*np.pi, 2*np.pi/len(data))
                    data_labels = np.linspace(0, 2*np.pi, len(data.flatten()), endpoint=False)

                    data_labels = np.array(data_labels, dtype='str')
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             data_labels
                                            ))
                    instance.get_features().set_data(
                                                    np.reshape(new_features, (1,-1)),
                                                    labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)

            data_pair = GTMOEPDataPair(train_data=data_list[0],
                                            test_data=data_list[1])

    gc.collect(); return data_pair


def my_rms_2d(data_pair, axis=0, mode=STREAM_TO_FEATURES):
    """Computes rms of 2D data

    Args:
        data_pair: given datapair
        axis: id of which axis to use
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with rms feature added

    Raises:
        ValueError
    """
    axis_options = [0, 1]
    axis = axis_options[axis%len(axis_options)]
    if isinstance(data_pair, GTMOEPImagePair):
        raise ValueError(
            'No Features to Features available for 2D RMS'
            )
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Features to Features available for 2D RMS'
                        )
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                data = np.array([np.linalg.norm(data, 2, axis=axis)])
               # if mode is FEATURES_TO_FEATURES:
               #     data_labels = np.array([str(p)+'-Norm' + str(len(data.flatten()))])
               #     instance.get_features().set_data(data, labels=data_labels)
                if mode is STREAM_TO_STREAM:
                    data_labels = np.array(['rms2d-' + str(i) for i in np.arange(data.shape[1])])
                    instance.get_stream().set_data(data, labels=data_labels)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    data_labels = np.array(['rms2d-' + str(i) for i in np.arange(len(new_features))])
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             data_labels
                                            ))
                    instance.get_features().set_data(
                                                    np.reshape(new_features, (1,-1)),
                                                    labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_norm(data_pair, p=2, mode=STREAM_TO_FEATURES):
    """Computes the p norm on a dataset

    Args:
        data_pair: given datapair
        p: value of p
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with p norm feature added
    """
    if p < 1:
        p = 1
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.array([[np.linalg.norm(train, p)]]))
                new_test_list.append(np.array([[np.linalg.norm(test, p)]]))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                data = np.array([[np.linalg.norm(data, p)]])

                if mode is FEATURES_TO_FEATURES:
                    data_labels = np.array([str(p)+'-Norm' + str(len(data.flatten()))])
                    instance.get_features().set_data(data, labels=data_labels)
                elif mode is STREAM_TO_STREAM:
                    data_labels = np.array([str(p)+'-Norm' + str(len(data.flatten()))])
                    instance.get_stream().set_data(data, labels=data_labels)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    data_labels = np.array([str(p)+'-Norm' + str(len(new_features))])
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             data_labels
                                            ))
                    instance.get_features().set_data(
                                                    np.reshape(new_features, (1,-1)),
                                                    labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_sum(data_pair, mode=FEATURES_TO_FEATURES):
    """Sum a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with sum feature added
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.array([[np.sum(train)]]))
                new_test_list.append(np.array([[np.sum(test)]]))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                data = np.array([[np.sum(data)]])
                data_labels = np.array(['Sum'])

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data, labels=data_labels)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data, labels=data_labels)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             data_labels
                                            ))
                    instance.get_features().set_data(
                                                    np.reshape(new_features, (1,-1)),
                                                    labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)

        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_prod(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform an fft on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with prod feature added
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.array([[np.prod(train)]]))
                new_test_list.append(np.array([[np.prod(test)]]))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                data = np.array([[np.prod(data)]])
                data_labels = np.array(['Prod'])

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data, labels=data_labels)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data, labels=data_labels)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             data_labels
                                            ))
                    instance.get_features().set_data(
                                                    np.reshape(new_features, (1,-1)),
                                                    labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_cum_prod(data_pair, mode=FEATURES_TO_FEATURES):
    """Computes the cumulative product on each point in a dataset.

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with cumulative product feature added
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.cumprod(train, axis=-1))
                new_test_list.append(np.cumprod(test, axis=-1))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()
                data = np.cumprod(data, axis=-1)
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    labels = np.array(['CumProd' + str(i) for i in range(len(data.flatten()))])

                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             labels
                                            ))
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1,-1)),
                                                     labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_cum_sum(data_pair, mode=FEATURES_TO_FEATURES):
    """Computes the cumulative sum on each point in a dataset.

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with cumulative sum feature added
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.cumsum(train, axis=-1))
                new_test_list.append(np.cumsum(test, axis=-1))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                data = np.cumsum(data, axis=-1)
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    labels = np.array(['CumSum' + str(i) for i in range(len(data.flatten()))])
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             labels
                                            ))
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1,-1)),
                                                     labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_fft(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform an fft on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with fft feature added
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.abs(np.fft.fft(train)))
                new_test_list.append(np.abs(np.fft.fft(test)))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                data = np.abs(np.fft.fft(data))

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    #print data.flatten().shape, old_features.flatten().shape
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    data_labels = np.linspace(0, 2*np.pi, len(data.flatten()), endpoint=False)
                    data_labels = np.array(data_labels, dtype='str')
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             data_labels
                                            ))
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1,-1)),
                                                     labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_abs(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform an absolute value on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with absolute value feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.abs(train))
                new_test_list.append(np.abs(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for abs'
                        )
                data = np.abs(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_exp(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform an exponential on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with exponential feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.exp(train))
                new_test_list.append(np.exp(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Exponential'
                        )
                data = np.exp(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_tangent(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a tangent on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with tangent feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.tan(train))
                new_test_list.append(np.tan(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Tangent'
                        )
                data = np.tan(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_cosine(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a cosine on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with cosine feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.cos(train))
                new_test_list.append(np.cos(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Cosine'
                        )
                data = np.cos(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_sine(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a sine on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with sine feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.sin(train))
                new_test_list.append(np.sin(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Sine'
                        )
                data = np.sin(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_arctangent(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform an arctangent on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with arctangent feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.arctan(train))
                new_test_list.append(np.arctan(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Arctangent'
                        )
                data = np.arctan(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_arccosine(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform an arccosine on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with arccosine feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.arccos(train))
                new_test_list.append(np.arccos(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Arccosine'
                        )
                data = np.arccos(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_arcsine(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform an arcsine on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with arcsine feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.arcsin(train))
                new_test_list.append(np.arcsin(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for arcsine'
                        )
                data = np.arcsin(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_log(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a natural log on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with natural log feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.log(train))
                new_test_list.append(np.log(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Log'
                        )
                data = np.log(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_round(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a round on a dataset

    Args:
        datapair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with round feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(np.round(train))
                new_test_list.append(np.round(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Round'
                        )

                data = np.round(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_kalman_filter(data_pair, Q=1e-5, R=0.1**2, mode=FEATURES_TO_FEATURES):
    """Perform a kalman filter on a dataset

    Args:
        data_pair: given datapair
        Q: equation paramater
        R: equation parameter
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with kalman filter feature added

    Raises:
        ValueError
    """
    Q = np.abs(Q)
    R = np.abs(R)
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                N = train.shape[1]
                new_data = []
                for row in data:
                    xhat = np.zeros(N)          # A posteri estimate of x
                    P = np.zeros(N)             # A posteri error estimate
                    xhatminus = np.zeros(N)     # A priori estimate of x
                    Pminus = np.zeros(N)        # A priori error estimate
                    K = np.zeros(N)             # gain or blending factor
                    xhat[0] = row[0]
                    P[0] = 1.0                  # What should this be?
                    for k in range(1, N):
                        xhatminus[k] = xhat[k-1]
                        Pminus[k] = P[k-1] + Q

                        K[k] = Pminus[k]/(Pminus[k] + R)
                        xhat[k] = xhatminus[k] + K[k]*(row[k]-xhatminus[k])
                        P[k] = (1 - K[k])*Pminus[k]

                    new_data.append(xhat)
                new_train_list.append(np.array(new_data))
                N = test.shape[1]
                new_data = []
                for row in data:
                    xhat = np.zeros(N)          # A posteri estimate of x
                    P = np.zeros(N)             # A posteri error estimate
                    xhatminus = np.zeros(N)     # A priori estimate of x
                    Pminus = np.zeros(N)        # A priori error estimate
                    K = np.zeros(N)             # gain or blending factor
                    xhat[0] = row[0]
                    P[0] = 1.0                  # What should this be?
                    for k in range(1, N):
                        xhatminus[k] = xhat[k-1]
                        Pminus[k] = P[k-1] + Q

                        K[k] = Pminus[k]/(Pminus[k] + R)
                        xhat[k] = xhatminus[k] + K[k]*(row[k]-xhatminus[k])
                        P[k] = (1 - K[k])*Pminus[k]

                    new_data.append(xhat)
                new_test_list.append(np.array(new_data))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Kalman Filter'
                        )

                N = data.shape[1]
                new_data = []
                for row in data:
                    xhat = np.zeros(N)          # A posteri estimate of x
                    P = np.zeros(N)             # A posteri error estimate
                    xhatminus = np.zeros(N)     # A priori estimate of x
                    Pminus = np.zeros(N)        # A priori error estimate
                    K = np.zeros(N)             # gain or blending factor
                    xhat[0] = row[0]
                    P[0] = 1.0                  # What should this be?
                    for k in range(1, N):
                        xhatminus[k] = xhat[k-1]
                        Pminus[k] = P[k-1] + Q

                        K[k] = Pminus[k]/(Pminus[k] + R)
                        xhat[k] = xhatminus[k] + K[k]*(row[k]-xhatminus[k])
                        P[k] = (1 - K[k])*Pminus[k]

                    new_data.append(xhat)
                data = np.array(new_data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_linear_predictive_coding(data_pair, p=5, mode=STREAM_TO_FEATURES):
    """Computes the LPC coefficients

    Args:
        data_pair: given datapair
        p: equation parameter
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with LPC coefficients feature added
    """
    p = np.abs(p)
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                rows = []
                for row in train:
                    F = np.fft.fft(np.array([row]))
                    S = np.conjugate(F)*F
                    R = np.fft.ifft(S)
                    R = np.real(R)
                    A = np.zeros((int(p), int(p)))
                    for i in range(0, p):
                        for j in range(0, p):
                            A[i, j] = R[0, np.abs(i-j)]
                    colR = np.transpose(np.reshape(R[0, 1:p+1], (1, -1)))
                    coeffs = np.linalg.inv(A)*colR
                    coeffs = np.transpose(coeffs)
                    row = coeffs[0, :]
                    rows.extend(row)
                new_train_list.append(np.array([rows]))
                rows = []
                for row in test:
                    F = np.fft.fft(np.array([row]))
                    S = np.conjugate(F)*F
                    R = np.fft.ifft(S)
                    R = np.real(R)
                    A = np.zeros((int(p), int(p)))
                    for i in range(0, p):
                        for j in range(0, p):
                            A[i, j] = R[0, np.abs(i-j)]
                    colR = np.transpose(np.reshape(R[0, 1:p+1], (1, -1)))
                    coeffs = np.linalg.inv(A)*colR
                    coeffs = np.transpose(coeffs)
                    row = coeffs[0, :]
                    rows.extend(row)
                new_test_list.append(np.array([rows]))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_FEATURES or STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()
                rows = []
                for row in data:
                    F = np.fft.fft(np.array([row]))
                    S = np.conjugate(F)*F
                    R = np.fft.ifft(S)
                    R = np.real(R)
                    A = np.zeros((int(p), int(p)))
                    for i in range(0, p):
                        for j in range(0, p):
                            A[i, j] = R[0, np.abs(i-j)]
                    colR = np.transpose(np.reshape(R[0, 1:p+1], (1, -1)))
                    coeffs = np.linalg.inv(A)*colR
                    coeffs = np.transpose(coeffs)
                    row = coeffs[0, :]
                    rows.extend(row)
                data = np.array([rows])
                labels = np.array(['LPC' + str(i) for i in range(data.shape[1])])
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data, labels=labels)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data, labels=labels)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             labels
                                            ))
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1,-1)),
                                                     labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_wiener_filter(data_pair, mode=FEATURES_TO_FEATURES):
    """Perform a wiener on a dataset

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with wiener feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(scipy.signal.wiener(train))
                new_test_list.append(scipy.signal.wiener(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Wiener Filter'
                        )
                data = scipy.signal.wiener(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_savitzky_golay_filter(data_pair, N=11, M=3, deriv=0, mode=FEATURES_TO_FEATURES):
    """Computes the Savitzky-Golay filter on the data

    Args:
        data_pair: given datapair
        N: inner method paramater
        M: inner method parameter
        deriv: inner method parameter
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with wiener feature added

    Raises:
        ValueError
    """
    N = np.abs(N)
    M = np.abs(M)
    deriv = np.abs(deriv)
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                rows = []
                for row in train:
                    rows.append(savitzky_golay(row, N, M, deriv))
                new_train_list.append(np.array(rows))
                rows = []
                for row in test:
                    rows.append(savitzky_golay(row, N, M, deriv))
                new_test_list.append(np.array(rows))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Savitzky-Golay'
                        )
                rows = []
                for row in data:
                    rows.append(savitzky_golay(row, N, M, deriv))
                data = np.array(rows)
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def gaussian_hmm(data_pair, n_components=2, algorithm='viterbi'):
    """Implements a Gaussian hidden markov model

    Args:
        data_pair: given datapair
        n_components: number of states
        algorithm: decoder algorithm

    Returns:
        Data Pair transformed with hidden markov model
    """
    # Fix to remove warning from boolean
    n_components = int(n_components)

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            my_hmm = hmmlearn.hmm.GaussianHMM(n_components = n_components,algorithm=algorithm)
            my_hmm.fit(train_list)
            for train, test in zip(train_list, test_list):
                new_train_list.append(my_hmm.predict(train))
                new_test_list.append(my_hmm.predict(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_pair = cp.deepcopy(data_pair)
        training_data = data_pair.get_train_data().get_numpy()
        target_values = np.array([inst.get_target()[0] for
                                  inst in data_pair.get_train_data().get_instances()])
        my_hmm = hmmlearn.hmm.GaussianHMM(n_components = n_components,algorithm=algorithm)
        my_hmm.fit([training_data])

        testing_data = data_pair.get_test_data().get_numpy()
        predicted_classes = my_hmm.predict(testing_data)
        [inst.set_target([target]) for inst, target in
            zip(data_pair.get_test_data().get_instances(), predicted_classes)]

        # Set the self-predictions of the training data
        trained_classes = my_hmm.predict(training_data)
        [inst.set_target([target]) for inst, target in
            zip(data_pair.get_train_data().get_instances(), trained_classes)]

        data_pair = makeFeatureFromClass(data_pair, name="HMM")

    gc.collect(); return data_pair

def my_pca(data_pair, n_components=None, whiten=False):
    """Implements scikit's pca method

    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Args:
        data_pair: given datapair
        n_components: number of components to keep
        whiten: When True (False by default) the components_ vectors are
        multiplied by the square root of n_samples and then divided by the
        singular values to ensure uncorrelated outputs with unit
        component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by making
        their data respect some hard-wired assumptions. --sklearn

    Returns:
        Data Pair transformed with pca
    """
    # Convert n_components to int in case a boolean is received
    n_components = int(n_components)
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            my_pca_transform = sklearn.decomposition.PCA(n_components=n_components,
                                whiten=whiten)
            my_pca_transform.fit(train_list)
            for train, test in zip(train_list, test_list):
                new_train_list.append(my_pca_transform.predict(train))
                new_test_list.append(my_pca_transform.predict(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_pair = cp.deepcopy(data_pair)
        training_data = data_pair.get_train_data().get_numpy()
        #target_values = np.array([inst.get_features().get_target()[0] for
        #                   inst in data_pair.get_train_data().get_instances()])
        my_pca_transform = sklearn.decomposition.PCA(n_components=n_components,
                            whiten=whiten)
        testing_data = data_pair.get_test_data().get_numpy()

        new_training_data = my_pca_transform.fit_transform(training_data)
        # For debugging purposes let's print out method name
        print('pca'), new_training_data.shape; sys.stdout.flush()
        new_testing_data = my_pca_transform.transform(testing_data)
        labels = np.array(['PCA' + str(i)
            for i in range(new_testing_data.shape[1])])
        for old_instance, new_instance in zip(
            data_pair.get_train_data().get_instances(),
            new_training_data):

            old_instance.get_features().set_data(np.reshape(new_instance,(1,-1)), labels=labels)

        for old_instance, new_instance in zip(
            data_pair.get_test_data().get_instances(),
            new_testing_data):

            old_instance.get_features().set_data(np.reshape(new_instance,(1,-1)), labels=labels)

    gc.collect(); return data_pair

def my_sparse_pca(data_pair, n_components=None, alpha=1):
    """Implements scikit's sparse pca method

    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA

    Args:
        data_pair: given datapair
        n_components: number of components to keep

    Returns:
        Data Pair transformed with sparse pca
    """
    # Convert n_components to int in case a boolean is received
    n_components = int(n_components)
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            my_spca_transform = sklearn.decomposition.SparsePCA(n_components=n_components,
                                alpha=alpha)
            my_spca_transform.fit(train_list)
            for train, test in zip(train_list, test_list):
                new_train_list.append(my_spca_transform.predict(train))
                new_test_list.append(my_spca_transform.predict(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_pair = cp.deepcopy(data_pair)
        training_data = data_pair.get_train_data().get_numpy()
        #target_values = np.array([inst.get_features().get_target()[0] for
        #                   inst in data_pair.get_train_data().get_instances()])
        my_spca_transform = sklearn.decomposition.SparsePCA(n_components=n_components,
                            alpha=alpha)
        testing_data = data_pair.get_test_data().get_numpy()

        new_training_data = my_spca_transform.fit_transform(training_data)
        # For debugging purposes let's print out method name
        print('spca'), new_training_data.shape; sys.stdout.flush()
        new_testing_data = my_spca_transform.transform(testing_data)
        labels = np.array(['SparsePCA' + str(i)
            for i in range(new_testing_data.shape[1])])
        for old_instance, new_instance in zip(
            data_pair.get_train_data().get_instances(),
            new_training_data):

            old_instance.get_features().set_data(np.reshape(new_instance,(1,-1)), labels=labels)

        for old_instance, new_instance in zip(
            data_pair.get_test_data().get_instances(),
            new_testing_data):

            old_instance.get_features().set_data(np.reshape(new_instance,(1,-1)), labels=labels)

    gc.collect(); return data_pair

def my_ica(data_pair, n_components=None, whiten=True):
    """Implements scikit's ica method

    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

    Args:
        data_pair: given datapair
        n_components: number of components to keep
        whiten: When True (True by default) the components_ vectors are
        multiplied by the square root of n_samples and then divided by the
        singular values to ensure uncorrelated outputs with unit
        component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by making
        their data respect some hard-wired assumptions. --sklearn

    Returns:
        Data Pair transformed with ica
    """
    # Convert n_components to int in case a boolean is received
    n_components = int(n_components)
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            my_ica_transform = sklearn.decomposition.FastICA(n_components=n_components,
                                whiten=whiten)
            my_ica_transform.fit(train_list)
            for train, test in zip(train_list, test_list):
                new_train_list.append(my_ica_transform.predict(train))
                new_test_list.append(my_ica_transform.predict(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_pair = cp.deepcopy(data_pair)
        training_data = data_pair.get_train_data().get_numpy()
        #target_values = np.array([inst.get_features().get_target()[0] for
        #                   inst in data_pair.get_train_data().get_instances()])
        my_ica_transform = sklearn.decomposition.FastICA(n_components=n_components,
                            whiten=whiten)
        testing_data = data_pair.get_test_data().get_numpy()

        new_training_data = my_ica_transform.fit_transform(training_data)
        # For debugging purposes let's print out method name
        print('Ica'), new_training_data.shape; sys.stdout.flush()
        new_testing_data = my_ica_transform.transform(testing_data)
        labels = np.array(['ICA' + str(i)
            for i in range(new_testing_data.shape[1])])
        for old_instance, new_instance in zip(
            data_pair.get_train_data().get_instances(),
            new_training_data):

            old_instance.get_features().set_data(np.reshape(new_instance,(1,-1)), labels=labels)

        for old_instance, new_instance in zip(
            data_pair.get_test_data().get_instances(),
            new_testing_data):

            old_instance.get_features().set_data(np.reshape(new_instance,(1,-1)), labels=labels)

    gc.collect(); return data_pair

def my_spectral_embedding(data_pair, n_components=2):
    """Implements scikit's spectral embedding method

    http://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html

    Args:
        data_pair: given datapair
        n_components: the dimension of the projected subspace

    Returns:
        Data Pair transformed with spectral embedding
    """
    # Convert n_components to int in case a boolean is received
    n_components = int(n_components)
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            my_ica_transform = sklearn.decomposition.FastICA(n_components=n_components)
            my_ica_transform.fit(train_list)
            for train, test in zip(train_list, test_list):
                new_train_list.append(my_ica_transform.predict(train))
                new_test_list.append(my_ica_transform.predict(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_pair = cp.deepcopy(data_pair)
        training_data = data_pair.get_train_data().get_numpy()
        #target_values = np.array([inst.get_features().get_target()[0] for
        #                   inst in data_pair.get_train_data().get_instances()])
        my_spectral_transform = SpectralEmbedding(n_components=n_components)
        testing_data = data_pair.get_test_data().get_numpy()

        new_training_data = my_spectral_transform.fit_transform(training_data)
        # For debugging purposes let's print out method name
        print('spectral embedding'), new_training_data.shape; sys.stdout.flush()
        new_testing_data = my_spectral_transform.fit_transform(testing_data)
        labels = np.array(['Spectral' + str(i)
            for i in range(new_testing_data.shape[1])])
        for old_instance, new_instance in zip(
            data_pair.get_train_data().get_instances(),
            new_training_data):

            old_instance.get_features().set_data(np.reshape(new_instance,(1,-1)), labels=labels)

        for old_instance, new_instance in zip(
            data_pair.get_test_data().get_instances(),
            new_testing_data):

            old_instance.get_features().set_data(np.reshape(new_instance,(1,-1)), labels=labels)

    gc.collect(); return data_pair

def my_ecdf(data_pair, n_components=10, mode=STREAM_TO_FEATURES):
    """Implements ECDF method

    Args:
        data_pair: given datapair
        n_components: the dimension of projected space
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair transformed with ECDF
    """
    # Convert n_components to int in case a boolean is received
    n_components = int(n_components)

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(ecdfRep(train, n_components))
                new_test_list.append(ecdfRep(test, n_components))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()

                data = ecdfRep(data, n_components)

                labels = np.array(['LPC' + str(i) for i in range(data.shape[1])])
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data, labels=labels)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data, labels=labels)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             labels
                                            ))
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1,-1)),
                                                     labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_rebase(data_pair, mode=STREAM_TO_FEATURES):
    """Subtracts min from signal

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with rebase feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                new_train_list.append(train - np.min(train))
                new_test_list.append(test - np.min(test))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    #data = cp.deepcopy(instance.get_features().get_data())
                    data = instance.get_features().get_data()
                    #labels = cp.deepcopy(instance.get_features().get_labels())
                    labels = instance.get_features().get_labels()
                elif mode is STREAM_TO_STREAM:
                    #data = cp.deepcopy(instance.get_stream().get_data())
                    data = instance.get_stream().get_data()
                    #labels = cp.deepcopy(instance.get_stream().get_labels())
                    labels = instance.get_stream().get_labels()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Rebase'
                        )

                data = data - np.min(data)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data, labels=labels)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data, labels=labels)


            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_peak_finder(data_pair, start_delta=5, lookForMax=1, mode=STREAM_TO_FEATURES):
    """Zero out all but the peaks in the data

    delta is the amount to withdraw before locking

    Args:
        data_pair: given datapair
        start_delta: initial value of delta
        lookForMax: max parameter
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with peak finder feature added
    """
    if isinstance(data_pair, GTMOEPImagePair):
        # print("WHER")
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                rows = []
                for row in train:
                    min_val, max_val = np.inf, -np.inf
                    min_pos, max_pos = -1, 1
                    min_locs = []
                    max_locs = []
                    delta = start_delta
                    for i in np.arange(len(row)):
                        val = row[i]
                        if val > max_val:
                            max_val = val
                            max_pos = i
                        if val < min_val:
                            min_val = val
                            min_pos = i

                        if lookForMax:
                            if val < (max_val - delta):
                                max_locs += [max_pos]
                                min_val = val
                                min_pos = i
                                lookForMax = 0
                        else:
                            if val > (min_val + delta):
                                min_locs += [min_pos]
                                max_val = val
                                max_pos = i
                                lookForMax = 1
                    peak_locs = max_locs # concat two lists
                    not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
                    row[not_peaks] = 0
                    rows.append(row)
                new_train_list.append(np.array(rows))
                rows = []
                for row in test:
                    min_val, max_val = np.inf, -np.inf
                    min_pos, max_pos = -1, 1
                    min_locs = []
                    max_locs = []
                    delta = start_delta
                    for i in np.arange(len(row)):
                        val = row[i]
                        if val > max_val:
                            max_val = val
                            max_pos = i
                        if val < min_val:
                            min_val = val
                            min_pos = i

                        if lookForMax:
                            if val < (max_val - delta):
                                max_locs += [max_pos]
                                min_val = val
                                min_pos = i
                                lookForMax = 0
                        else:
                            if val > (min_val + delta):
                                min_locs += [min_pos]
                                max_val = val
                                max_pos = i
                                lookForMax = 1
                    peak_locs = max_locs # concat two lists
                    not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
                    row[not_peaks] = 0
                    rows.append(row)
                new_test_list.append(np.array(rows))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        # print("HOWHOW")
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    #data = cp.deepcopy(instance.get_features().get_data())
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_FEATURES or STREAM_TO_FEATURES:
                    #data = cp.deepcopy(instance.get_stream().get_data())
                    data = instance.get_stream().get_data()
                rows = []
                for row in data:
                    min_val, max_val = np.inf, -np.inf
                    min_pos, max_pos = -1, 1
                    min_locs = []
                    max_locs = []
                    delta = start_delta
                    for i in np.arange(len(row)):
                        val = row[i]
                        if val > max_val:
                            max_val = val
                            max_pos = i
                        if val < min_val:
                            min_val = val
                            min_pos = i

                        if lookForMax:
                            if val < (max_val - delta):
                                max_locs += [max_pos]
                                min_val = val
                                min_pos = i
                                lookForMax = 0
                        else:
                            if val > (min_val + delta):
                                min_locs += [min_pos]
                                max_val = val
                                max_pos = i
                                lookForMax = 1
                        #delta = delta*0.9999
                    #print minLocs, maxLocs
                    #peak_locs = min_locs + max_locs # concat two lists
                    #print delta
                    peak_locs = max_locs # concat two lists
                    not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
                    #print row
                    row[not_peaks] = 0
                    rows.append(row)
                data = np.array(rows)
                # print(data)
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    labels = np.array(['Peaks' + str(i) for i in range(len(data.flatten()))])
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             labels
                                            ))
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1, -1)),
                                                     labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_peak_finder_2(data_pair, start_delta=5, lookForMax=1, mode=STREAM_TO_FEATURES):
    """Zero out all but the peaks in the data

    delta is the amount to withdraw before locking

    Args:
        data_pair: given datapair
        start_delta: initial value of delta
        lookForMax: max parameter
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with peak finder feature added
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                rows = []
                for row in train:
                    min_val, max_val = np.inf, -np.inf
                    min_pos, max_pos = -1, 1
                    min_locs = []
                    max_locs = []
                    delta = start_delta
                    for i in np.arange(len(row)):
                        val = row[i]
                        if val > max_val:
                            max_val = val
                            max_pos = i
                        if val < min_val:
                            min_val = val
                            min_pos = i

                        if lookForMax:
                            if val < (max_val - delta):
                                max_locs += [max_pos]
                                min_val = val
                                min_pos = i
                                lookForMax = 0
                        else:
                            if val > (min_val + delta):
                                min_locs += [min_pos]
                                max_val = val
                                max_pos = i
                                lookForMax = 1
                    peak_locs = max_locs # concat two lists
                    not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
                    row[not_peaks] = 0
                    rows.append(row)
                new_train_list.append(np.array(rows))
                rows = []
                for row in test:
                    min_val, max_val = np.inf, -np.inf
                    min_pos, max_pos = -1, 1
                    min_locs = []
                    max_locs = []
                    delta = start_delta
                    for i in np.arange(len(row)):
                        val = row[i]
                        if val > max_val:
                            max_val = val
                            max_pos = i
                        if val < min_val:
                            min_val = val
                            min_pos = i

                        if lookForMax:
                            if val < (max_val - delta):
                                max_locs += [max_pos]
                                min_val = val
                                min_pos = i
                                lookForMax = 0
                        else:
                            if val > (min_val + delta):
                                min_locs += [min_pos]
                                max_val = val
                                max_pos = i
                                lookForMax = 1
                    peak_locs = max_locs # concat two lists
                    not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
                    row[not_peaks] = 0
                    rows.append(row)
                new_test_list.append(np.array(rows))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    #data = cp.deepcopy(instance.get_features().get_data())
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_FEATURES or STREAM_TO_FEATURES:
                    #data = cp.deepcopy(instance.get_stream().get_data())
                    data = instance.get_stream().get_data()
                rows = []
                for row in data:
                    min_val, max_val = np.inf, -np.inf
                    min_pos, max_pos = -1, 1
                    min_locs = []
                    max_locs = []
                    delta = start_delta
                    for i in np.arange(len(row)):
                        val = row[i]
                        if val > max_val:
                            max_val = val
                            max_pos = i
                        if val < min_val:
                            min_val = val
                            min_pos = i

                        if lookForMax:
                            if val < (max_val - delta):
                                max_locs += [max_pos]
                                min_val = val
                                min_pos = i
                                lookForMax = 0
                        else:
                            if val > (min_val + delta):
                                min_locs += [min_pos]
                                max_val = val
                                max_pos = i
                                lookForMax = 1
                        #delta = delta*0.9999
                    #print minLocs, maxLocs
                    #peak_locs = min_locs + max_locs # concat two lists
                    #print delta
                    peak_locs = max_locs # concat two lists
                    X = [0 for x in np.arange(len(row))]
                    X[0] = peak_locs[0]
                    if len(peak_locs) > 1:
                        X[1] = peak_locs[-1]
                    rows.append(X)
                    # import matplotlib.pyplot as plt
                    # plt.plot(row)
                    # plt.title('peaks ' + str(peak_locs))
                    # plt.show()
                data = np.array(rows)
                # print(data)
                # print(data[0][data[0] > 0])
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    labels = np.array(['Peaks' + str(i) for i in range(len(data.flatten()))])
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             labels
                                            ))
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1, -1)),
                                                     labels=new_labels
                                                    )

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

# def my_peak_finder(data_pair, start_delta=5, lookForMax=1, mode=STREAM_TO_FEATURES):
#     """Zero out all but the peaks in the data
#
#     delta is the amount to withdraw before locking
#
#     Args:
#         data_pair: given datapair
#         start_delta: initial value of delta
#         lookForMax: max parameter
#         mode: iteration mode, options: FEATURES_TO_FEATURES,
#         STREAM_TO_STREAM, or STREAM_TO_FEATURES
#
#     Returns:
#         Data Pair with peak finder feature added
#     """
#     if isinstance(data_pair, GTMOEPImagePair):
#         train_list = data_pair.get_train_batch()
#         test_list = data_pair.get_test_batch()
#         new_train_list = []
#         new_test_list = []
#         try:
#             for train, test in zip(train_list, test_list):
#                 rows = []
#                 for row in train:
#                     min_val, max_val = np.inf, -np.inf
#                     min_pos, max_pos = -1, 1
#                     min_locs = []
#                     max_locs = []
#                     delta = start_delta
#                     for i in np.arange(len(row)):
#                         val = row[i]
#                         if val > max_val:
#                             max_val = val
#                             max_pos = i
#                         if val < min_val:
#                             min_val = val
#                             min_pos = i
#
#                         if lookForMax:
#                             if val < (max_val - delta):
#                                 max_locs += [max_pos]
#                                 min_val = val
#                                 min_pos = i
#                                 lookForMax = 0
#                         else:
#                             if val > (min_val + delta):
#                                 min_locs += [min_pos]
#                                 max_val = val
#                                 max_pos = i
#                                 lookForMax = 1
#                     peak_locs = max_locs # concat two lists
#                     not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
#                     row[not_peaks] = 0
#                     rows.append(row)
#                 new_train_list.append(np.array(rows))
#                 rows = []
#                 for row in test:
#                     min_val, max_val = np.inf, -np.inf
#                     min_pos, max_pos = -1, 1
#                     min_locs = []
#                     max_locs = []
#                     delta = start_delta
#                     for i in np.arange(len(row)):
#                         val = row[i]
#                         if val > max_val:
#                             max_val = val
#                             max_pos = i
#                         if val < min_val:
#                             min_val = val
#                             min_pos = i
#
#                         if lookForMax:
#                             if val < (max_val - delta):
#                                 max_locs += [max_pos]
#                                 min_val = val
#                                 min_pos = i
#                                 lookForMax = 0
#                         else:
#                             if val > (min_val + delta):
#                                 min_locs += [min_pos]
#                                 max_val = val
#                                 max_pos = i
#                                 lookForMax = 1
#                     peak_locs = max_locs # concat two lists
#                     not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
#                     row[not_peaks] = 0
#                     rows.append(row)
#                 new_test_list.append(np.array(rows))
#
#
#             data_pair.set_train_batch(new_train_list)
#             data_pair.set_test_batch(new_test_list)
#         except:
#             data_pair.clear()
#             raise
#     else:
#         data_list = []
#         for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
#             instances = cp.deepcopy(data_set.get_instances())
#
#             for instance in instances:
#                 if mode is FEATURES_TO_FEATURES:
#                     #data = cp.deepcopy(instance.get_features().get_data())
#                     data = instance.get_features().get_data()
#                 elif mode is STREAM_TO_FEATURES or STREAM_TO_FEATURES:
#                     #data = cp.deepcopy(instance.get_stream().get_data())
#                     data = instance.get_stream().get_data()
#                 rows = []
#                 for row in data:
#                     min_val, max_val = np.inf, -np.inf
#                     min_pos, max_pos = -1, 1
#                     min_locs = []
#                     max_locs = []
#                     delta = start_delta
#                     for i in np.arange(len(row)):
#                         val = row[i]
#                         if val > max_val:
#                             max_val = val
#                             max_pos = i
#                         if val < min_val:
#                             min_val = val
#                             min_pos = i
#
#                         if lookForMax:
#                             if val < (max_val - delta):
#                                 max_locs += [max_pos]
#                                 min_val = val
#                                 min_pos = i
#                                 lookForMax = 0
#                         else:
#                             if val > (min_val + delta):
#                                 min_locs += [min_pos]
#                                 max_val = val
#                                 max_pos = i
#                                 lookForMax = 1
#                         #delta = delta*0.9999
#                     #print minLocs, maxLocs
#                     #peak_locs = min_locs + max_locs # concat two lists
#                     #print delta
#                     peak_locs = max_locs # concat two lists
#                     not_peaks = [i for i in np.arange(len(row)) if i not in peak_locs]
#                     #print row
#                     row[not_peaks] = 0
#                     rows.append(row)
#                 data = np.array(rows)
#
#                 if mode is FEATURES_TO_FEATURES:
#                     instance.get_features().set_data(data)
#                 elif mode is STREAM_TO_STREAM:
#                     instance.get_stream().set_data(data)
#                 elif mode is STREAM_TO_FEATURES:
#                     old_features = instance.get_features().get_data()
#                     new_features = np.concatenate((old_features.flatten(), data.flatten()))
#                     labels = np.array(['Peaks' + str(i) for i in range(len(data.flatten()))])
#                     new_labels = np.concatenate((
#                                              instance.get_features().get_labels(),
#                                              labels
#                                             ))
#                     instance.get_features().set_data(
#                                                      np.reshape(new_features, (1, -1)),
#                                                      labels=new_labels
#                                                     )
#
#             new_data_set = GTMOEPData(instances)
#             data_list.append(new_data_set)
#         data_pair = GTMOEPDataPair(train_data=data_list[0],
#                                        test_data=data_list[1])
#
#     gc.collect(); return data_pair

def my_informed_search(data_pair, peak_data_pair, search_window=5, log_slope_thresh=0.08, mode=STREAM_TO_FEATURES):
    """Zero out all but the peaks in the data

    delta is the amount to withdraw before locking

    Args:
        data_pair: given datapair
        peak_data_pair: given peak datapair
        search_window: size of search window
        log_slope_thresh: equation parameter
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair with informed search feature added

    Raises:
        ValueError
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        peak_train_list = peak_data_pair.get_train_batch()
        peak_test_list = peak_data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        deriv = 2
        order = 3
        window_length = 11
        maxzcs = 0
        try:
            for train in zip(train_list, peak_train_list):
                rows = []
                for row, peak_row in zip(train[0], train[1]):
                    #print row
                    zero_crossings = []
                    log_slopes = []
                    all_smoothed_data = savitzky_golay(row, window_length,
                        order, deriv)
                    for i in np.arange(search_window,len(peak_row)):
                        if peak_row[i] > 0:
                            k = i - search_window
                            # Take the Second derivative
                            smoothed_data = all_smoothed_data[k:i]
                            zero_crossing = None
                            # Find the Zero crossing if any
                            interest_point_zero_crossings = []
                            interest_point_log_slopes = []
                            for j in np.arange(search_window-1):
                                if smoothed_data[j]*smoothed_data[j+1] <= 0:
                                    log_slope = abs(np.log10(row[k+j+3]) - np.log10(row[k+j-3]))
                                    zero_crossing = k+j

                                    # log the point
                                    interest_point_zero_crossings.append(zero_crossing)
                                    interest_point_log_slopes.append(log_slope)
                            interest_point_log_slopes = np.array(interest_point_log_slopes)
                            interest_point_zero_crossings = np.array(interest_point_zero_crossings)
                            if len(interest_point_log_slopes) > 0:
                                log_slope = np.max(interest_point_log_slopes)
                                #print('SLOPE', log_slope, 'THRESH', log_slope_thresh)
                                if log_slope >= log_slope_thresh:
                                    log_slope_loc = np.argmax(interest_point_log_slopes)
                                    zero_crossings.append(interest_point_zero_crossings[log_slope_loc])
                    if maxzcs == 0:
                        # This will protect against empty data
                        maxzcs = 1
                    # Keep track of most zero crossings for pading
                    if len(zero_crossings) > maxzcs:
                        maxzcs = len(zero_crossings)
                    rows.append(zero_crossings)
                # Pad the rows
                for i in np.arange(len(rows)):
                    rows[i].extend(np.arange(maxzcs-len(rows[i]))*np.nan)
                new_train_list.append(np.array(rows))
            for test in zip(test_list, peak_test_list):
                rows = []
                for row, peak_row in zip(test[0], test[1]):
                    #print row
                    zero_crossings = []
                    log_slopes = []
                    all_smoothed_data = savitzky_golay(row, window_length,
                        order, deriv)
                    for i in np.arange(search_window,len(peak_row)):
                        if peak_row[i] > 0:
                            k = i - search_window
                            # Take the Second derivative
                            smoothed_data = all_smoothed_data[k:i]
                            zero_crossing = None
                            # Find the Zero crossing if any
                            interest_point_zero_crossings = []
                            interest_point_log_slopes = []
                            for j in np.arange(search_window-1):
                                if smoothed_data[j]*smoothed_data[j+1] <= 0:
                                    log_slope = abs(np.log10(row[k+j+3]) - np.log10(row[k+j-3]))
                                    zero_crossing = k+j

                                    # log the point
                                    interest_point_zero_crossings.append(zero_crossing)
                                    interest_point_log_slopes.append(log_slope)
                            interest_point_log_slopes = np.array(interest_point_log_slopes)
                            interest_point_zero_crossings = np.array(interest_point_zero_crossings)
                            if len(interest_point_log_slopes) > 0:
                                log_slope = np.max(interest_point_log_slopes)
                                #print('SLOPE', log_slope, 'THRESH', log_slope_thresh)
                                if log_slope >= log_slope_thresh:
                                    log_slope_loc = np.argmax(interest_point_log_slopes)
                                    zero_crossings.append(interest_point_zero_crossings[log_slope_loc])
                    if maxzcs == 0:
                        # This will protect against empty data
                        maxzcs = 1
                    # Keep track of most zero crossings for pading
                    if len(zero_crossings) > maxzcs:
                        maxzcs = len(zero_crossings)
                    rows.append(zero_crossings)
                # Pad the rows
                for i in np.arange(len(rows)):
                    rows[i].extend(np.arange(maxzcs-len(rows[i]))*np.nan)
                new_test_list.append(np.array(rows))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        dataset_number = 0
        for data_set, peak_data_set in zip(
            [data_pair.get_train_data(), data_pair.get_test_data()],
            [peak_data_pair.get_train_data(), peak_data_pair.get_test_data()]
            ):

            instances = cp.deepcopy(data_set.get_instances())
            peak_instances = cp.deepcopy(peak_data_set.get_instances())
            instance_number = 0
            for instance, peak_instance in zip(instances, peak_instances):
                if mode is FEATURES_TO_FEATURES:
                    #data = cp.deepcopy(instance.get_features().get_data())
                    #peak_data = cp.deepcopy(peak_instance.get_features().get_data())
                    data = instance.get_features().get_data()
                    peak_data = peak_instance.get_features().get_data()
                elif mode is STREAM_TO_FEATURES or STREAM_TO_FEATURES:
                    #data = cp.deepcopy(instance.get_stream().get_data())
                    #peak_data = cp.deepcopy(peak_instance.get_stream().get_data())
                    data = instance.get_stream().get_data()
                    peak_data = peak_instance.get_stream().get_data()
                if peak_data.shape != data.shape:
                    raise ValueError('Peaks and data do not match in shape')
                deriv = 2
                order = 3
                window_length = 11
                rows = []
                maxzcs = 0
                # print(dataset_number)
                # if dataset_number == 1:
                #     print(data)
                #     print(peak_data)
                #     print(data)
                #     print(peak_data)
                #     if instance_number == 0:
                #         print(str(peak_data[0]))
                #     print(np.count_nonzero(peak_data[0]))
                #     print(np.nonzero(peak_data[0]), len(np.nonzero(peak_data[0])), peak_data[np.nonzero(peak_data)])
                #     import matplotlib.pyplot as plt
                #     plt.plot(data[0,:])
                #     # plt.plot(peak_data[0,:], '*')
                #     plt.show()
                for row, peak_row in zip(data, peak_data):
                    # print(row)
                    zero_crossings = []
                    log_slopes = []
                    all_smoothed_data = savitzky_golay(row, window_length,
                        order, deriv)
                    # print(all_smoothed_data)
                    # search_window
                    # print(row, peak_row)
                    for i in np.arange(search_window,len(peak_row)):
                        if peak_row[i] > 0:
                            k = i - search_window
                            # print(k)
                            # Take the Second derivative
                            smoothed_data = all_smoothed_data[k:i]
                            zero_crossing = None
                            # Find the Zero crossing if any
                            interest_point_zero_crossings = []
                            interest_point_log_slopes = []
                            for j in np.arange(search_window-1):
                                if smoothed_data[j]*smoothed_data[j+1] <= 0:
                                    # try:
                                    # print(k)
                                    # print(j)
                                    # print(abs(np.log10(row[k+j+3]) - np.log10(row[k+j-3])))
                                    log_slope = abs(np.log10(row[k+j+3]+1e-12) - np.log10(row[k+j-3]+1e-12))
                                    # if instance_number == 0 and dataset_number == 1:
                                        # print("log_slope " + str(type(log_slope)) + str(log_slope))
                                    # except ZeroDivisionError:
                                    #     print("Divide by Zero error found")
                                    zero_crossing = k+j

                                    # log the point
                                    interest_point_zero_crossings.append(zero_crossing)
                                    # if not np.isnan(log_slope):
                                    interest_point_log_slopes.append(log_slope)
                            # if len(interest_point_log_slopes) == 0:
                            #     print("Zero " + str(interest_point_zero_crossings) + str(interest_point_log_slopes))
                            # else:
                            #     print("Non-Zero " + str(interest_point_zero_crossings) + str(interest_point_log_slopes))
                            interest_point_log_slopes = np.array(interest_point_log_slopes)
                            interest_point_zero_crossings = np.array(interest_point_zero_crossings)
                            # print(interest_point_log_slopes)
                            # print(interest_point_zero_crossings)
                            if len(interest_point_log_slopes) > 0:
                                log_slope = np.max(interest_point_log_slopes)
                                # if instance_number == 144:
                                    # print("HERE " + str(log_slope))
                                # print('SLOPE', log_slope, 'THRESH', log_slope_thresh)
                                if log_slope >= log_slope_thresh:
                                    log_slope_loc = np.argmax(interest_point_log_slopes)
                                    zero_crossings.append(interest_point_zero_crossings[log_slope_loc])
                            # print("")
                            # if instance_number == 144:
                            #     print("Values " + " " + str(log_slope_loc) + " " + str(zero_crossing))
                            # print("")
                    if maxzcs == 0:
                        # This will protect against empty data
                        maxzcs = 1
                    # Keep track of most zero crossings for pading
                    if len(zero_crossings) > maxzcs:
                        maxzcs = len(zero_crossings)
                    rows.append(zero_crossings)
                # Pad the rows
                for i in np.arange(len(rows)):
                    rows[i].extend(np.ones(maxzcs-len(rows[i]))*-1)
                data = np.array(rows)
                # if dataset_number == 1 and instance_number == 0:
                #     print("Data " + str(data))
                #     time.sleep(20)
                if mode is FEATURES_TO_FEATURES:
                    labels = np.array(['InterestPts' + str(i) for i in range(len(data.flatten()))])
                    instance.get_features().set_data(data,labels=labels)
                elif mode is STREAM_TO_STREAM:
                    labels = np.array(['InterestPts' + str(i) for i in range(data.shape[1])])
                    instance.get_stream().set_data(data, labels=labels)
                elif mode is STREAM_TO_FEATURES:
                    old_features = instance.get_features().get_data()
                    # print(old_features)
                    # print(data)
                    # print(old_features.flatten())
                    # print(data.flatten())

                    new_features = np.concatenate((old_features.flatten(), data.flatten()))
                    # if dataset_number == 1 and instance_number == 0:
                    #     print("New Features " + str(new_features))
                    #     print("Old Features " + str(old_features))
                    labels = np.array(['InterestPts' + str(i) for i in range(len(data.flatten()))])
                    new_labels = np.concatenate((
                                             instance.get_features().get_labels(),
                                             labels
                                            ))
                    # print(new_features)
                    # print(new_labels)
                    instance.get_features().set_data(
                                                     np.reshape(new_features, (1,-1)),
                                                     labels=new_labels
                                                    )
                    # if instance_number == 144:
                    #     print(instance.get_features().get_data())
                instance_number += 1
            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
            dataset_number += 1
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def savitzky_golay(y, window_size, order, deriv=0):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techhniques.

    Args:
        y: the values of the time history of the signal
        window_size: the length of the window. Must be an odd integer number
        order: the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1
        deriv: the order of the derivative to compute
        (default = 0 means only smoothing)

    Returns:
        the smoothed signal (or it's n-th derivative)

    Raises:
        ValueError, TypeError
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')

def ecdfRep(data, components):
    """Estimate ecdf-representation according to Hammerla, Nils Y., et al.
    "On preserving statistical characteristics of accelerometry data using
    their empirical cumulative distribution." ISWC. ACM, 2013.

    Args:
        data: input data (rows = samples)
        components: number of components to extract per axis

    Returns:
        data representation with M = d*components+d elements
    """
    m = data.mean(1)
    data = np.sort(data, axis=1)
    data = data[:, np.int32(np.around(np.linspace(0,data.shape[1]-1,num=components)))]
    data = data.flatten(0)
    return np.hstack((data, m)).reshape((1, -1))

def myArgMin(dataPair):
    """Calculates Argmin of Data Pair

    Args:
        dataPair: given Data Pair

    Returns:
        modified Data Pair
    """
    dataPair = cp.deepcopy(dataPair)
    mins = [];
    for row in dataPair.testData.numpy['features']:
        mins += [np.argmin(row)]

    dataPair.testData.numpy['classes'] = mins
    dataPair.testData.numpy['classes'] = np.array([[str(int(val))]for val in dataPair.testData.numpy['classes']])
    dataPair.testData.rebuildFromNumpy()
    # TODO: We want to add on new class value as a feature as well
    return dataPair

def my_richardson_lucy(data_pair, iterations = 50, mode=FEATURES_TO_FEATURES):
    """Perform richardson-lucy deconvolution on a dataset

    Args:
        data_pair: given datapair
        iterations: number of iterations for deconvolution
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair transformed with richardson-lucy deconvolution
    """
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                psf = np.ones((train.shape)) / train.shape[1]
                new_train_list.append(restoration.richardson_lucy(train, psf, iterations=iterations))
                psf = np.ones((test.shape)) / test.shape[1]
                new_test_list.append(restoration.richardson_lucy(test, psf, iterations=iterations))


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for Richardson-Lucy Filter'
                        )
                
                psf = np.ones((data.shape)) / data.shape[1]
                data = restoration.richardson_lucy(data, psf, iterations=iterations)
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_supersampling(data_pair, size_factor=0.1, resample = 1, mode=FEATURES_TO_FEATURES):
    """Perform supersampling on the datapair

    Args:
        data_pair: given datapair
        size_factor: Target size for downsampling
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair transformed through supersampling
    """
    if isinstance(data_pair, GTMOEPImagePair):
        resample = resample % 4
        if resample == 0:
            resample = Image.BILINEAR
        elif resample == 1:
            resample = Image.NEAREST
        elif resample == 2:
            resample = Image.BICUBIC
        else:
            resample = Image.LANCZOS
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                im = Image.fromarray(train)
                new_train_list.append(np.array(im.resize(size = (train.shape[0],np.round(train.shape[1]*size_factor)), resample=resample)))

                im = Image.fromarray(test)
                new_test_list.append(np.array(im.resize(size = (test.shape[0], np.round(test.shape[1]*size_factor)), resample=resample)))



            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        resample = resample % 4
        if resample == 0:
            resample = Image.BILINEAR
        elif resample == 1:
            resample = Image.NEAREST
        elif resample == 2:
            resample = Image.BICUBIC
        else:
            resample = Image.LANCZOS
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for supersampling Filter'
                        )
                
                im = Image.fromarray(data)
                im.resize(size = (data.shape[0],int(np.round(data.shape[1]*size_factor))), resample=resample)

                data = np.array(im)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_bspline(data_pair, spline_degree=3, mode=FEATURES_TO_FEATURES):
    """Perform b-spline curve fitting

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair from the curve that was fit using b spline
    """
    # print('working')
    if isinstance(data_pair, GTMOEPImagePair):

        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                X = [x+1 for x in range(train.shape[1])]
                # print('still working')
                # if w_mode:
                #     w_param = np.random.rand(len(X))
                # else:
                #     w_param = np.ones(len(X))
                # print(w_param)
                for dimension in range(train.shape[0]):
                    # print("here")
                    (t,c,k) = interpolate.splrep(x=X, y=train[dimension], k=spline_degree)
                    spline = interpolate.BSpline(t,c,k)
                    train[dimension] = spline(X)
                # print("finsih")
                new_train_list.append(train)

                for dimension in range(test.shape[0]):

                    (t,c,k) = interpolate.splrep(x=X, y=test[dimension], k=spline_degree)
                    spline = interpolate.BSpline(t,c,k)
                    test[dimension] = spline(X)


                new_test_list.append(test)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())
            # print('still working')

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for b spline curve fitting'
                        )


                X = [x+1 for x in range(data.shape[1])]
                # if w_mode:
                #     w_param = np.random.rand(len(X))
                # else:
                #     w_param = np.ones(len(X))
                for dimension in range(data.shape[0]):
                    # print("here")
                    t,c,k = interpolate.splrep(x=X, y=data[dimension], s=0, k=spline_degree)
                    spline = interpolate.BSpline(t,c,k)
                    data[dimension] = spline(X)
                # print("finsih")

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_matched_filtering(data_pair, method=2, mode=FEATURES_TO_FEATURES):
    """Perform matched filtering on a dataset
        https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.correlate.html
    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Data Pair transformed with match filtering
    """

    if isinstance(data_pair, GTMOEPImagePair):
        method = method % 3
        if method == 0:
            method_name = 'direct'
        elif method == 1:
            method_name = 'fft'
        else:
            method_name = 'auto'
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                for dimension in range(train.shape[0]):
                    train[dimension] = scipy.signal.correlate(train[dimension], np.ones(train.shape[1]), mode='same', method=method_name) / train.shape[1]
                new_train_list.append(train)

                for dimension in range(test.shape[0]):
                    test[dimension] = scipy.signal.correlate(test[dimension], np.ones(test.shape[1]), mode='same', method=method_name) / test.shape[1]
                
                new_test_list.append(test)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        method = method % 3
        if method == 0:
            method_name = 'direct'
        elif method == 1:
            method_name = 'fft'
        else:
            method_name = 'auto'
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for matched filter'
                        )
                

                for dimension in range(data.shape[0]):
                    data[dimension] = scipy.signal.correlate(data[dimension], np.ones(data.shape[1]), mode='valid', method=method_name) / data.shape[1]

                
                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair


def my_gaussian_peak_lm(data_pair, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2, mode=FEATURES_TO_FEATURES):
    """Fit the data to the sum of two gaussians using the levenburg-markwardt method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of two surface responses
    """

    def gaussian(x, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2):
        y = np.zeros_like(x)
        y = (amp_1 * np.exp(-((x - cen_1) / wid_1)**2)) + (amp_2 * np.exp(-((x - cen_2) / wid_2)**2))
        return y

    # def gaussian(x, *params):
    #     y = np.zeros_like(x)
    #     y = (params[0] * np.exp(-((x - params[1]) / params[2])**2)) + (params[3] * np.exp(-((x - params[4]) / params[5])**2))
    #     return y

    if isinstance(data_pair, GTMOEPImagePair):
        pass
        # train_list = data_pair.get_train_batch()
        # test_list = data_pair.get_test_batch()
        # new_train_list = []
        # new_test_list = []
        # try:
        #
        #     for train, test in zip(train_list, test_list):
        #         X = [x + 1 for x in range(train.shape[1])]
        #         for dimension in range(train.shape[0]):
        #             guess = [1, 1, 1, 1, 1, 1]
        #             try:
        #                 popt, pcov = curve_fit(gaussian, X, train[dimension], p0=guess, method='lm', maxfev=5000)
        #                 fit = gaussian(X, *popt)
        #                 data = fit
        #                 # print('ANSWER', popt[0],popt[3])
        #
        #                 # max_locs = [np.round(popt[0]), np.round(popt[3])]
        #                 # peak_locs = max_locs  # concat two lists
        #                 # not_peaks = [i for i in np.arange(len(train[dimension])) if i not in peak_locs]
        #                 # train[dimension][not_peaks] = 0
        #
        #             except RuntimeError:
        #                 print("Error - curve_fit failed")
        #                 not_peaks = [i for i in np.arange(len(train[dimension]))]
        #                 train[dimension][not_peaks] = 0
        #
        #         new_test_list.append(test)
        #         new_train_list.append(train)
        #
        #     data_pair.set_train_batch(new_train_list)
        #     data_pair.set_test_batch(new_test_list)
        # except:
        #     data_pair.clear()
        #     raise
    else:
        # import matplotlib.pyplot as plt
        data_list = []
        gmodel = Model(gaussian)
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())
            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Feature to Features available for my_gaussian_peak'
                    )
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()
                X = [x for x in range(len(data[0]))]
                new_data = []
                for row in data:
                    try:
                        # import matplotlib.pyplot as plt
                        result = gmodel.fit(row, x=X, amp_1=amp_1, cen_1=cen_1, wid_1=wid_1,
                                            amp_2=amp_2, cen_2=cen_2, wid_2=wid_2)
                        # plt.plot(row)
                        if mode is STREAM_TO_STREAM:
                            fit = result.best_fit
                            row = fit
                            row[row < 0] = 0
                        elif mode is STREAM_TO_FEATURES:
                            row = [result.params['cen_1'].value, result.params['cen_2'].value]
                        new_data.append(row)

                        # plt.title("Initial Centers " + str(result.init_params['cen_1'].value) + " " +
                        #           str(result.init_params['cen_2'].value) +
                        #           "   Optimal Centers " +
                        #           str(result.params['cen_1'].value) + " " +
                        #           str(result.params['cen_2'].value), loc='left')
                        # plt.plot(fit)
                        # plt.show()

                    except RuntimeError:
                        print("Error - curve_fit failed")
                new_data = np.array(new_data)
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Feature to Features available for my_gaussian_peak'
                    )
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(new_data)
                elif mode is STREAM_TO_FEATURES:
                    instance.get_features().set_data(new_data, labels=np.array(['cen_1', 'cen_2']))
            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])
    gc.collect()
    return data_pair


def gaussian_peak_lm(data_pair, peak_data_pair, mode=FEATURES_TO_FEATURES):
    """Fit the data to the sum of two gaussians using the levenburg-markwardt method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of two surface responses
    """

    def gaussian(x, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2):
        y = np.zeros_like(x)
        y = (amp_1 * np.exp(-((x - cen_1) / wid_1)**2)) + (amp_2 * np.exp(-((x - cen_2) / wid_2)**2))
        return y

    # def gaussian(x, *params):
    #     y = np.zeros_like(x)
    #     y = (params[0] * np.exp(-((x - params[1]) / params[2])**2)) + (params[3] * np.exp(-((x - params[4]) / params[5])**2))
    #     return y

    if isinstance(data_pair, GTMOEPImagePair):
        pass
        # train_list = data_pair.get_train_batch()
        # test_list = data_pair.get_test_batch()
        # new_train_list = []
        # new_test_list = []
        # try:
        #
        #     for train, test in zip(train_list, test_list):
        #         X = [x + 1 for x in range(train.shape[1])]
        #         for dimension in range(train.shape[0]):
        #             guess = [1, 1, 1, 1, 1, 1]
        #             try:
        #                 popt, pcov = curve_fit(gaussian, X, train[dimension], p0=guess, method='lm', maxfev=5000)
        #                 fit = gaussian(X, *popt)
        #                 data = fit
        #                 # print('ANSWER', popt[0],popt[3])
        #
        #                 # max_locs = [np.round(popt[0]), np.round(popt[3])]
        #                 # peak_locs = max_locs  # concat two lists
        #                 # not_peaks = [i for i in np.arange(len(train[dimension])) if i not in peak_locs]
        #                 # train[dimension][not_peaks] = 0
        #
        #             except RuntimeError:
        #                 print("Error - curve_fit failed")
        #                 not_peaks = [i for i in np.arange(len(train[dimension]))]
        #                 train[dimension][not_peaks] = 0
        #
        #         new_test_list.append(test)
        #         new_train_list.append(train)
        #
        #     data_pair.set_train_batch(new_train_list)
        #     data_pair.set_test_batch(new_test_list)
        # except:
        #     data_pair.clear()
        #     raise
    else:
        # import matplotlib.pyplot as plt
        data_list = []
        gmodel = Model(gaussian)
        for data_set, peak_set in zip([data_pair.get_train_data(), data_pair.get_test_data()],
                                      [peak_data_pair.get_train_data(), peak_data_pair.get_test_data()]):
            instances = cp.deepcopy(data_set.get_instances())
            peak_instances = cp.deepcopy(peak_set.get_instances())
            for instance, peak_instance in zip(instances, peak_instances):
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Feature to Features available for my_gaussian_peak'
                    )
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                    peak_data = peak_instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()
                    peak_data = peak_instance.get_stream().get_data()
                if peak_data.shape != data.shape:
                    raise ValueError('Peaks and data do not match in shape')
                X = [x for x in range(len(data[0]))]
                new_data = []
                for row, peak_row in zip(data, peak_data):
                    try:
                        # import matplotlib.pyplot as plt
                        # plt.plot(row)
                        # plt.show()
                        # print(peak_row, type(peak_row), np.any(peak_row > 0))
                        # print(peak_row[peak_row > 0])
                        peaks = peak_row[peak_row > 0]
                        if len(peaks) < 2:
                            peaks = np.array(peaks.tolist().append(peaks[0] + np.random.randint(0, 100)))
                        # import matplotlib.pyplot as plt
                        # print(peaks)
                        result = gmodel.fit(row, x=X, amp_1=row[peaks[0]], cen_1=peaks[0], wid_1=50,
                                            amp_2=row[peaks[-1]], cen_2=peaks[-1], wid_2=50)
                        # plt.plot(row)
                        row = [result.params['cen_1'].value, result.params['cen_2'].value]
                        new_data.append(row)

                        # plt.title("Initial Centers " + str(result.init_params['cen_1'].value) + " " +
                        #           str(result.init_params['cen_2'].value) +
                        #           "   Optimal Centers " +
                        #           str(result.params['cen_1'].value) + " " +
                        #           str(result.params['cen_2'].value), loc='left')
                        # plt.plot(fit)
                        # plt.show()

                    except RuntimeError:
                        print("Error - curve_fit failed")
                new_data = np.array(new_data)
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Feature to Features available for my_gaussian_peak'
                    )
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(new_data)
                elif mode is STREAM_TO_FEATURES:
                    instance.get_features().set_data(new_data, labels=np.array(['cen_1', 'cen_2']))
            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])
    gc.collect()
    return data_pair


def gaussian_peak_em(data_pair, mode=FEATURES_TO_FEATURES):
    """Fit the data to the sum of three gaussians using the expectation-maximization method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of the two surface responses
    """

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                X = [x + 1 for x in range(train.shape[1])]

                for dimension in range(train.shape[0]):
                    popt = [1, 1, 1, 1, 1, 1, 1, 1, 1]
                    try:
                        popt, pcov = curve_fit(tri_gaussian, X, train[dimension], p0=popt, method='lm', maxfev=1000)
                    except RuntimeError:
                        print("Error - curve_fit failed")

                    train[dimension] = np.array([tri_gaussian(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5],
                                                              popt[6], popt[7], popt[8]) for x in X])

                new_train_list.append(train)

                X = data.reshape(len(data), len(data[0]))

                function = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1] / 2.) \
                                         * np.exp(-.5 * np.einsum('ij, ij -> i', \
                                                                  X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

                mu = X[np.random.choice(d, 2, False), :]
                Sigma = [np.eye(d)] * 2
                w = [0.5, 0.5]
                R = np.zeros((n, 2))
                log_likelihoods = []
                num_components = len(w)
                Q = np.zeros(length, num_components)
                pf = np.zeros(length, num_components)
                spf = np.zeros(length)
                nq = np.zeros(num_components)
                nqi = np.zeros(num_components)
                nqims = np.zeros(num_components)
                while len(log_likelihoods) < 1000:
                    for j in range(num_components):
                        pf[:, j] = w[j] * function(mu[j], Sigma[j])

                    log_likelihood = np.sum(np.log(np.sum(R, axis=1)))

                    for i in range(length):
                        spf[i] = sum(pf[i])

                    Q[i][j] = pf[i][j] / spf[i]

                    for j in range(num_components):
                        nq[j] = X * Q[j]

                        w[j] = nq[j] / sum(X)

                    for j in range(num_components):
                        nqi[j] = X * Q[j] * np.arange(length)

                        mu[j] = nqi[j] / (w[j]) * sum(X)

                    for j in range(num_components):
                        nqims[j] = X * Q[j] * (np.arange(length) - mu[j]) ** 2

                        Sigma[j] = np.sqrt(nqims[j] / (w[j]) * sum(X))

                    log_likelihoods.append(log_likelihood)
                    if len(log_likelihoods) < 2: continue
                    if np.abs(log_likelihood - log_likelihoods[-2]) < 0.0001: break

                new_test_list.append(test)

            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for gaussian lm Filter'
                    )

                num_components = 2
                print(data.shape)
                X = data.reshape(data.shape[1], data.shape[0])
                n, d = X.shape

                mu = X[np.random.choice(n, num_components, False), :]
                Sigma = [np.eye(d)] * num_components
                w = [1. / num_components] * num_components
                R = np.zeros((n, num_components))
                log_likelihoods = []

                P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-X.shape[1] / 2.) \
                                  * np.exp(-.5 * np.einsum('ij, ij -> i', \
                                                           X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

                while len(log_likelihoods) < 1000:
                    for k in range(num_components):
                        R[:, k] = w[k] * P(mu[k], Sigma[k])

                    log_likelihood = np.sum(np.log(np.sum(R, axis=1)))
                    print(np.sum(R, axis=1))
                    log_likelihoods.append(log_likelihood)
                    R = (R.T / np.sum(R, axis=1)).T
                    N_ks = np.sum(R, axis=0)
                    for k in range(num_components):
                        mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis=1).T
                        print('MU', mu)
                        x_mu = np.matrix(X - mu[k])

                        Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T, R[:, k]), x_mu))

                        w[k] = 1. / n * N_ks[k]

                    if len(log_likelihoods) < 2: continue
                    if np.abs(log_likelihood - log_likelihoods[-2]) < 0.0001: break

                data = []

                p = [0]
                for k in range(num_components):
                    p.append(p[k] + w[k])
                for i in range(d):
                    for k in range(num_components):
                        if random.random() < p[k + 1]:
                            data.append(np.random.multivariate_normal.rvs(mu[k + 1], Sigma[k + 1]))
                print(data)

                data = np.array(data)
                print(data.shape)

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])

    gc.collect();
    return data_pair

def lognormal_lm(data_pair, mode=FEATURES_TO_FEATURES):
    """Fits a lognormal curve to each dimension and returns a stream from the lognormal distribution using the levenburg-markwardt method
    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
       Peak locations of surface response
    """

    def lognormal(x, mean, std):
        return (1/(x*std*np.sqrt(2*np.pi))) * np.exp(-(np.log(x)-mean)**2 /(2*(std**2)))

    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []
        try:
            for train, test in zip(train_list, test_list):
                X = [x+1 for x in range(train.shape[1])]
               
                for dimension in range(train.shape[0]):

                    try:
                        popt, pcov = curve_fit(lognormal, X, train[dimension], p0=popt,method='lm', maxfev = 10000)
                    except RuntimeError:
                        print("Error - curve_fit failed")
                
                    train[dimension] = np.array([lognormal(x, popt[0], popt[1]) for x in X])
                
                new_train_list.append(train)

                X = [x+1 for x in range(test.shape[1])]
                
                for dimension in range(test.shape[0]):

                    try:
                        popt, pcov = curve_fit(gaussian, X, test[dimension],p0=popt, method='lm',maxfev = 10000)
                    except RuntimeError:
                        print("Error - curve_fit failed")
                
                    test[dimension] = np.array([lognormal(x, popt[0], popt[1]) for x in X])
                
                new_test_list.append(test)


            data_pair.set_train_batch(new_train_list)
            data_pair.set_test_batch(new_test_list)
        except:
            data_pair.clear()
            raise
    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for lognormal lm Filter'
                        )
                

                X = [x+1 for x in range(data.shape[1])]
                for dimension in range(data.shape[0]):

                    popt = [np.mean(data[dimension]), np.std(data[dimension])]

                    popt, pcov = curve_fit(lognormal, X, data[dimension], p0 = popt, method='lm',maxfev = 10000)
                    
                    data[dimension] = np.array([lognormal(x, popt[0], popt[1]) for x in X])
                    

                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                       test_data=data_list[1])

    gc.collect(); return data_pair

def my_nnls_deconvolution(data_pair, mode=STREAM_TO_STREAM):
    if isinstance(data_pair, GTMOEPImagePair):
        train_list = data_pair.get_train_batch()
        test_list = data_pair.get_test_batch()
        new_train_list = []
        new_test_list = []

    else:
        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())

            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    data = instance.get_features().get_data()
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    raise ValueError(
                        'No Stream to Features available for supersampling Filter'
                    )

                # import matplotlib.pylot as plt
                # plt.plot(data)
                # plt.show()



                if mode is FEATURES_TO_FEATURES:
                    instance.get_features().set_data(data)
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(data)

            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])

    gc.collect()
    return data_pair

def my_waveform_fitting_2(data_pair, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2, x_1, x_2, x_3, x_4, y_3, y_4, mode=FEATURES_TO_FEATURES):
    """Fit the data to the sum of two gaussians using the levenburg-markwardt method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of two surface responses
    """

    # def fit_method(x, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2, x_1, x_2, x_3, x_4, y_3, y_4):
    #     y = np.zeros_like(x)
    #     temp_x = cp.deepcopy(x).astype(float)
    #     temp_x[temp_x < x_1] = 0
    #     temp_x[np.logical_and(temp_x >= x_1, temp_x <= x_2)] = y_3 * ((temp_x[np.logical_and(temp_x >= x_1, temp_x <= x_2)] - x_1) / (x_2 - x_1))
    #     temp_x[np.logical_and(temp_x >= x_2, temp_x <= x_3)] = ((y_3 * x_3 - x_2 * y_4) + temp_x[np.logical_and(temp_x >= x_2, temp_x <= x_3)] * (y_4 - y_3)) / (x_3 - x_2)
    #     temp_x[np.logical_and(temp_x >= x_3, temp_x <= x_4)] = y_4 * ((x_4 - temp_x[np.logical_and(temp_x >= x_3, temp_x <= x_4)]) / (x_4 - x_3))
    #     temp_x[temp_x >= x_4] = 0
    #     import matplotlib.pyplot as plt
    #
    #     x = np.array([i for i in range(len(x))])
    #     y = (amp_1 * np.exp(-1 * ((x - cen_1) / wid_1)**2))
    #     y += (amp_2 * np.exp(-1 * ((x - cen_2) / wid_2)**2))
    #     plt.plot(y)
    #     y += temp_x
    #     plt.plot(y)
    #     plt.show()

    def fit_method(x, amp_1, cen_1, wid_1, amp_2, cen_2, wid_2, a, b, c, d, e, g):
        y = np.zeros_like(x)
        temp_x = cp.deepcopy(x).astype(float)
        temp_x[temp_x <= a] = 0
        temp_x[np.logical_and(a < temp_x, temp_x <= b)] = e * ((temp_x[np.logical_and(a < temp_x, temp_x <= b)] - a) / (b - a))
        temp_x[np.logical_and(b < temp_x, temp_x <= c)] = ((e * c - b * g) + temp_x[np.logical_and(b < temp_x, temp_x <= c)] * (g - e)) / (c - b)
        temp_x[np.logical_and(c < temp_x, temp_x <= d)] = g * ((d - temp_x[np.logical_and(c < temp_x, temp_x <= d)]) / (d - c))
        temp_x[temp_x >= x_4] = 0
        x = np.array([i for i in range(len(x))])
        y = (amp_1 * np.exp(-1 * ((x - cen_1) / wid_1)**2))
        y += (amp_2 * np.exp(-1 * ((x - cen_2) / wid_2)**2))
        y += temp_x

    # def cost_function(params, x, stream_data):
    #     vals = params.valuesdict()
    #     amp_1 = vals['amp_1']
    #     cen_1 = vals['cen_1']
    #     wid_1 = vals['wid_1']
    #     amp_2 = vals['amp_2']
    #     cen_2 = vals['cen_2']
    #     wid_2 = vals['wid_2']
    #
    #     x_1 = vals['x_1']
    #     x_2 = vals['x_2']
    #     x_3 = vals['x_3']
    #     x_4 = vals['x_4']
    #     y_3 = vals['y_3']
    #     y_4 = vals['y_4']
    #
    #     y = np.zeros_like(x)
    #     temp_x = cp.deepcopy(x).astype(float)
    #     temp_x[temp_x < x_1] = 0
    #     temp_x[np.logical_and(temp_x >= x_1, temp_x <= x_2)] = y_3 * ((temp_x[np.logical_and(temp_x >= x_1, temp_x <= x_2)] - x_1) / (x_2 - x_1))
    #     temp_x[np.logical_and(temp_x >= x_2, temp_x <= x_3)] = ((y_3 * x_3 - x_2 * y_4) + temp_x[np.logical_and(temp_x >= x_2, temp_x <= x_3)] * (y_4 - y_3)) / (x_3 - x_2)
    #     temp_x[np.logical_and(temp_x >= x_3, temp_x <= x_4)] = y_4 * ((x_4 - temp_x[np.logical_and(temp_x >= x_3, temp_x <= x_4)]) / (x_4 - x_3))
    #     temp_x[temp_x >= x_4] = 0
    #
    #     x = np.array([i for i in range(len(x))])
    #     y = (amp_1 * np.exp(-1 * ((x - cen_1) / wid_1)**2))
    #     y += (amp_2 * np.exp(-1 * ((x - cen_2) / wid_2)**2))
    #     y += temp_x
    #     return np.sum(abs(stream_data - y))


    if isinstance(data_pair, GTMOEPImagePair):
        pass
        # train_list = data_pair.get_train_batch()
        # test_list = data_pair.get_test_batch()
        # new_train_list = []
        # new_test_list = []
        # try:
        #
        #     for train, test in zip(train_list, test_list):
        #         X = [x + 1 for x in range(train.shape[1])]
        #         for dimension in range(train.shape[0]):
        #             guess = [1, 1, 1, 1, 1, 1]
        #             try:
        #                 popt, pcov = curve_fit(gaussian, X, train[dimension], p0=guess, method='lm', maxfev=5000)
        #                 fit = gaussian(X, *popt)
        #                 data = fit
        #                 # print('ANSWER', popt[0],popt[3])
        #
        #                 # max_locs = [np.round(popt[0]), np.round(popt[3])]
        #                 # peak_locs = max_locs  # concat two lists
        #                 # not_peaks = [i for i in np.arange(len(train[dimension])) if i not in peak_locs]
        #                 # train[dimension][not_peaks] = 0
        #
        #             except RuntimeError:
        #                 print("Error - curve_fit failed")
        #                 not_peaks = [i for i in np.arange(len(train[dimension]))]
        #                 train[dimension][not_peaks] = 0
        #
        #         new_test_list.append(test)
        #         new_train_list.append(train)
        #
        #     data_pair.set_train_batch(new_train_list)
        #     data_pair.set_test_batch(new_test_list)
        # except:
        #     data_pair.clear()
        #     raise
    else:
        data_list = []
        # fit_params = Parameters()
        # fit_params.add('amp_1', value=amp_1)
        # fit_params.add('cen_1', value=cen_1)
        # fit_params.add('wid_1', value=wid_1)
        # fit_params.add('amp_2', value=amp_2)
        # fit_params.add('cen_2', value=cen_2)
        # fit_params.add('wid_2', value=wid_2)
        # fit_params.add('x_1', value=x_1)
        # fit_params.add('x_2', value=x_2)
        # fit_params.add('x_3', value=x_3)
        # fit_params.add('x_4', value=x_4)
        # fit_params.add('y_3', value=y_3)
        # fit_params.add('y_4', value=y_4)
        gmodel = Model(fit_method)
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())
            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Feature to Features available for my_gaussian_peak'
                    )
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                    indicies = instance.get_stream().get_labels()
                elif mode is STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()
                    indicies = instance.get_stream().get_labels()
                # X = [x for x in range(len(data[0]))]
                new_data = []
                for row in data:
                    try:
                        result = gmodel.fit(row, x=indicies, amp_1=amp_1, cen_1=cen_1, wid_1=wid_1,
                                            amp_2=amp_2, cen_2=cen_2, wid_2=wid_2, a=a, b=b, c=c, d=d, e=e, g=g)
                        if mode is STREAM_TO_STREAM:
                            fit = result.best_fit
                            row = fit
                            row[row < 0] = 0
                        elif mode is STREAM_TO_FEATURES:
                            row = [result.params['cen_1'].value, result.params['cen_2'].value]
                        new_data.append(row)
                    except RuntimeError:
                        print("Error - curve_fit failed")
                new_data = np.array(new_data)
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Feature to Features available for my_gaussian_peak'
                    )
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(new_data)
                elif mode is STREAM_TO_FEATURES:
                    instance.get_features().set_data(new_data, labels=np.array(['cen_1', 'cen_2']))
            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])
    gc.collect()
    return data_pair

def my_progressive_waveform_decomposition(data_pair, mode=FEATURES_TO_FEATURES):
    """Fit the data to the sum of two gaussians using the levenburg-markwardt method

    Args:
        data_pair: given datapair
        mode: iteration mode, options: FEATURES_TO_FEATURES,
        STREAM_TO_STREAM, or STREAM_TO_FEATURES

    Returns:
        Peak locations of two surface responses
    """

    def find_local_min(deriv, start=0, direction=1):
        if direction == 1:  # positive direction
            # find negative
            for i, val in enumerate(deriv[start:]):
                if val <= 0:
                    # print(i,val)
                    break
                else:
                    pass
            for j, val in enumerate(deriv[start + i:]):
                if val > 0:
                    # print(j,val)
                    # check to see if it's just noise
                    if deriv[start + i + j + 1] <= 0:
                        continue
                    else:
                        # print(j,val)
                        break
                else:
                    pass
            return start + i + j
        else:
            if start == 0:
                start = len(deriv)
            else:
                pass
            for i, val in enumerate(reversed(deriv[:start])):
                if val >= 0:
                    # print(i,val)
                    break
                else:
                    pass
            for j, val in enumerate(reversed(deriv[:start - i])):
                if val < 0:
                    # print(j,val)
                    # check to see if it's just noise
                    if deriv[start - 1 - i - j - 1] >= 0:
                        continue
                    else:
                        # print(j,val)
                        break
                else:
                    pass
            return start - i - j

    def find_local_max(deriv, start=0, direction=1):
        if direction == 1:  # positive direction
            # find negative
            for i, val in enumerate(deriv[start:]):
                if val >= 0:
                    # print(i,val)
                    break
                else:
                    pass
            for j, val in enumerate(deriv[start + i:]):
                if val < 0:
                    # print(j,val)
                    # check to see if it's just noise
                    if deriv[start + i + j + 1] >= 0:
                        continue
                    else:
                        # print(j,val)
                        break
                else:
                    pass
            return start + i + j - 1
        else:
            if start == 0:
                start = len(deriv)
                # note that we are using len(deriv) and data[len(deriv)] is out of bounds so indexing will be weird here!
            else:
                pass
            for i, val in enumerate(reversed(deriv[:start])):
                if val <= 0:
                    # print(i,val)
                    break
                else:
                    pass
            for j, val in enumerate(reversed(deriv[:start - i])):
                if val > 0:
                    # print(j,val)
                    # check to see if it's just noise
                    if deriv[start - 1 - i - j - 1] <= 0:
                        continue
                    else:
                        # print(j,val)
                        break
                else:
                    pass
            return start - i - j - 1

    def aveX(data, end, direction=1):
        # end is the index value of the last value in the end of the sequence...inclusive! don't accidentally not include it
        if direction == 1:
            # sum(data[:end+1])/(end+1)
            return np.mean(data[:end + 1])
        else:
            # sum(data[end:])/(len(data)-end)
            return np.mean(data[end:])

    def standard_dev(data, minInd, maxInd, direction=1):
        stdev = []
        for val in [maxInd, minInd]:
            aveVal = aveX(data, val, direction)
            if direction == 1:
                sd = np.sqrt(np.mean(np.square(data[:val + 1] - aveVal)))
            else:
                sd = np.sqrt(np.mean(np.square(data[val:] - aveVal)))
            stdev.append(sd)
        return stdev

    def search(data, multiplier=3, noisy=False):
        deriv = np.gradient(data)
        time_blocks = []
        for direction in [1, -1]:
            start = 0
            searching = True
            while searching:
                local_min = find_local_min(deriv, start, direction)
                if noisy & (direction == 1):
                    local_max = local_min + 5
                elif noisy & (direction == -1):
                    local_max = local_min - 5
                else:
                    local_max = find_local_max(deriv, local_min, direction)
                stdev = standard_dev(data, local_min, local_max, direction)
                if (stdev[0] > multiplier * stdev[1]) & (stdev[0] != 0) & (stdev[1] != 0):
                    # good
                    searching = False
                    break
                    # exit()
                else:
                    start = local_min
                    if (direction == 1) & (start >= len(data) - 1):
                        print("reached the end:", start)
                        exit()
                    elif (direction == -1) & (start <= 1):
                        print("reached the end:", start)
                        exit()
                    else:
                        pass
                        # print(local_min, "...didn't work")
                        # _ = input("%s, %s, %s ...didn't work" % (direction, local_min, local_max))
            print("direction:", direction, local_min, local_max)
            time_blocks.append(local_min)
        return time_blocks

    def fit_gauss(x, mu, sig, A):
        # x numpy array
        y = A * np.exp(-np.square(x - mu) / (2 * np.square(sig)))
        return y

    def fit_trap(wave, a, b, c, d, e, g):
        # assuming the index is also the position
        x = np.arange(len(wave))
        y = np.zeros(shape=x.shape, dtype=float)

        y[a + 1:b + 1] = e * ((x[a + 1:b + 1] - a) / (b - a))
        y[b + 1:c + 1] = np.exp(((np.log(g) - np.log(e)) * x[b + 1:c + 1] + (c * np.log(e) - b * np.log(g))) / (c - b))
        y[c + 1:d + 1] = g * ((d - x[c + 1:d + 1]) / (d - c))

        return y

    def residual(params, *args):
        # print(args, type(args))
        wave = np.array(args, dtype=float)
        # print(type(wave),wave.shape)
        x = np.arange(wave.shape[0])
        y = np.zeros(shape=x.shape, dtype=float)
        parvals = params.valuesdict()

        amplitude_1 = parvals['amp_1']
        center_1 = parvals['cen_1']
        half_width_1 = parvals['wid_1']

        amplitude_2 = parvals['amp_2']
        center_2 = parvals['cen_2']
        half_width_2 = parvals['wid_2']

        a = int(parvals['a'])
        b = int(parvals['b'])
        c = int(parvals['c'])
        d = int(parvals['d'])
        e = parvals['e']
        g = parvals['g']

        # print(a,b,c,d,e,g, amplitude_1,center_1,half_width_1, amplitude_2,center_2,half_width_2)

        # trap
        y[a + 1:b + 1] = e * ((x[a + 1:b + 1] - a) / (b - a))
        y[b + 1:c + 1] = np.exp(((np.log(g) - np.log(e)) * x[b + 1:c + 1] + (c * np.log(e) - b * np.log(g))) / (c - b))
        y[c + 1:d + 1] = g * ((d - x[c + 1:d + 1]) / (d - c))

        # gaus 1
        y += amplitude_1 * np.exp(-np.square(x - center_1) / (2 * np.square(half_width_1)))

        # gaus 2
        y += amplitude_2 * np.exp(-np.square(x - center_2) / (2 * np.square(half_width_2)))

        return wave - y

    def fit(params, wave):
        x = np.arange(len(wave))
        y = np.zeros(shape=x.shape, dtype=float)
        ytrap = np.zeros(shape=x.shape, dtype=float)
        y1 = np.zeros(shape=x.shape, dtype=float)
        y2 = np.zeros(shape=x.shape, dtype=float)
        parvals = params.valuesdict()
        '''
        amp_1=result.params['amp_1'].value
        cen_1=result.params['cen_1'].value
        wid_1=result.params['wid_1'].value
        amp_2=result.params['amp_2'].value
        cen_2=result.params['cen_2'].value
        wid_2=result.params['wid_2'].value,
        a=result.params['a'].value
        b=result.params['b'].value
        c=result.params['c'].value
        d=result.params['d'].value
        e=result.params['e'].value
        g=result.params['g'].value
        '''

        amplitude_1 = parvals['amp_1']
        center_1 = parvals['cen_1']
        half_width_1 = parvals['wid_1']

        amplitude_2 = parvals['amp_2']
        center_2 = parvals['cen_2']
        half_width_2 = parvals['wid_2']

        a = int(parvals['a'])
        b = int(parvals['b'])
        c = int(parvals['c'])
        d = int(parvals['d'])
        e = parvals['e']
        g = parvals['g']

        # trap
        ytrap[a + 1:b + 1] = e * ((x[a + 1:b + 1] - a) / (b - a))
        ytrap[b + 1:c + 1] = np.exp(
            ((np.log(g) - np.log(e)) * x[b + 1:c + 1] + (c * np.log(e) - b * np.log(g))) / (c - b))
        ytrap[c + 1:d + 1] = g * ((d - x[c + 1:d + 1]) / (d - c))

        # gaus 1
        y1 = amplitude_1 * np.exp(-np.square(x - center_1) / (2 * np.square(half_width_1)))

        # gaus 2
        y2 = amplitude_2 * np.exp(-np.square(x - center_2) / (2 * np.square(half_width_2)))

        y = ytrap + y1 + y2
        return y, ytrap, y1, y2



    if isinstance(data_pair, GTMOEPImagePair):
        pass
        # train_list = data_pair.get_train_batch()
        # test_list = data_pair.get_test_batch()
        # new_train_list = []
        # new_test_list = []
        # try:
        #
        #     for train, test in zip(train_list, test_list):
        #         X = [x + 1 for x in range(train.shape[1])]
        #         for dimension in range(train.shape[0]):
        #             guess = [1, 1, 1, 1, 1, 1]
        #             try:
        #                 popt, pcov = curve_fit(gaussian, X, train[dimension], p0=guess, method='lm', maxfev=5000)
        #                 fit = gaussian(X, *popt)
        #                 data = fit
        #                 # print('ANSWER', popt[0],popt[3])
        #
        #                 # max_locs = [np.round(popt[0]), np.round(popt[3])]
        #                 # peak_locs = max_locs  # concat two lists
        #                 # not_peaks = [i for i in np.arange(len(train[dimension])) if i not in peak_locs]
        #                 # train[dimension][not_peaks] = 0
        #
        #             except RuntimeError:
        #                 print("Error - curve_fit failed")
        #                 not_peaks = [i for i in np.arange(len(train[dimension]))]
        #                 train[dimension][not_peaks] = 0
        #
        #         new_test_list.append(test)
        #         new_train_list.append(train)
        #
        #     data_pair.set_train_batch(new_train_list)
        #     data_pair.set_test_batch(new_test_list)
        # except:
        #     data_pair.clear()
        #     raise
    else:
        reflection_buffer = 2000
        get_echo = True
        noisy = False
        withQuad = True

        data_list = []
        for data_set in [data_pair.get_train_data(), data_pair.get_test_data()]:
            instances = cp.deepcopy(data_set.get_instances())
            for instance in instances:
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Feature to Features available for my_gaussian_peak'
                    )
                elif mode is STREAM_TO_STREAM:
                    data = instance.get_stream().get_data()
                elif mode is STREAM_TO_FEATURES:
                    data = instance.get_stream().get_data()
                X = [x for x in range(len(data[0]))]
                new_data = []
                for row in data:
                    try:
                        row = np.array(row, dtype=float)
                        transmitted = row[:reflection_buffer]
                        reflected = data[reflection_buffer:]

                        # Data Pretreatment

                        ## Selection of the Effective Part
                        # def find_effective_region(data):
                        transmit_region = search(transmitted, multiplier=10)
                        reflect_region = search(reflected, multiplier=5)

                        # denoise
                        # Make a parameter
                        noise_sampling_count = 100
                        noise_samples = np.concatenate(
                            [reflected[reflect_region[0] - noise_sampling_count:reflect_region[0]],
                             reflected[reflect_region[1]:reflect_region[1] + noise_sampling_count]])
                        m_noise = np.mean(noise_samples)
                        # Make a parameter
                        lambda_noise = 1.1  # 1.1-1.5
                        threshold_noise = m_noise * lambda_noise
                        reflected[reflected < m_noise] = threshold_noise

                        # smooth
                        # assume halfwidth is the width at half height
                        transmitted_sigma = 0.5
                        halfwidth = transmitted_sigma
                        gauss_sigma = halfwidth / 2.355  # http://hyperphysics.phy-astr.gsu.edu/hbase/Math/gaufcn2.html
                        kernel = [1, 1, 1]  # http://dev.theomader.com/gaussian-kernel-calculator/
                        # gaussian_filter(reflected[reflect_region[0]:reflect_region[1]], sigma=, )
                        smooth_echo = convolve(input=reflected[reflect_region[0]:reflect_region[1]],
                                               weights=kernel)
                        super_ori_echo = deepcopy(smooth_echo)

                        # if withQuad:
                        init_a = smooth_echo.argmax()  # round( len(smooth_echo)*.10 )
                        init_b = init_a + round(len(smooth_echo) * .05)
                        init_c = round(len(smooth_echo) * .80)
                        init_d = round(len(smooth_echo) * .90)
                        init_e = smooth_echo[init_b]
                        init_g = smooth_echo[init_c]
                        trap = fit_trap(smooth_echo, init_a, init_b, init_c, init_d, init_e, init_g)

                        smooth_echo = smooth_echo - trap

                        # else:
                        # peak detection
                        k = 50.0  # 1.5-3.0 ...empirical value direct proprotional to the strength of peak detection
                        C_threshold = k * m_noise  # only local maxima with amplitude larger than threshold will get selected
                        # smooth_echo = smooth_echo[ smooth_echo>C_threshold ]
                        smooth_echo[smooth_echo < C_threshold] = C_threshold
                        ori_echo = deepcopy(smooth_echo)
                        gaus = []
                        amp = []
                        cen = []
                        wid = []
                        # while smooth_echo.max() > C_threshold:
                        for _ in range(2):
                            x_max1 = smooth_echo.argmax()
                            d = np.gradient(smooth_echo)
                            dd = np.gradient(d)
                            infl_1a = find_local_max(dd, start=x_max1, direction=-1)
                            infl_1b = find_local_min(dd, start=x_max1, direction=1)
                            halfwidth_1 = np.abs(infl_1b - infl_1a) / 2.
                            amplitude_1 = smooth_echo[x_max1]
                            print(amplitude_1)
                            # first gaussian fit
                            amp.append(amplitude_1)
                            cen.append(x_max1)
                            wid.append(halfwidth_1)
                            gaus_1 = fit_gauss(x=np.arange(len(smooth_echo)), mu=x_max1, sig=halfwidth_1, A=amplitude_1)
                            smooth_echo = smooth_echo - gaus_1
                            gaus.append(gaus_1)

                        # minimize
                        params = Parameters()
                        params.add('amp_1', value=amp[0], min=0)
                        params.add('cen_1', value=cen[0], min=0)
                        params.add('wid_1', value=wid[0], min=0)

                        params.add('amp_2', value=amp[1], min=0)
                        params.add('cen_2', value=cen[1], min=0)
                        params.add('wid_2', value=wid[1], min=0)

                        params.add('a', value=init_a, min=0)
                        params.add('b', value=init_b, min=0)
                        params.add('c', value=init_c, min=0)
                        params.add('d', value=init_d, min=0)
                        params.add('e', value=init_e, min=0)
                        params.add('g', value=init_g, min=0)

                        result = minimize(residual, params, args=(super_ori_echo))

                        fit_wave, ytrap, y1, y2 = fit(result.params, super_ori_echo)

                        if mode is STREAM_TO_STREAM:
                            new_data.append(fit_wave)
                        elif mode is FEATURES_TO_FEATURES:
                            new_data.append([result.params['cen_1'], result.params['cen_2']])


                    except RuntimeError:
                        print("Error - curve_fit failed")
                new_data = np.array(new_data)
                if mode is FEATURES_TO_FEATURES:
                    raise ValueError(
                        'No Feature to Features available for my_progressive_waveform_decomposition'
                    )
                elif mode is STREAM_TO_STREAM:
                    instance.get_stream().set_data(new_data)
                elif mode is STREAM_TO_FEATURES:
                    instance.get_features().set_data(new_data, labels=np.array(['cen_1', 'cen_2']))
            new_data_set = GTMOEPData(instances)
            data_list.append(new_data_set)
        data_pair = GTMOEPDataPair(train_data=data_list[0],
                                   test_data=data_list[1])
    gc.collect()
    return data_pair
