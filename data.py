"""
Programmed by Jason Zutty
Modified by VIP Team
Implements a new data type for use with DEAP
"""

import numpy as np
import pandas as pd
from keras.utils import img_to_array, load_img
import copy as cp
import sys
import os
import shutil
import pickle
import hashlib
import re
import gzip
import itertools
import operator
from pathlib import Path

def load_feature_data_from_file(filename, batch_size=None):
    """
    Loads feature data written in a typical csv format:
    First row is the header starting with pound, or no header
    remaining rows are data.

    Args:
        filename: name of data file

    Returns:
        A GTMOEP Object

    """
    points = []
    with gzip.open(filename, 'rb') as my_file:
        first_line = True
        label_data = np.array([])
        class_data = np.array([])
        feature_data = np.array([[]])
        for line in my_file:
            line = line.decode('utf-8')
            line = line.strip().split(',')
            if first_line and line[0].startswith('#'):
                first_line = False
                label_data = np.array(line[:-1])
                continue
            elif first_line:
                first_line = False
                label_data = np.arange(len(line)-1)

            class_data = np.array([np.float(line[-1])])
            feature_data = np.array([line[:-1]], dtype='d')

            point = GTMOEPDataInstance(target=class_data)
            point.set_stream(
                StreamData(
                    np.array([[]]),
                    np.array([])
                    )
                )
            point.set_features(
                FeatureData(
                    feature_data,
                    label_data
                    )
                )
            points.append(point)

    return GTMOEPData(points)

def load_images_from_file(base_directory, batch_size):
    """
    Loads image files in a typical format

    Args:
        filename: name of data file

    Returns:
        A GTMOEP Object

    """
    # Setup directories for data assuming that folders are prepped properly
    directory = os.fsencode(base_directory)
    if not os.path.exists(base_directory + "gen_data"):
        os.makedirs(base_directory + "gen_data")
    gen_directory = os.fsencode(base_directory + "gen_data")
    fold = re.sub('[/]', '', base_directory[-7:])

    # checks whether data directory is empty
    if os.listdir(gen_directory) == []:
        # variable for counting queue size
        c = 0
        # stores queue for batch creation
        batch_queue = []
        for f in os.listdir(directory):
            filename = os.fsdecode(f)
            if filename.endswith(".jpeg"):
                c += 1
                img = load_img(base_directory + filename)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                batch_queue.append(x)
                # if queue is full, save and compress data then flush queue
                if c % batch_size == 0:
                    # adds type and number to end of filename for identification when loading files
                    # pickles and compresses data
                    f = gzip.GzipFile(base_directory + "gen_data/image_" + str(c) + "_data_" + fold + ".npy.gz", "w")
                    np.save(file=f, arr=np.vstack(batch_queue))
                    f.close()

                    batch_queue = []

        # open labels csv file
        labels = pd.read_csv(base_directory + "labels_" + fold +".csv", header=None).values

        # s variable for identifying label batches when loading data later on
        s = 1
        batch_queue = []
        for i in range(labels.shape[0]):
            row = labels[i, :]
            batch_queue.append(row)
            if (i + 1) % batch_size == 0:
                # adds type and number to end of filename for identification when loading later on
                # pickles and compresses data
                f = gzip.GzipFile(base_directory + "gen_data/image_" + str(s) + "_label_" + fold + ".npy.gz", "w")
                np.save(file=f, arr=np.vstack(batch_queue))
                f.close()

                s += 1
                batch_queue = []


    return ImageData(base_directory, os.fsdecode(gen_directory), batch_size, fold, None, [])

def generate_image_truth(image_data):
    """
    Helper method for creating truth data of image data

    Used in gtMOEP to map truth data to dataset for evaluation

    Args:
        image_data: object with image data location and parameters

    Returns:
        List of truth data numpy arrays
    """
    base_directory = image_data.get_base_directory()
    fold = image_data.get_fold()
    data_list = [os.fsdecode(f) for f in os.listdir(base_directory + "gen_data/") if os.fsdecode(f).endswith("data_" + fold + ".npy.gz")]

    label_list = []
    for i in range(len(data_list)):
        f = gzip.GzipFile(base_directory + "gen_data/image_" + str(i + 1) + "_label_" + fold + ".npy.gz", "r")
        label_array = np.load(f)
        f.close()

        label_list.append(label_array)

    return np.vstack(label_list).flatten()

def save_images_from_array(batch_list, image_data):
    """
    Helper method for saving images locally from array

    Args:
        batch_list: list of batches in numpy array
        image_data = given ImageData object

    """
    base_directory = image_data.get_base_directory()
    batch_size = image_data.get_batch_size()
    fold = image_data.get_fold()

    n = 1
    file_check = True
    collision_check = True
    # loop over possible directories to save data in until one is available
    while file_check:
        if os.path.exists(base_directory + "gen_data" + str(n)):
            for f in os.listdir(base_directory + "gen_data" + str(n)):
                if not os.fsdecode(f).endswith(fold + ".npy.gz"):
                    file_check = False
            if file_check:
                n += 1
        else:
            while collision_check:
                try:
                    os.makedirs(base_directory + "gen_data" + str(n))
                    collision_check = False
                except:
                    n += 1

            file_check = False

    # save directory for file generation
    new_directory = "gen_data" + str(n)
    gen_directory = base_directory + new_directory

    c = 0
    # loop over every batch and save + compress the batch
    for batch in batch_list:
        c += batch_size
        f = gzip.GzipFile(gen_directory + "/image_" + str(c) + "_data_" + fold + ".npy.gz", "w")
        np.save(file=f, arr=batch)
        f.close()

    new_history = image_data.get_history()
    new_history.append(image_data.get_gen_directory)

    return ImageData(base_directory, gen_directory, batch_size, fold, image_data.get_prediction(), new_history)

def load_train_data_from_images(image_data):
    """
    Helper method for converting image data into GTMOEP data

    Args:
        image_data: given image data object with parameters

    Returns:
        GTMOEPData of image data

    """
    gen_directory = image_data.get_gen_directory()
    base_directory = image_data.get_base_directory()
    batch_size = image_data.get_batch_size()
    fold = image_data.get_fold()
    batch_list = load_batch_from_data(image_data)

    # load in truth data
    truth_data = generate_image_truth(image_data)

    points = []
    i = 0
    c = batch_size
    for batch_data in batch_list:

        # use truth data as target
        labels = truth_data[i:c]

        i += batch_size
        c += batch_size

        # format batch data into correct shape for processing
        batch_data = np.squeeze(batch_data)

        if len(batch_data.shape) == 4:
            d1, d2, d3, d4 = batch_data.shape
            batch_data = batch_data.reshape((d1, d2*d3*d4))
        elif len(batch_data.shape) == 3:
            d1, d2, d3 = batch_data.shape
            batch_data = batch_data.reshape((d1, d2*d3))

        for x in range(batch_size):
            feature_data = batch_data[x]
            # reshape batched data for concatenation
            feature_data = feature_data.reshape((1, feature_data.shape[0]))
            # set class data to truth data labels
            class_data = np.array([labels[x]])
            # set label data to array of 0s because it is unused in image data
            label_data = np.zeros(batch_data.shape[1])

            point = GTMOEPDataInstance(target=class_data)
            point.set_stream(
                StreamData(
                    np.array([[]]),
                    np.array([])
                    )
                )
            point.set_features(
                FeatureData(
                    feature_data,
                    label_data
                    )
                )
            points.append(point)

    return GTMOEPData(points)

def load_test_data_from_images(image_data):
    """
    Helper method for converting image data into GTMOEP data

    Args:
        image_data: given image data object with parameters

    Returns:
        GTMOEPData of image data

    """
    gen_directory = image_data.get_gen_directory()
    base_directory = image_data.get_base_directory()
    predict_list = image_data.get_prediction()
    batch_size = image_data.get_batch_size()
    fold = image_data.get_fold()
    batch_list = load_batch_from_data(image_data)

    points = []
    s = 0
    for batch_data in batch_list:

        if predict_list is not None:
            labels = np.squeeze(predict_list[s])

        s += 1

        # format batch data into correct shape for processing
        batch_data = np.squeeze(batch_data)

        if len(batch_data.shape) == 4:
            d1, d2, d3, d4 = batch_data.shape
            batch_data = batch_data.reshape((d1, d2*d3*d4))
        elif len(batch_data.shape) == 3:
            d1, d2, d3 = batch_data.shape
            batch_data = batch_data.reshape((d1, d2*d3))

        for x in range(batch_size):
            feature_data = batch_data[x]
            # reshape batched data for concatenation
            feature_data = feature_data.reshape((1, feature_data.shape[0]))
            if predict_list is not None:
                # if data has a prediction set class data to predicted labels
                class_data = labels[x]
            else:
                # if data has no prediction set labels to an empty array
                class_data = np.array([[0]])
            # set label data to array of 0s because it is unused in image data
            label_data = np.zeros(batch_data.shape[1])

            point = GTMOEPDataInstance(target=class_data)
            point.set_stream(
                StreamData(
                    np.array([[]]),
                    np.array([])
                    )
                )
            point.set_features(
                FeatureData(
                    feature_data,
                    label_data
                    )
                )
            points.append(point)

    return GTMOEPData(points)

def load_batch_from_data(image_data):
    """
    Helper method for loading image batches stored locally

    Args:
        image_data: given image data object with parameters

    Returns:
        Numpy array of image data

    """
    gen_directory = image_data.get_gen_directory()
    fold = image_data.get_fold()
    data_list = [os.fsdecode(f) for f in os.listdir(gen_directory) if os.fsdecode(f).endswith("data_" + fold + ".npy.gz")]

    batch_list = []
    for filename in data_list:
        f = gzip.GzipFile(gen_directory + "/" + filename, "r")
        batch_data = np.load(f)
        f.close()
        batch_list.append(batch_data)

    return batch_list

def load_labels_from_data(image_data):
    """
    Helper method for loading label data stored locally

    Args:
        image_data: given image data object with parameters

    Returns:
        Numpy array of image labels

    """
    gen_directory = image_data.get_gen_directory()
    base_directory = image_data.get_base_directory()
    batch_size = image_data.get_batch_size()
    fold = image_data.get_fold()
    data_list = [os.fsdecode(f) for f in os.listdir(gen_directory) if os.fsdecode(f).endswith("data_" + fold + ".npy.gz")]

    s = 1
    batch_list = []
    for i in data_list:
        f = gzip.GzipFile(base_directory + "gen_data" + "/image_" + str(s) + "_label_" + fold + ".npy.gz", "r")
        label = np.load(f)
        f.close()
        label_data = np.ravel(label)

        # transform label array into proper format (batch_size, 1)
        label_data = np.reshape(label_data, (batch_size, -1))
        batch_list.append(label_data)
        s += 1

    return batch_list

def delete_image_data(train_data, test_data):
    """
    Helper method for deleting data stored locally

    Args:
        train_data: given image data object with parameters
        test_data: given image data object with parameters
    """
    train_history = train_data.get_history()
    test_history = test_data.get_history()

    if len(train_history) > 1 and len(test_history) > 1:
        i = 0
        for train_dir, test_dir in zip(train_history, test_history):
            if i > 0:
                shutil.rmtree(train_dir, ignore_errors=True)
                shutil.rmtree(test_dir, ignore_errors=True)
            i += 1


def load_many_to_one_from_file(filename, batch_size = 1):
    """
    Loads stream data written in a many to one format:
        time,data,time,data,time,data,...,truth_val
    per sample

    Args:
        filename: name of data file

    Returns:
        A GTMOEPData Object

    """
    points = []
    with gzip.open(filename, 'r') as my_file:
        # Start by reading the first line which specifies how many rows
        # per column, i.e. data will be written col,row,row,row,...col,...
        header = my_file.readline().decode('utf-8')
        # Check first line for N=Number
        match = re.search(r"N=(\d+)", header)
        if match:
            # Assign N to be what the file specifies
            N = int(match.group(1))
        else:
            # Default is one row per col, i.e. a vector
            N = 1
            # Reset the file pointer to the beginning of file
            my_file.seek(0)
        for line in my_file:
            line = line.decode('utf-8')
            times = []
            counts = []
            line = line.strip().split(',')
            depth = np.array([float(line[-1])])
            #waveform = [float(elem) for elem in line[:-1]]
            times = line[:-1:(N+1)]
            for row in np.arange(N):
                counts.append(line[(row+1):-1:(N+1)])

            counts = np.array(counts, dtype='d')
            times = np.array(times)
            #print counts.shape, times.shape
            #print times
            #print counts
            point = GTMOEPDataInstance(target=depth)
            point.set_stream(
                ManyToOne(
                    counts, times
                    )
                )
            point.set_features(
                FeatureData(
                    np.array([[]]),
                    np.array([])
                    )
                )
            points.append(point)
    return GTMOEPData(points)


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)]*n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def load_many_to_many_from_file(filename, batch_size=1):
    """
    This method is designed to read in data that uses a "stream"
    of data to predict a "stream" of data. A potential example of this paradigm
    would be a filter design problem
    The format of this data will be interlaced, such that you have:
    Sample_Data
    Truth_Data
    Sample_Data
    Truth_Data
    etc...
    Therefore we will always have an even number of lines
    """
    points = []
    with gzip.open(filename, 'r') as my_file:
        # Start by reading the first line which specifies how many rows
        # per column, i.e. data will be written col,row,row,row,...col,...
        header = my_file.readline().decode('utf-8')

        # Check first line for N=Number, M=Number
        match_N = re.search(r"N=(\d+)", header)
        N = int(match_N.group(1)) if match_N else 1
        match_M = re.search(r"M=(\d+)", header)
        M = int(match_M.group(1)) if match_M else N

        if not match_N and not match_M:
            # Reset the file pointer to the beginning of file
            my_file.seek(0)

        for lines in grouper(my_file, 2):
            lines = [line.decode('utf-8') for line in lines]
            if lines[1] is None:
                break

            times_1 = []
            times_2 = []
            counts_1 = []
            counts_2 = []
            line_1 = lines[0].strip().split(',')
            line_2 = lines[1].strip().split(',')

            times_1 = line_1[::(N+1)]
            for row in np.arange(N):
                counts_1.append(line_1[(row+1)::(N+1)])
            times_2 = line_2[::(M+1)]
            for row in np.arange(M):
                counts_2.append(line_2[(row+1)::(M+1)])

            counts_1 = np.array(counts_1, dtype='d')
            times_1 = np.array(times_1)
            counts_2 = np.array(counts_2, dtype='d')
            times_2 = np.array(times_2)

            if counts_1.shape[1] != counts_2.shape[1]:
                raise ValueError('Sample and truth data length mismatched: {}, {}'.format(counts_1.shape[1], counts_2.shape[1]))

            point = GTMOEPDataInstance(target=counts_2)
            point.set_stream(
                ManyToMany(
                    counts_1, times_1
                    )
                )
            point.set_features(
                FeatureData(
                    np.array([[]]),
                    np.array([])
                    )
                )
            points.append(point)
    return GTMOEPData(points)

# subclass based on truth (and data?)

class GTMOEPDataObject(object):
    """
    Serves as the parent class for GTMOEP data objects.

    Represented by three attributes:
        _data: an array containing data.
        _labels: an array labeling the data.

    """

    def __init__(self,
            data=np.array([[]]),
            labels=np.array([])
            ):
        # Let's call the methods of the class to set the data
        self.set_data(data, labels=labels)
        ## One dimensional data check
        #if data.shape == labels.shape:
        #    self._data = data
        #    self._labels = labels
        ## Two dimensional data check
        #elif len(data) == 0 and len(labels) == 0 or data.shape[1] == labels.shape[0]:
        #    self._data = data
        #    self._labels = labels
        #else:
        #    raise ValueError(
        #            'Data array and label array do not match in shape.'
        #            )


    def get_data(self):
        """
        Returns the data stored in the object.

        """
        return self._data

    def set_data(self, data, labels=None):
        """
        Sets the data object.
        Optionally takes in a corresponding set of labels for the data.

        Args:
            data: data object

        Raises:
            ValueError: if data is not all finite
            ValueError: if data array and label array do not match

        """
        if labels is None:
            labels = self._labels
        # For debugging purposes I am going to raise an exception if the data is not all finite,
        # I may leave this in, as the sci-kit methods seem to require all data be finite
        if len(data) > 0 and not np.all(np.isfinite(data)):
            raise ValueError('Non-finite data produced: ' + str(data) + ' ' + str(type(data)))

        # One dimensional data check
        if data.shape == labels.shape:
            self._data = data
            self._labels = labels
        # Two dimensional data check
        elif len(data) == 0 and len(labels) == 0 or data.shape[1] == labels.shape[0]:
            self._data = data
            self._labels = labels
        else:
            raise ValueError(
                    'Data array and label array do not match in shape.'
                    )

    def get_labels(self):
        """
        Return the labels associated with the objects data.

        """
        return self._labels

    def set_labels(self, labels, data=None):
        """
        Sets the labels associated with the data objects.
        Optionally takes in a a corresponding set of data for the labels.

        Args:
            labels: data object labels

        Raises:
            ValueError: if label array and data array do not match

        """
        if data == None:
            data = self._data

        if labels.shape == data.shape:
            self._labels = labels
            self._data = data
        else:
            raise ValueError(
                    'Label array and data array do not match in shape.'
                    )

    def type_check(self, other):
        """
        Checks to see if the types of self and other match.

        Raises:
            ValueError: if types of self and other do not match

        """
        if type(self) is not type(other):
            raise ValueError('The two objects are of different classes.')


    def __add__(self, other):
        """
        ADDITION

        Returns:
            A new object with (self data PLUS other data) and labels

        Raises:
            ValueError: if data sizes do not match

        """

        if isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = type(self)(
                    data=self.get_data() + other,
                    labels=self.get_labels()
                    )
            return new
        else:
            self.type_check(other)

            if self.get_data().shape == other.get_data().shape or (other.get_data().shape[0] == self.get_data().shape[0] and (other.get_data().shape[1] == 1 or self.get_data().shape[1] == 1)):
                new = type(self)(
                        data=self.get_data() + other.get_data(),
                        labels=self.get_labels()
                        )
                return new
            else:
                raise ValueError('Data sizes did not match.')

    def __sub__(self, other):
        """
        SUBTRACTION

        Returns:
            A new object with (self data MINUS other data) and labels

        Raises:
            ValueError: if data sizes do not match

        """

        if isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = type(self)(
                    data=self.get_data() - other,
                    labels=self.get_labels()
                )
            return new
        else:
            self.type_check(other)

            if self.get_data().shape == other.get_data().shape or (other.get_data().shape[0] == self.get_data().shape[0] and (other.get_data().shape[1] == 1 or self.get_data().shape[1] == 1)):
                new = type(self)(
                        data=self.get_data() - other.get_data(),
                        labels=self.get_labels()
                        )
                return new
            else:
                raise ValueError('Data sizes did not match.')

    def __mul__(self, other):
        """
        MULTIPLICATION

        Returns:
            A new object with (self data TIMES other data) and labels

        Raises:
            ValueError: if data sizes do not match

        """

        if isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = type(self)(
                    data=self.get_data() * other,
                    labels=self.get_labels()
                )
            return new
        else:
            self.type_check(other)

            if self.get_data().shape == other.get_data().shape or (other.get_data().shape[0] == self.get_data().shape[0] and (other.get_data().shape[1] == 1 or self.get_data().shape[1] == 1)):
                new = type(self)(
                        data=self.get_data() * other.get_data(),
                        labels=self.get_labels()
                        )
                return new
            else:
                raise ValueError('Data sizes did not match.')

    def __truediv__(self, other):
        """
        DIVISION

        Returns:
            A new object with (self data DIVIDED BY other data) and labels

        Raises:
            ValueError: if data sizes do not match

        """

        if isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = type(self)(
                    data=self.get_data() / other,
                    labels=self.get_labels()
                )
            return new
        else:
            self.type_check(other)
            # We can divide if shapes match or if either has one column, i.e. is a scalar
            if self.get_data().shape == other.get_data().shape or (other.get_data().shape[0] == self.get_data().shape[0] and (other.get_data().shape[1] == 1 or self.get_data().shape[1] == 1)):
                new = type(self)(
                        data=self.get_data() / other.get_data(),
                        labels=self.get_labels()
                        )
                return new
            else:
                raise ValueError('Data sizes did not match.')

    def __repr__(self):
        return '\n'.join(['Data:', str(self._data)])

    #def __setattr__(self,name,value):
    #   if name in ('_data', '_labels', '_truth'):
    #       raise AttributeError('Access %s through its set method' % name)
    #   else:
    #       super(GTMOEPDataObject,self).__setattr__(name,value)

class FeatureData(GTMOEPDataObject):
    """
    This class represents the parent class for feature type data.
    """
    def __init__(self,
            data=np.array([[]]),
            labels=np.array([])
            ):
        super(FeatureData, self).__init__(data, labels)


class StreamData(GTMOEPDataObject):
    """
    This class represents the parent class for stream type data.
    """
    def __init__(self,
            data=np.array([[]]),
            labels=np.array([])
            ):
        super(StreamData, self).__init__(data, labels)

class ImageData(object):
    """
    This class represents the parent class for image type data.
    """
    def __init__(self, base_directory, gen_directory, batch_size, fold, prediction_data, history):
        self.base_directory = base_directory
        self.gen_directory = gen_directory
        self.batch_size = batch_size
        self.fold = fold
        self.prediction_data = prediction_data
        self.history = history

    """
    Returns the base directory of the image data
    """
    def get_base_directory(self):
        return self.base_directory

    """
    Returns the gen directory of the image data
    """
    def get_gen_directory(self):
        return self.gen_directory

    """
    Returns the batch_size of the image data
    """
    def get_batch_size(self):
        return self.batch_size

    """
    Returns the fold id of the image data
    """
    def get_fold(self):
        return self.fold

    """
    Returns the prediction labels of the image data
    """
    def get_prediction(self):
        return self.prediction_data

    """
    Sets the predictions of the image data
    """
    def set_prediction(self, predictions):
        self.prediction_data = predictions

    """
    Sets the gen directory of the image data
    """
    def set_gen_directory(self, directory):
        self.gen_directory = directory

    """
    Returns the generated history of the image data
    """
    def get_history(self):
        return self.history

class OneToOne(StreamData):
    """
    This class represents stream data where the target data lines up
    on a one to one basis with the raw data, i.e. each point has a target value
    in the time series.
    """
    def __init__(self,
            data=np.array([[]]),
            labels=np.array([])
            ):

        super(OneToOne, self).__init__(data, labels)


    def slice(self, step, win):
        """
        Slices each instance into pieces of size 'win'.
        - win must be < len(self.labels)
        - step must be < win

        Args:
            win: window size (size of each 'chunk')
            step: step size (how far forward to move chunk's starting position after each iteration)

        """
        new_labels = []
        new_data = []
        new_target = []

        self._labels = self._labels.flatten()
        self._data = self._data.flatten()
        self._target = self._target.flatten()

        num_slices = (len(self._labels)-win+step)/step

        for i in np.arange(num_slices):
            new_labels.append(self._labels[(i*step):(i*step+win)])
            new_data.append(self._data[(i*step):(i*step+win)])
            new_target.append(self._target[(i*step):(i*step+win)])

        self._labels = np.array(new_labels)
        self._data = np.array(new_data)
        self._target = np.array(new_target)


class ManyToOne(StreamData):
    """
    This class represents stream data where there is only one point of
    target data per raw data, i.e. each series has a target value.
    """
    def __init__(self,
            data=np.array([[]]),
            labels=np.array([])
            ):

        super(ManyToOne, self).__init__(data, labels)



class ManyToMany(StreamData):
    """
    This class represents stream data where there is only one point of
    target data per raw data, i.e. each series has a target value.
    """
    def __init__(self,
            data=np.array([[]]),
            labels=np.array([])
            ):
        super(ManyToMany, self).__init__(data, labels)

    # def slice(self, step, win):
    #     """
    #     Slice each instance in to pieces of size N
    #     """
    #     new_labels = []
    #     new_data = []
    #     new_target = []

    #     self._labels = self._labels.flatten()
    #     self._data = self._data.flatten()
    #     self._target = self._target.flatten()

    #     num_slices = (len(self._labels)-win+step)/step

    #     for i in np.arange(num_slices):
    #         new_labels.append(self._labels[(i*step):(i*step+win)])
    #         new_data.append(self._data[(i*step):(i*step+win)])
    #         new_target.append(self._target[(i*step):(i*step+win)])

    #     self._labels = np.array(new_labels)
    #     self._data = np.array(new_data)
    #     self._target = np.array(new_target)

    # def combine_targets(self, other):
    #     if self.get_target().shape == other.get_target().shape:
    #         return np.logical_or(self.get_target(), other.get_target())
    #     else:
    #         raise ValueError('Target arrays have different sizes.')

class GTMOEPDataInstance(object):
    """
    This is a class where each implementation of it is a training or testing
    instance GTMOEP will operate on a deapDataInstance collection.
    Methods will handle the collection of these instances.

    Each deapDataInstance will contain two members
        data:  a numpy.ndarray
        truth: a scalar value on what the data should produce

    """

    def __init__(self,
            features=FeatureData(),
            stream=StreamData(),
            target=np.array([])
            ):
        self._features = None
        self._stream = None
        self._target = None
        self.set_features(features)
        self.set_stream(stream)
        self.set_target(target)

    def get_features(self):
        """
        Returns the feature object of the data instance.
        The feature object will be of some subclass of FeatureData.
        """
        return self._features

    def set_features(self, features):
        """
        Set the feature object of a data instance,
        features must be a subclass of FeatureData.

        Raises:
            ValueError: if feature is not a derived class of FeatureData

        """
        if isinstance(features, FeatureData):
            self._features = features
        else:
            raise ValueError(
                    'Features need to be a derived class of FeatureData.'
                    )

    def get_stream(self):
        """
        Returns the stream object of the data instance.
        The stream object will be of some subclass of StreamData.
        """
        return self._stream

    def set_stream(self, stream):
        """
        Set the stream object of a data instance.
        stream must be a subclass of StreamData.

        Raises:
            ValueError: if stream is not a derived class of StreamData

        """
        if isinstance(stream, StreamData):
            self._stream = stream
        else:
            raise ValueError('Stream needs to be a derived class of StreamData')

    def get_target(self):
        """
        Returns the target for the data, i.e. The value the data represents.
        """
        return self._target

    def set_target(self, target):
        """
        Set the target array
        """
        self._target = target

    # Pulled this from the GTMOEPDataObject class to keep one target for both stream
    # And features
    def combine_targets(self, other, func):
        """
        Abstract placeholder of parent method
        """
        if len(self.get_target().shape) == 1 and len(other.get_target().shape) == 1:
            return self.get_target() #or other.get_target()
        elif self.get_target().shape == other.get_target().shape:
            return func(self.get_target(), other.get_target())
        else:
            raise ValueError('Child class should implement this method')


    def __add__(self, other):
        """
        ADDITION

        Returns:
            A new object with (self features/stream PLUS other features/stream)

        Raises:
            ValueError: if type of other is incompatible with operation

        """

        if isinstance(other, GTMOEPDataInstance):
            new = GTMOEPDataInstance(target=self.combine_targets(other, operator.__add__))
            new.set_features(self.get_features() + other.get_features())
            new.set_stream(self.get_stream() + other.get_stream())
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPDataInstance(target=self.get_target())
            new.set_features(self.get_features() + other)
            new.set_stream(self.get_stream() + other)
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __sub__(self, other):
        """
        SUBTRACTION

        Returns:
            A new object with (self features/stream MINUS other features/stream)

        Raises:
            ValueError: if type of other is incompatible with operation

        """

        if isinstance(other, GTMOEPDataInstance):
            new = GTMOEPDataInstance(target=self.combine_targets(other, operator.__sub__))
            new.set_features(self.get_features() - other.get_features())
            new.set_stream(self.get_stream() - other.get_stream())
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPDataInstance(target=self.get_target())
            new.set_features(self.get_features() - other)
            new.set_stream(self.get_stream() - other)
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __mul__(self, other):
        """
        MULTIPLICATION

        Returns:
            A new object with (self features/stream TIMES other features/stream)

        Raises:
            ValueError: if type of other is incompatible with operation

        """

        if isinstance(other, GTMOEPDataInstance):
            new = GTMOEPDataInstance(target=self.combine_targets(other, operator.__mul__))
            new.set_features(self.get_features() * other.get_features())
            new.set_stream(self.get_stream() * other.get_stream())
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPDataInstance(target=self.get_target())
            new.set_features(self.get_features() * other)
            new.set_stream(self.get_stream() * other)
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __truediv__(self, other):
        """
        TRUE DIVISION
        In python 3.X there is the / operator, which typically returns a float, as well
        as a // operator (__floordiv__), which typically returns an int.

        Returns:
            A new object with (self features/stream DIVIDED BY other features/stream)

        Raises:
            ValueError: if type of other is incompatible with operation

        """

        if isinstance(other, GTMOEPDataInstance):
            new = GTMOEPDataInstance(target=self.combine_targets(other, operator.__truediv__))
            new.set_features(self.get_features() / other.get_features())
            new.set_stream(self.get_stream() / other.get_stream())
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPDataInstance(target=self.get_target())
            new.set_features(self.get_features() / other)
            new.set_stream(self.get_stream() / other)
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __repr__(self):
        return '\n'.join(['Features:', str(self.get_features()),
                          'Stream:', str(self.get_stream()),
                          'Target:', str(self.get_target())])


class GTMOEPData(object):
    """
    This class represents the collection of instances in to a single
    data set.
    """
    def __init__(self,
            instances=np.array([]),
            subject_ids=np.array([])
            ):
        """
        Subject ids are used for folding the data.
        """
        self._instances = []
        self.set_instances(instances)
        self._subject_ids = subject_ids

    def get_instances(self):
        """
        Returns the instance list.
        """
        return self._instances

    def set_instances(self, instances):
        """
        Sets the objects instances list with the given list of instances.

        Raises:
            ValueError: if instance not of GTMOEPDataInstances
        """

        if all(
            isinstance(instance, GTMOEPDataInstance) for instance in instances
            ):
            # The copy is here, because otherwise two GTMOEPData instances
            # Could share GTMOEPDataInstance instances, and a change to one
            # would propagate to the other
            self._instances = [instance for instance in instances]
        else:
            raise ValueError('Expected an array of GTMOEPDataInstances')

    def get_numpy(self):
        """
        Returns the data objects features as a numpy array.
        """
        return np.array([instance.get_features().get_data()[0, :]
                        for instance in self._instances])

    def __add__(self, other):
        """
        ADDITION

        Returns:
            A new set with (self instances PLUS other instances)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if  (isinstance(other, GTMOEPData) and
              len(self.get_instances()) == len(other.get_instances())
            ):
            new = GTMOEPData(subject_ids=self._subject_ids)
            new.set_instances([inst_1 + inst_2 for inst_1, inst_2
                               in
                               zip(self.get_instances(), other.get_instances())
                               ])
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPData(subject_ids=self._subject_ids)
            new.set_instances([inst_1 + other for inst_1
                                   in self.get_instances()])
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __sub__(self, other):
        """
        SUBTRACTION

        Returns:
            A new set with (self instances MINUS other instances)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if (isinstance(other, GTMOEPData)
             and len(self.get_instances()) == len(other.get_instances())
            ):
            new = GTMOEPData(subject_ids=self._subject_ids)
            new.set_instances([inst_1 - inst_2 for inst_1, inst_2
                               in
                               zip(self.get_instances(), other.get_instances())
                               ])
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPData(subject_ids=self._subject_ids)
            new.set_instances([inst_1 - other for inst_1
                                   in self.get_instances()])
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __mul__(self, other):
        """
        MULTIPLICATION

        Returns:
            A new set with (self instances TIMES other instances)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if (isinstance(other, GTMOEPData)
            and len(self.get_instances()) == len(other.get_instances())
            ):
            new = GTMOEPData(subject_ids=self._subject_ids)
            new.set_instances([inst_1 * inst_2 for inst_1, inst_2
                               in
                               zip(self.get_instances(), other.get_instances())
                               ])
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPData(subject_ids=self._subject_ids)
            new.set_instances([inst_1 * other for inst_1
                                   in self.get_instances()])
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __truediv__(self, other):
        """
        DIVISION

        Returns:
            A new set with (self instances DIVIDED BY other instances)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if (isinstance(other, GTMOEPData)
            and len(self.get_instances()) == len(other.get_instances())
            ):
            new = GTMOEPData(subject_ids=self._subject_ids)
            new.set_instances([inst_1 / inst_2 for inst_1, inst_2
                               in
                               zip(self.get_instances(), other.get_instances())
                               ])
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPData(subject_ids=self._subject_ids)
            new.set_instances([inst_1 / other for inst_1
                                   in self.get_instances()])
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __repr__(self):
        return '\n'.join([str(instance) for instance in self.get_instances()])


class GTMOEPDataPairIn(object):
    pass


class GTMOEPDataPair(GTMOEPDataPairIn):
    """
    This class groups together a training and testing data object,
    so that they may be passed around the genetic programming structure
    in tandem.
    """

    def __init__(self, train_data, test_data):
        self._train_data = None
        self._test_data = None

        self.set_train_data(train_data)
        self.set_test_data(test_data)

        if self._train_data is None:
            self._train_data = GTMOEPData()
        if self._test_data is None:
            self._test_data = GTMOEPData()

    def get_train_data(self):
        """
        Return the training data set,
        data is of class GTMOEPData.
        """
        return self._train_data

    def set_train_data(self, train_data):
        """
        Set the training data set,
        train_data must be of type GTMOEPData.
        """
        if isinstance(train_data, GTMOEPData):
            self._train_data = train_data

    def get_test_data(self):
        """
        Return the testing data set,
        data is of class GTMOEPData.
        """
        return self._test_data

    def set_test_data(self, test_data):
        """
        Return the testing data set,
        data is of class GTMOEPData.
        """
        if isinstance(test_data, GTMOEPData):
            self._test_data = test_data

    def get_sha1(self):
        """
        Return the sha1 hash of the data pair.
        """
        pickled_string = pickle.dumps(self, -1)
        hash_function = hashlib.sha1()
        hash_function.update(pickled_string)
        return hash_function.hexdigest()

    def __add__(self, other):
        """
        ADDITION

        Returns:
            A new GTMOEPDataPair with (self train_data/test_data PLUS other train_data/test_data)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if isinstance(other, GTMOEPDataPair):
            new = GTMOEPDataPair(
                self.get_train_data() + other.get_train_data(),
                self.get_test_data() + other.get_test_data()
                )
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPDataPair(
                self.get_train_data() + other,
                self.get_test_data() + other
                )
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __sub__(self, other):
        """
        SUBTRACTION

        Returns:
            A new GTMOEPDataPair with (self train_data/test_data MINUS other train_data/test_data)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if isinstance(other, GTMOEPDataPair):
            new = GTMOEPDataPair(
                self.get_train_data() - other.get_train_data(),
                self.get_test_data() - other.get_test_data()
                )
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPDataPair(
                self.get_train_data() - other,
                self.get_test_data() - other
                )
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __mul__(self, other):
        """
        MULTIPLICATION

        Returns:
            A new GTMOEPDataPair with (self train_data/test_data TIMES other train_data/test_data)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if isinstance(other, GTMOEPDataPair):
            new = GTMOEPDataPair(
                self.get_train_data() * other.get_train_data(),
                self.get_test_data() * other.get_test_data()
                )
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPDataPair(
                self.get_train_data() * other,
                self.get_test_data() * other
                )
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __truediv__(self, other):
        """
        DIVISION

        Returns:
            A new GTMOEPDataPair with (self train_data/test_data DIVIDED BY other train_data/test_data)

        Raises:
            ValueError: if type of other is incompatible with operation
        """

        if isinstance(other, GTMOEPDataPair):
            new = GTMOEPDataPair(
                self.get_train_data() / other.get_train_data(),
                self.get_test_data() / other.get_test_data()
                )
            return new
        elif isinstance(other, (int, float, bool)) or (isinstance(other, np.ndarray)
                                                     and len(other) == 1):
            new = GTMOEPDataPair(
                self.get_train_data() / other,
                self.get_test_data() / other
                )
            return new
        else:
            raise ValueError('Incompatible operation for type %s' %type(other))

    def __repr__(self):
        return '\n'.join(['Train', str(self.get_train_data()),
                          'Test', str(self.get_test_data())])

    def clear(self):
        """
        PASS

        Function is only needed in class GTMOEPImagePair
        """
        pass

class GTMOEPImagePair(GTMOEPDataPair):
    """
    This class is a subclass of GTMOEPDataPair designed to handle image data
    """
    def __init__(self, train_data, test_data):
        super(GTMOEPImagePair, self).__init__(train_data, test_data)

    def get_train_data(self):
        """
        Return the training data set as a GTMOEPData,
        data is of class ImageData.
        """
        return load_train_data_from_images(self._train_data)

    def get_train_batch(self):
        """
        Return a list containing train data in batches
        """
        return load_batch_from_data(self._train_data)

    def set_train_data(self, train_data):
        """
        Set the training data set
        """
        if isinstance(train_data, ImageData):
            self._train_data = train_data

    def set_train_batch(self, train_data):
        """
        Set the training data set,
        save new image data and location
        """
        self._train_data = save_images_from_array(train_data, self._train_data)

    def get_test_data(self):
        """
        Return the testing data set as a GTMOEPData,
        data is of class ImageData.
        """
        return load_test_data_from_images(self._test_data)

    def get_test_batch(self):
        """
        Return a list containing test data in batches
        """
        return load_batch_from_data(self._test_data)

    def set_test_data(self, test_data):
        """
        Set the testing data set
        """
        if isinstance(test_data, ImageData):
            self._test_data = test_data

    def set_test_batch(self, test_data):
        """
        Set the testing data set,
        save new image data and location
        """
        self._test_data = save_images_from_array(test_data, self._test_data)

    def get_train_labels(self):
        """
        Return a list containing training data labels in batches
        """
        return load_labels_from_data(self._test_data)

    def set_prediction(self, predictions):
        """
        Save prediction data locally
        """
        self._test_data.set_prediction(predictions)

    def clear(self):
        """
        Clear generated data
        """
        delete_image_data(self._train_data, self._test_data)

    def get_batch_size(self):
        """
        Return batch size
        """
        return self._train_data.get_batch_size()

    def get_base_directory(self):
        """
        Return base directory of training data
        """
        return self._train_data.get_base_directory()

    def set_gen_directories(self, file_name):
        self._train_data.set_gen_directory(get_base_directory() + file_name)
        self._test_data.set_gen_directory(get_base_directory() + file_name)
