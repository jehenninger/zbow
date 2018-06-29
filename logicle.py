import numpy as np
import pandas as pd
import math
import subprocess
import sys
import os

def default_transform_data(raw, params):

    t = 262144
    m = 4.5
    a = 0

    # calculate default widths
    w = []
    count = 0

    transformed_data = {}

    num_of_chunks = int(math.ceil(len(raw)/1000))  # find how many chunks of 1000 we need
    raw_chunks = np.array_split(raw, num_of_chunks)

    for p in params:
        w.append((m - np.log10(t / np.abs(np.min(raw[p])))) / 2)

        if w[count] < 0:
            w[count] = 0

        # default transform

        transform_parameters = [t.__str__(), w[count].__str__(), m.__str__(), a.__str__()]

        temp_data = []
        for j in range(num_of_chunks):
            if getattr(sys, 'frozen', False):
                # running in a bundle
                bundle_dir = sys._MEIPASS
                command = [os.path.join(bundle_dir, 'bin/logicle/logicle.out')]
            else:
                # running live
                command = ['bin/logicle/logicle.out']

            data = raw_chunks[j]
            data = data[p].tolist()
            data_as_string = [str(i) for i in data]

            command.extend(transform_parameters)
            command.extend(data_as_string)

            output = subprocess.run(command, stdout=subprocess.PIPE)
            temp_data_chunk = output.stdout.decode("utf-8").splitlines()
            temp_data_chunk = [float(i) for i in temp_data_chunk]

            temp_data.extend(temp_data_chunk)

        transformed_data.update({p: temp_data})

        count = count + 1

    transformed_data = pd.DataFrame(data=transformed_data, columns=params)

    return transformed_data


def custom_transform_data(raw, params):

    t = 262144
    m = 4.5
    a = 0

    # calculate default widths
    w = [1.5, 1.75, 1.75]  # order is RGB
    count = 0

    transformed_data = {}


    # need to break data into chunks to not overload the command line
    num_of_chunks = int(math.ceil(len(raw)/500))  # find how many chunks of 1000 we need

    for p in params:

        # custom transform

        transform_parameters = [t.__str__(), w[count].__str__(), m.__str__(), a.__str__()]

        raw_chunks = np.array_split(raw, num_of_chunks)
        temp_data = []
        for j in range(num_of_chunks):
            if getattr(sys, 'frozen', False):
                # running in a bundle
                bundle_dir = sys._MEIPASS
                command = [os.path.join(bundle_dir, 'bin/logicle/logicle.out')]
            else:
                # running live
                command = ['bin/logicle/logicle.out']

            data = raw_chunks[j]
            data = data[p].tolist()
            data_as_string = [str(i) for i in data]

            command.extend(transform_parameters)
            command.extend(data_as_string)

            output = subprocess.run(command, stdout=subprocess.PIPE)

            temp_data_chunk = output.stdout.decode("utf-8").splitlines()
            temp_data_chunk = [float(i) for i in temp_data_chunk]

            temp_data.extend(temp_data_chunk)

        transformed_data.update({p: temp_data})

        count = count + 1

    transformed_data = pd.DataFrame(data=transformed_data, columns=params)
    return transformed_data

