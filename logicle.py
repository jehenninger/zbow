def default_transform_data(raw, params):
    import numpy as np
    import pandas as pd
    import subprocess

    t = 262144
    m = 4.5
    a = 0

    # calculate default widths
    w = []
    count = 0

    transformed_data = {}

    for p in params:
        w.append((m - np.log10(t / np.abs(np.min(raw[p])))) / 2)

        if w[count] < 0:
            w[count] = 0

        # default transform
        command = ['bin/logicle/logicle.out']
        transform_parameters = [t.__str__(), w[count].__str__(), m.__str__(), a.__str__()]

        data = raw[p].tolist()
        data_as_string = [str(i) for i in data]

        command.extend(transform_parameters)
        command.extend(data_as_string)

        output = subprocess.run(command, stdout=subprocess.PIPE)

        temp_data = output.stdout.decode("utf-8").splitlines()
        temp_data = [float(i) for i in temp_data]

        transformed_data.update({p: temp_data})

        count = count + 1

    transformed_data = pd.DataFrame(data=transformed_data, columns=params)
    return transformed_data


def custom_transform_data(raw, params):
    import pandas as pd
    import subprocess

    t = 262144
    m = 4.5
    a = 0

    # calculate default widths
    w = [1.5, 1.75, 1.75]  # order is RGB
    count = 0

    transformed_data = {}

    for p in params:

        # custom transform
        command = ['bin/logicle/logicle.out']
        transform_parameters = [t.__str__(), w[count].__str__(), m.__str__(), a.__str__()]

        data = raw[p].tolist()
        data_as_string = [str(i) for i in data]

        command.extend(transform_parameters)
        command.extend(data_as_string)

        output = subprocess.run(command, stdout=subprocess.PIPE)

        temp_data = output.stdout.decode("utf-8").splitlines()
        temp_data = [float(i) for i in temp_data]

        transformed_data.update({p: temp_data})

        count = count + 1


    transformed_data = pd.DataFrame(data=transformed_data, columns=params)
    return transformed_data

