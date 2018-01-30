def default_transform_data(raw, params):
    import numpy as np
    import subprocess

    t = 262144
    m = 4.5
    a = 0

    # calculate default widths
    w = []
    count = 0

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

        # for i in range(0, raw.__len__()):
        #     process = subprocess.run([command, t.__str__(), w[count].__str__(), m.__str__(), a.__str__(),
        #                               raw[p][i].__str__()], stdout=subprocess.PIPE)
        #
        #     transformed_data[p][i] = process.stdout

        output = subprocess.run(command, stdout=subprocess.PIPE)

        transformed_data = output.stdout.split("\n")
        print(transformed_data)

        count = count + 1
        # @START need a way to read the output of the logicle command line. Currently, it's a byte stream.

   #  return transformed_data


def custom_transform_data(raw, params):
    import numpy as np
    import subprocess

    t = 262144
    m = 4.5
    a = 0

    transformed_data = raw

    # calculate default widths
    w = []
    count = 0

    for p in params:
        w.append((m - np.log10(t / np.abs(np.min(raw[p])))) / 2)

        if w[count] < 0:
            w[count] = 0

        # default transform
        command = 'bin/logicle/logicle.out'

        for i in range(0, raw.__len__()):
            process = subprocess.run([command, t.__str__(), w[count].__str__(), m.__str__(), a.__str__(),
                                      raw[p][i].__str__()], stdout=subprocess.PIPE)

            transformed_data[p][i] = process.stdout.__float__()

        count = count + 1
