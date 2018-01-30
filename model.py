import subprocess
import pickle
import pandas as pd

class SessionData:
    def __init__(self):
        # variables
        self.file_name = str
        self.sample_name = str
        self.path_name = str
        self.params = tuple
        self.data = pd.DataFrame

    # methods

    def load_fcs_file(self):
        # Old method to call command line
        # command = '/Users/jon/PycharmProjects/zbow/fcs_read.py %s' % self.file_name
        #
        # process = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        # return_code = process.returncode
        # output = process.stdout
        # print('Python2 finished with return code %d\n' % return_code)
        # return output

        import fcsparser

        def fcs_read(file_name):


            # read in the data
            meta, data = fcsparser.parse(file_name, meta_data_only=False, reformat_meta=True)
            params = meta['_channel_names_']

            return params, data

        params, data = fcs_read(self.file_name)
        return params, data

    def transform_data(data):
        # logicle defaults
        T = 2 ^ 18
        M = 4.5
        A = 0

        # calculate default widths



            input_data = pickle.dumps(data)
            command = 'bin/logicle/logicle.out'

            process = subprocess.run(command, input=input_data, stdout=subprocess.PIPE)
            return_code = process.returncode
            print('Transform data finished with return code %d\n' % return_code)
            print(process.stdout)
            output = pickle.loads(process.stdout)
            print(output)
            return output


