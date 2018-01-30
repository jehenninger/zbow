# This was using a different method call to the command line.
# #!/Users/jon/anaconda/envs/py27/bin/python
#
# import sys
#
#
# def fcs_read(file_name):
#     import cytoflow as flow
#
#     tube = flow.Tube(file=file_name)
#     sample = flow.ImportOp(tubes=[tube])
#     experiment = sample.apply()
#
#     return experiment.channels, experiment.data.shape
#
#
# f_name = str(sys.argv[1])
# parameters, size = fcs_read(f_name)
# print(parameters)
# print('\n')
# print(size)

# New method

