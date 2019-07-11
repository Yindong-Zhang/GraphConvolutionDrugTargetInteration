

def make_config_str(args):
    """
    sometimes the config string returned may be too long for a filename...
    :param args: args should be a argparser namespace
    :return:
    """
    kwargs = vars(args)
    string = '_'.join(['%s~%s' %(key, kwargs[key]) for key in kwargs])
    return string