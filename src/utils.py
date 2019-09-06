import os
CODEPATH = os.path.dirname(__file__)
PROJPATH = os.path.dirname(CODEPATH)

def make_config_str(args):
    """
    sometimes the config string returned may be too long for a filename...
    :param args: args should be a argparser namespace
    :return:
    """
    kwargs = vars(args)
    kwstr = {}
    # replace [ to (, ] to ) to avoid linux fileNotFound problem...
    for key, value in kwargs.items():
        if isinstance(value, list):
            kwstr[key] = list2str(value)
        else:
            kwstr[key] = str(value)
    string = '_'.join(['%s~%s' %(key, kwstr[key]) for key in kwstr])
    return string

def list2str(ls):
    listStr = str(ls)
    newStr = listStr.replace('[', '(').replace(']', ')')
    return newStr

def log(obj, f):
    obj_str = str(obj)
    print(obj_str)
    print(obj_str, file = f)

if __name__ == "__main__":
    # make_config_str({"a": 1, "b": [1, 2]})
    pass