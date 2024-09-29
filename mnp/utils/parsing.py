import argparse
import collections
from typing import Any

class dotdict(dict):
    """
    From user derek73 at 
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    and user Trevor Gross in the comments.
    
    dot.notation access to dictionary attributes.
    """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getattr__(self, key:str)-> Any:
        return self.__getitem__(key)


class Map(dict):
    """
    From user epool at
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary

    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v
    def __getattr__(self, attr):
        return self.get(attr)
    def __setattr__(self, key, value):
        self.__setitem__(key, value)
    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})
    def __delattr__(self, item):
        self.__delitem__(item)
    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def parse_args():
    """
    Adapted from user wim at
    https://stackoverflow.com/questions/21920989/parse-non-pre-defined-argument
    """

    parser = argparse.ArgumentParser()
    known, unknown_args = parser.parse_known_args()

    # initialize every argument value with a list, in case some arguments
    # receive more than one value
    unknown_options = collections.defaultdict(list)
    key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg[2:]
        else:
            unknown_options[key].append(arg)
    
    # simplify single-valued arguments from lists to strings
    for k,v in unknown_options.items():
        if len(v) == 1:
            unknown_options[k] = v[0]

    # simplify empty arguments from lists to None
    # convert strings 'None' to None
    for k,v in unknown_options.items():
        if len(v) == 0 or v == 'None':
            unknown_options[k] = None

    # convert ints and floats to appropriate types
    for k,v in unknown_options.items():
        try:
            # if we can cast to a number, cast to either float or int
            float(v)
            if '.' in v:
                unknown_options[k] = float(v)
            else:
                unknown_options[k] = int(v)
        except (ValueError, TypeError):
            pass

    # convert 'True' or 'False' to True or False
    for k,v in unknown_options.items():
        if v == 'True':
            unknown_options[k] = True
        elif v == 'False':
            unknown_options[k] = False


    return dotdict(unknown_options), unknown_options