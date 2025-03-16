import collections
import copy
import json
import numpy as np

def process_data_structures(data_obj, verbose=False):
    if type(data_obj) is dict and '_value' in data_obj:
        if '_string' not in data_obj:
            keyType = eval(data_obj['_type'])
            return keyType(data_obj['_value'])
        elif data_obj['_type'] == 'python_object (type = float32)':
            return np.float32(data_obj['_string'])
        else:
            return None
    elif type(data_obj) is dict:
        new_dict = {}
        for key, value in data_obj.items():
            new_dict[key] = process_data_structures(value, verbose)
        return new_dict
    elif type(data_obj) is list:
        new_list = []
        for value in data_obj:
            new_list.append(process_data_structures(value, verbose))
        return new_list
    return data_obj


class JSONArgs:
    def __init__(self, json_file, verbose=False):
        with open(json_file, 'r') as f:
            self._raw_data = json.load(f)
            self._data = copy.deepcopy(self._raw_data)

        self._process_data_structures(verbose)

    def _process_data_structures(self, verbose):
        for key, value in self._data.items():
            processed_value = process_data_structures(value, verbose)
            if processed_value is not None:
                self._data[key] = processed_value
            else:
                if verbose: print(' [ utils/json_args ] Unable to process complex data structure:', key)
                self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    def __contains__(self, key):
        return key in self._data

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'JSONArgs' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == '_data' or key == '_raw_data':
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key):
        if key == '_data' or key == '_raw_data':
            super().__delattr__(key)
        else:
            del self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __eq__(self, other):
        return self._data == other

    def __ne__(self, other):
        return self._data != other

    def __hash__(self):
        return hash(tuple(sorted(self._data.items())))

    def __copy__(self):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._data = self._data.copy()
        return new_instance

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance
        new_instance._data = copy.deepcopy(self._data, memo)
        return new_instance

    def copy(self):
        return self.__copy__()

    def deepcopy(self, memo=None):
        if memo is None:
            memo = {}
        return self.__deepcopy__(memo)

    def to_dict(self):
        return self._data

    def to_json(self):
        return json.dumps(self._data)

    def to_file(self, json_file):
        with open(json_file, 'w') as f:
            json.dump(self._data, f)

    def __call__(self, key):
        return self._data[key]
