import hashlib
import inspect


def sanitize_params_dict(params):
    new_params = {}
    for key, value in params.iteritems():
        if not inspect.ismethod(value):
            new_params[key] = value
    return new_params


def get_library_hash(file_list=None):
    if file_list is None:
        file_list = ['ibeoCSVImporter.py', 'SequenceWrangler.py']
    hash_value = 0
    for full_path in file_list:
        hash_value += abs(hash(hashlib.md5(open(full_path, 'rb').read()).hexdigest()))
    return hash_value
