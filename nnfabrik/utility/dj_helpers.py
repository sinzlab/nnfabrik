# helper functions that the datajoint tables are requiring
import hashlib

def make_hash(config):
    """"
    takes any non-nested dict to return a 32byte hash
        :args: config -- dictionary. Must not contain objects or nested dicts
        :returns: 32byte hash
    """
    hashed = hashlib.md5()
    for k, v in sorted(config.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()
