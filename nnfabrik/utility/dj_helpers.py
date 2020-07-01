# helper functions for use with DataJoint tables

import warnings
from datetime import datetime
import hashlib
import datajoint as dj
from git import Repo, cmd
import numpy as np
import inspect
from datetime import date, datetime
from datajoint.utils import to_camel_case
from collections import OrderedDict, Iterable, Mapping

# try/except is necessary to support all versions of dj
try:
    # for versions < 0.12.6
    from datajoint.schema import Schema
except:
    # for versions >= 0.12.6
    from datajoint.schemas import Schema


def cleanup_numpy_scalar(data):
    """
    Recursively cleanups up a (potentially nested data structure of)
    objects, replacing any scalar numpy instance with the corresponding
    Python native datatype.
    """
    if isinstance(data, np.generic):
        if data.shape == ():
            data = data.item()
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = cleanup_numpy_scalar(v)
    elif isinstance(data, (list, tuple)):
        data = [cleanup_numpy_scalar(e) for e in data]
    return data


def make_hash(obj):
    """
    Given a Python object, returns a 32 character hash string to uniquely identify
    the content of the object. The object can be arbitrary nested (i.e. dictionary 
    of dictionary of list etc), and hashing is applied recursively to uniquely 
    identify the content.
    
    For dictionaries (at any level), the key order is ignored when hashing
    so that {"a":5, "b": 3, "c": 4} and {"b": 3, "a": 5, "c": 4} will both
    give rise to the same hash. Exception to this rule is when an OrderedDict
    is passed, in which case difference in key order is respected. To keep
    compatible with previous versions of Python and the assumed general
    intentions, key order will be ignored even in Python 3.7+ where the
    default dictionary is officially an ordered dictionary.

    Args: 
        obj - A (potentially nested) Python object

    Returns:
        hash: str - a 32 charcter long hash string to uniquely identify the object.
    """
    hashed = hashlib.md5()
    
    if isinstance(obj, str):
        hashed.update(obj.encode())
    elif isinstance(obj, OrderedDict):
        for k, v in obj.items():
            hashed.update(str(k).encode())
            hashed.update(make_hash(v).encode())
    elif isinstance(obj, Mapping):
        for k in sorted(obj, key=str):
            hashed.update(str(k).encode())
            hashed.update(make_hash(obj[k]).encode())
    elif isinstance(obj, Iterable):
        for v in obj:
            hashed.update(make_hash(v).encode())
    else:
        hashed.update(str(obj).encode())
    
    return hashed.hexdigest()


def need_to_commit(repo, repo_name=""):
    changed_files = [item.a_path for item in repo.index.diff(None)]
    has_uncommited = bool(changed_files) or bool(repo.untracked_files)

    err_msg = []
    if has_uncommited:
        err_msg.append("\n{}".format(repo_name))
        if repo.untracked_files:
            for f in repo.untracked_files:
                err_msg.append("Untracked: \t" + f)
        if changed_files:
            for f in changed_files:
                err_msg.append("Changed: \t" + f)

    return "\n".join(err_msg)


def get_origin_url(g):
    for remote in g.remote(verbose=True).split("\n"):
        if remote.find("origin") + 1:
            origin_url = remote.split(" ")[0].split("origin\t")[-1]
            return origin_url
        else:
            warnings.warn("The repo does not have any remote url, named origin, specified.")


def check_repo_commit(repo_path):
    repo = Repo(path=repo_path)
    g = cmd.Git(repo_path)
    origin_url = get_origin_url(g)
    repo_name = origin_url.split("/")[-1].split(".")[0]
    err_msg = need_to_commit(repo, repo_name=repo_name)

    if err_msg:
        return '{}_error_msg'.format(repo_name), err_msg

    else:
        sha1, branch = repo.head.commit.name_rev.split()
        commit_date = datetime.fromtimestamp(repo.head.commit.authored_date).strftime("%A %d. %B %Y %H:%M:%S")
        committer_name = repo.head.commit.committer.name
        committer_email = repo.head.commit.committer.email

        return repo_name, {"sha1": sha1, "branch": branch, "commit_date": commit_date,
                          "committer_name": committer_name, "committer_email": committer_email,
                          "origin_url": origin_url}


def gitlog(repos=()):
    """
    A decorator on computed/imported tables.
    Monitors a list of repositories as pointed out by `repos` containing a list of paths to Git repositories. If any of these repositories 
    contained uncommitted changes, the `populate` is interrupted.
    Otherwise, the state of commits associated with all repositoreis are summarized and stored in the associated entry in the GitLog part table.

    Example:
    
    @schema
    @gitlog(['/path/to/repo1', '/path/to/repo2'])
    class MyComputedTable(dj.Computed):
        ...
    
    """
    def gitlog_wrapper(cls):
        # if repos list is empty, skip the modification alltogether
        if len(repos) == 0:
            return cls
            
        class GitLog(dj.Part):
            definition = """
            ->master
            ---
            info :              longblob
            """

        def check_git(self):
            commits_info = {name: info for name, info in [check_repo_commit(repo) for repo in repos]}
            assert len(commits_info) == len(repos)

            if any(['error_msg' in name for name in commits_info.keys()]):
                err_msgs = ["You have uncommited changes."]
                err_msgs.extend([info for name, info in commits_info.items() if 'error_msg' in name])
                err_msgs.append("\nPlease commit the changes before running populate.\n")
                raise RuntimeError('\n'.join(err_msgs))
                
            return commits_info
        
        cls._base_populate = cls.populate
        cls._base_make = cls.make
        cls.check_git = check_git
        cls.GitLog = GitLog
        cls._commits_info = None
        
        def alt_populate(self, *args, **kwargs):
            # the commits info must be attached to the class
            # as table instance is NOT shared between populate and
            # make calls
            self.__class__._commits_info = self.check_git()
            ret = self._base_populate(*args, **kwargs)
            self.__class__._commits_info = None
            return ret

        def alt_make(self, key):
            ret = self._base_make(key)
            if self._commits_info is not None:
                # if there was some Git commit info
                entry = dict(key, info=self._commits_info)
                self.GitLog().insert1(entry)
            return ret

        cls.populate = alt_populate
        cls.make = alt_make

        return cls
        
    return gitlog_wrapper


def create_param_expansion(f_name, container_table, fn_field=None, config_field=None, resolver=None, suffix='Param', default_to_str=False):
    """
    Given a function name `f_name` as would be found in the `container_table` class, this will create
    a new DataJoint computed table subclass with the correct definition to expand blobs corresponding to
    the `f_name`'s arguments.

    The `container_table` must be a class (not an instance), and is also
    expected to implement `resolve_fn` method that can be used to resolve the name of the function to
    a specific function object. The name of the attributes for the function name and function argument
    object inside `container_table` is automatically inferred based on attribute names (i.e. ending with
    `_fn` and `_config`, repsectively).

    Alternatively, you can specifically supply a `resolver` function to resolve the function name,
    and also specify the name of the function and config object attributes via `fn_field` and `config_field`.
    Resolver functions are implemented in nnfabrik.builder and can be imported from there (`resolve_model`)

    The resulting computed table will have a name of the form `MyFunctionNameParam` for a function named
    `my_function_name`. In otherwords, the name is converted from snake_case to CamelCase and `suffix`
    (default to 'Param') is appended.
    """

    if fn_field is None:
        fn_field = next(v for v in container_table.heading.attributes.keys() if v.endswith('_fn'))

    if config_field is None:
        config_field = next(v for v in container_table.heading.attributes.keys() if v.endswith('_config'))

    resolver = resolver or (lambda x: container_table.resolve_fn(x))
    f = resolver(f_name)

    def_str = make_definition(f, default_to_str=default_to_str)

    class NewTable(dj.Computed):
        definition = """
        -> {}
        ---
        {}
        """.format(container_table.__name__, def_str)

        @property
        def key_source(self):
            return container_table & '{}="{}"'.format(fn_field, f_name)

        def make(self, key):
            entries = (container_table & key).fetch1(config_field)
            entries = cleanup_numpy_scalar(entries)
            key = dict(key, **entries)
            if default_to_str:
                for k,v in key.items():
                    if type(v) in [list, tuple]:
                        key[k]=str(v)
            self.insert1(key, ignore_extra_fields=True)

    NewTable.__name__ = to_camel_case(f.__name__) + suffix
    return NewTable


def make_definition(f, exclude=('model', 'dataloaders', 'seed'), default_to_str=False):
    """
    Given a function `f`, creates a table definition string to house all arguments. The types
    of the arguments are inferred from (1) type annotation and (2) type of the default value if present,
    in that order. If type cannot be inferred, defaults to `longblob`.

    Arguments matching values in the exclude list will not be included in the definition.
    """
    type_lut = {
        int: 'int',
        float: 'float',
        str: 'varchar(255)',
        date: 'DATE',
        datetime: 'DATETIME',
        object: 'longblob',
        bool: 'bool',
    }
    argspec = inspect.getfullargspec(f)
    total_def = []
    def_lut = {}
    if argspec.defaults is not None:
        def_lut = {k: (d if d is not None else 'NULL') for k, d in zip(argspec.args[::-1], argspec.defaults[::-1])}

    for v in argspec.args:
        # skip arguments found in the exclude list
        if v in exclude:
            continue
        if v in argspec.annotations:
            t = argspec.annotations[v]
            if not t in [str, int, float, bool]:
                if default_to_str:
                    t = str
                else:
                    t = object
        elif v in def_lut:
            t = type(def_lut[v])
            if not t in [str, int, float, bool]:
                if default_to_str:
                    t = str
                else:
                    t = object
        else:
            t = object
        field = type_lut.get(t, 'longblob')  # default to longblob if no match found
        # if boolean field, turn default value into an integer
        if field == 'bool' and v in def_lut:
            def_lut[v] = int(def_lut[v])

        if v in def_lut:
            default_clause = '={!r}'.format(def_lut[v]) if def_lut[v] else '=NULL'
        else:
            default_clause = ''
        spec_str = '{}{}: {}   # autogenerated column - {}'.format(v, default_clause, field, field)
        total_def.append(spec_str)
    return '\n'.join(total_def)


class CustomSchema(Schema):
    def __call__(self, cls, *, context=None):
        context = context or self.context or inspect.currentframe().f_back.f_locals
        # Process part tables and replace with specific subclass
        for attr in dir(cls):
            if attr[0].isupper():
                part = getattr(cls, attr)
                if inspect.isclass(part) and issubclass(part, dj.Part):
                    class WrappedPartTable(part):
                        pass
                    WrappedPartTable.__name__ = attr
                    setattr(cls, attr, WrappedPartTable)
            return super().__call__(cls, context=context)