# helper functions that the datajoint tables are requiring

import warnings
from datetime import datetime
import hashlib
import datajoint as dj
from git import Repo, cmd

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


def need_to_commite(repo):
    changed_files = [item.a_path for item in repo.index.diff(None)]
    has_uncommited = bool(changed_files) or bool(repo.untracked_files)

    if has_uncommited:
        print("You have uncommited changes:\n")
        if repo.untracked_files:
            for f in repo.untracked_files:
                print("Untracked: \t" + f)
        if changed_files:
            for f in changed_files:
                print("Changed: \t" + f)
        print("\nPlease commit the changes before running populate.\n")

    return has_uncommited


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
    can_populate = not need_to_commite(repo)
    if can_populate:
        sha1, branch = repo.head.commit.name_rev.split()
        commit_date = datetime.fromtimestamp(repo.head.commit.authored_date).strftime("%A %d. %B %Y %H:%M:%S")
        committer_name = repo.head.commit.committer.name
        committer_email = repo.head.commit.committer.email
        origin_url = get_origin_url(g)
        repo_name = origin_url.split("/")[-1].split(".")[0]

        return repo_name, {"sha1": sha1, "branch": branch, "commit_date": commit_date,
                          "committer_name": committer_name, "committer_email": committer_email,
                          "origin_url": origin_url}
    else:
        return None, None


# def gitlog(cls):
#     """
#     Decorator that equips a datajoint class with an additional datajoint.Part table that stores the current sha1,
#     the branch, the date of the head commit,and whether the code was modified since the last commit,
#     for the class representing the master table. Use the instantiated version of the decorator.
#     Here is an example:
#     .. code-block:: python
#        :linenos:
#         import datajoint as dj
#         from djaddon import gitlog
#         schema = dj.schema('mydb',locals())
#         @schema
#         @gitlog
#         class MyRelation(dj.Computed):
#             definition = ...
#     """
#
#     class GitKey(dj.Part):
#         definition = """
#         ->master
#         ---
#         sha1 :             varchar(40)
#         branch :           varchar(50)
#         commit_date :      datetime
#         commiter_name :    varchar(50)
#         commit_email :     varchar(50)
#         origin_url :       varchar(100)
#         """
#
#     def log_key(self, key):
#         key = dict(key)
#         out = check_repo_commit(repo_path)
#         if out:
#             key['sha1'], key['branch'], key['commit_date'], key['commiter_name'], key['commiter_email'], key['original_url']
#             self.GitKey().insert1(key, skip_duplicates=True, ignore_extra_fields=True)
#             return key
#
#     cls.GitKey = GitKey
#     cls.log_git = log_key
#
#     return cls
