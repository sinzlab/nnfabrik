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
        return "{}_error_msg".format(repo_name), err_msg

    else:
        sha1, branch = repo.head.commit.name_rev.split()
        commit_date = datetime.fromtimestamp(repo.head.commit.authored_date).strftime("%A %d. %B %Y %H:%M:%S")
        committer_name = repo.head.commit.committer.name
        committer_email = repo.head.commit.committer.email

        return (
            repo_name,
            {
                "sha1": sha1,
                "branch": branch,
                "commit_date": commit_date,
                "committer_name": committer_name,
                "committer_email": committer_email,
                "origin_url": origin_url,
            },
        )
