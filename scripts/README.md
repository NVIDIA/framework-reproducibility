# Scripts

These scripts operate on a clone of the TensorFlow repo in this directory
(`./tensorflow/`). The first time a script is run, the clone will be created
automatically. On subsequent runs, the clone will be updated using a `git pull`.

## Find TensorFlow Commits

### Introduction and Use

This script searches through the all the commits between two dates, looking for
any that reference determinism, either in the commit message or in the file
diff. It provides a date and a GitHub URL for each relevant commit.

To use,

```
./find-tf-commits.sh
```

### Results

I ran this on all commits between 2015-11-06 (the date of the very first commit)
and 2020-01-30 and it found 1,391 commits out of a total of 77,508. That's 1.8%
of total commits. I also confirmed that the total count matches the commit count
reported by GitHub.

### Conclusions

  1. There are too many commits to manually sort.
  2. The realization that they represented a small, arbitrary selection led to
     the removal of two items from the [list][100] of pull requests and
     individual commits in the main project README: [PR 10636][101]
     (Non-determinism Docs) and [PR 24273][102] (Enable dataset.map to respect
     seeds from the outer context).

### Enhancement Opportunities

  * Search through all pull-requests as well.
  * Broaden the search terms to catch potentially relevant pull requests and
    individual commits that don't mention "determinism" or "deterministic",
    such as [PR 24273][102].
  * Auto-generate a searchable markdown file containing the results (including
    date, description, and a link to the commit or PR) that can be added to this
    repo.

[100]: ../README.md#tensorflow-pull-requests
[101]: https://github.com/tensorflow/tensorflow/pull/10636
[102]: https://github.com/tensorflow/tensorflow/pull/24273