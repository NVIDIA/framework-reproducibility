# Find TensorFlow Commits

This script searches through the all the commits between two dates, searching
for any that reference determinism, either in the commit message or in the file
diff. It provides a date and a GitHub URL for each relevant commit.

To use,

```
./find.sh
```

# Results

I ran this on all commits between 2015-11-06 (the date of the very fist commit)
and 2020-01-30 and it found 1,391 commits out of a total of 77,508. That's 1.8%
of total commits. I also confirmed that the total count matches the commit count
reported by GitHub.