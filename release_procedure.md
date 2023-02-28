# Release Procedure

This is the procedure to follow to finalize a release of
`framework-reproducibility` (`fwr13y`). See the official python documentation on
[setuptools][1].

## 1. Merge Release Branch

Thoroughly review the release branch (e.g. `r0.4`) and merge it into master.
Then delete the release branch.

## 2. Check Version Number

Confirm that `__version__` in `fwr13y/version.py` is correct.

## 3. Run Tests

Run all the tests and make sure they all pass.

For determinism:
```
$cd test/d9m
$./all.sh
```

For seeder: *to-be-completed*

## 4. Create a Source Distribution

```
python3 setup.py sdist
```

This will generate an appropriately-named `.tar.gz` source distribution file in
the `dist` directory.

After creating the source distribution, check that it contains all the packages,
sub-packages, modules, and other files that are expected (and no others). This
can be done by viewing the `stdout` of `python3 setup.py sdist`, by viewing
`framework_reproducibility.egg-info/SOURCES.txt`, or by running `tar -xf` on the
source distribution file and examing the resuling directory structure.

Note that `framework-reproducibility.egg-info/SOURCES.txt` is read in, modified,
and written back out by `python3 setup.py sdist`. Because of this, if any items
are to be excluded from the distribution (e.g. via `MANIFEST.in` or the
`packages` parameter of `setup()` in `setup.py`) then
`framework-reproducibility.egg-info` must be deleted prior to running
`python3 setup.py sdist`, otherwise the exluded items will remain.

Note that to install the source distribution, the user will need to have `pip`
installed a new-enough version of `setuptools` and also `wheel`.

## 5. Create a Universal Wheel

```
python3 setup.py bdist_wheel
```

This will generate an appopriately-named wheel in the `dist` directory, with a
`.whl` filename extension. Since this is actually a zip file, its contents can
be viewed using `zipinfo`. Use `zipinfo` to check that all the expected files
(and no unexpected files) are present in the wheel.

Note that `setup.cfg` specifies that wheels are universal by default.

[Common Python Packaging Mistakes][2] is a particularly useful resource that I
used in preparing the above two steps.

## 6. Upload to PyPI

Upload the source distribution and the universal wheel to the Python Package
Index (PyPI).

The following assumes that `$HOME/.pypirc` exists and contains the following:

```
distutils]
index-servers =
    pypi
    testpypi

[pypi]
# Use the upload tool's default repository URL
username: <username>

[testpypi]
repository: https://test.pypi.org/legacy/
username: <username>
```

### 6a. Test PyPI Server


```
twine upload --repository testpypi dist/framework-reproducibility-<version>*
twine upload --repository testpypi dist/framework_reproducibility-<version>*
```

Review the release online and try installing it.

```
cd ~/temp
python3 -m venv venv
venv/bin/pip install -i https://test.pypi.org/simple/framework-reproducibility
```

### 6b. Real PyPI Server

```
twine upload --repository pypi dist/framework-reproducibility-<version>*
twine upload --repository pypi dist/framework_reproducibility-<version>*
```

Again, review the release online and try installing it.

```
cd ~/temp
python3 -m venv venv
venv/bin/pip install framework-reproducibility
```

## 7. Create GitHub Release

Finally, on GitHub, create a new release with an appropriate version tag
(e.g. `v0.4.0`).

[1]: https://packaging.python.org/guides/distributing-packages-using-setuptools/
[2]: https://jwodder.github.io/kbits/posts/pypkg-mistakes/
