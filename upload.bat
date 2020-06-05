
@echo on
set path=D:\miniconda;D:\miniconda\Library\bin;D:\miniconda\Scripts;D:\miniconda\condabin;%path%
path
python setup.py sdist

twine check dist/*

twine upload dist/*

rd /s /Q dist

rd /s /Q featurebox.egg-info

pause

pause

exit