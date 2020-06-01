
@echo on
set path=D:\anaconda;D:\anaconda\Library\mingw-w64\bin;D:\anaconda\Library\usr\bin;D:\anaconda\Library\bin;D:\anaconda\Scripts;D:\anaconda\bin;D:\anaconda\condabin;%path%
path
python setup.py sdist

twine check dist/*

twine upload dist/*

rd /s /Q dist

rd /s /Q featurebox.egg-info

pause

pause

exit