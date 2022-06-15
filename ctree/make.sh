set -ex
rm -rf build
rm -rf cytree.cpp
python setup.py build_ext --inplace
