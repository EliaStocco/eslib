#!/bin/bash

ext=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
echo $ext

name=_rdfs_fort
python -m numpy.f2py -c -m $name $name.f90
# CWD=$(pwd)
# cd ../gtlib/utils
# ln -sf ${name}${ext}
# cd $CWD
