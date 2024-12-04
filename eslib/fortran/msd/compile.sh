#!/bin/bash

ext=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
echo $ext

name=_msd_fort
python -m numpy.f2py -c -m $name $name.f90
# f2py -c -m $name $name.f90
cd ..
ln -sf msd/_msd_fort.cpython-310-x86_64-linux-gnu.so
