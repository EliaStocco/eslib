# How to create these datasets

```bash
build-dataset.py -i train.extxyz -n [-1,0,0] -rc 6
rm dataset.val.pth
rm dataset.test.pth

build-dataset.py -i val.extxyz -n [-1,0,0] -rc 6 -o tmp
mv tmp.train.pth dataset.val.pth
rm tmp.val.pth
rm tmp.test.pth
```
