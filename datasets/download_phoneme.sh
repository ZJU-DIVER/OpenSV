#!/bin/sh
DIR="./CondGen"
mkdir $DIR
cd $DIR

rm -rf webnlg_2017
wget --content-disposition https://cloud.tsinghua.edu.cn/f/cd464aed35fc49429971/?dl=1
tar -zxvf webnlg_2017.tar.gz
rm -rf webnlg_2017.tar.gz
cd ..

#!/bin/sh
rm -rf FewGLUE
wget --content-disposition https://cloud.tsinghua.edu.cn/f/018f9c0b63a542f88791/?dl=1
tar -zxvf FewGLUE.tar.gz
rm -rf FewGLUE.tar.gz