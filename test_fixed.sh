#!/usr/bin/env sh

time  ./build/tools/caffe  test -model  fix_model/lenet_train_test.prototxt  -weights  fix_model/fixed.caffemodel    -iterations 100
