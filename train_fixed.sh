#!/usr/bin/env sh

time  ./build/tools/caffe  train -solver=fix_model/lenet_solver.prototxt  -weights  fix_model/fixed.caffemodel  
