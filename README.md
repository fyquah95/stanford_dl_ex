## stanford_dl_ex Solutions

This is a fork of [stanford_dl_ex](https://github.com/amaas/stanford_dl_ex) with the solutions for [Stanford UFLDL Exercises](http://ufldl.stanford.edu/tutorial).

Note that these are __NOT__ the official solutions : they are simply solutions which I have coded and worked decently to give me reasonable solutions.

Note that this repository is made for [__Octave__](http://www.gnu.org/software/octave/) . Mose functionalities _should_ work equally fine on Matlab, but I have not tested them on Matlab.

## Setting up

1. Clone this repository
2. Compile optimizers used throughout this code base:

~~~bash
cd path/to/repository/stanford_dl_ex/common/minFunc_2012
octave mexAll.m
# This will start some prerequiste files
~~~

3. Install the io and statistics package (used for `randsample` in gradient checking subroutines)

~~~bash
# in Any Directory
octave # enter the octave Prompt
octave:1> pkg install -forge io
octave:1> pkg install -forge statistics

~~~

## Core Codebase Changes

In addition to solving the problems, I have made some changes to adapt the codebase for Octave.

1. Change `mex -outdir ...` into `mex -o ...` in  [mexAll.m](https://github.com/fyquah95/stanford_dl_ex/blob/master/common/minFunc_2012/mexAll.m)
2. Manually load statistics package in `grad_check.m`
