#!/bin/bash
gfortran -c  tongji.f90 
gfortran -c  file_io.f90 
gfortran -c  calculation.f90
gfortran -c  bridge.f90
gfortran -o  bridge tongji.o file_io.o calculation.o bridge.o
