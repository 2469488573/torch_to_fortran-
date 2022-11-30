#!/bin/bash
ifort -c  tongji.f90 
ifort -c  file_io.f90 
ifort -c  calculation.f90
ifort -c  bridge.f90
ifort -o  bridge tongji.o file_io.o calculation.o bridge.o
