#!/bin/bash


awk '{print $1, $2}'     info.01.t.dat > Wire/info/vde.dat
awk '{print $1, $3}'     info.01.t.dat > Wire/info/vdh.dat
awk '{print $1, $4}'     info.01.t.dat > Wire/info/rhoe.dat
awk '{print $1, $5}'     info.01.t.dat > Wire/info/rhoh.dat
awk '{print $1, $6}'     info.01.t.dat > Wire/info/enge.dat
awk '{print $1, $7}'     info.01.t.dat > Wire/info/engh.dat
awk '{print $1, $8}'     info.01.t.dat > Wire/info/tempe.dat
awk '{print $1, $9}'     info.01.t.dat > Wire/info/temph.dat
awk '{print $1, $10}'    info.01.t.dat > Wire/info/rhomaxe.dat
awk '{print $1, $11}'    info.01.t.dat > Wire/info/rhomaxh.dat
awk '{print $1, $12}'    info.01.t.dat > Wire/info/ge.dat
awk '{print $1, $13}'    info.01.t.dat > Wire/info/gh.dat
awk '{print $1, -1*$14}' info.01.t.dat > Wire/info/ve.dat
awk '{print $1, $15}'    info.01.t.dat > Wire/info/vh.dat
awk '{print $1, $16}'    info.01.t.dat > Wire/info/I0.dat

awk '{print $1, 8.85e-12*$2}'     EP.01.t.dat > Wire/info/Ex.t.dat
awk '{print $1, 8.85e-12*$3}'     EP.01.t.dat > Wire/info/Ey.t.dat
awk '{print $1,          $4}'     EP.01.t.dat > Wire/info/Px.t.dat
awk '{print $1,          $5}'     EP.01.t.dat > Wire/info/Py.t.dat

