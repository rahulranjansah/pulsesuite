#!/bin/bash

awk {'print $1, $2}' EP.01.t.dat > Wire/info/Ex.01.dat
awk {'print $1, $3}' EP.01.t.dat > Wire/info/Ey.01.dat
awk {'print $1, $4}' EP.01.t.dat > Wire/info/Ez.01.dat
awk {'print $1, $5}' EP.01.t.dat > Wire/info/Px.01.dat
awk {'print $1, $6}' EP.01.t.dat > Wire/info/Py.01.dat
awk {'print $1, $7}' EP.01.t.dat > Wire/info/Pz.01.dat


awk {'print $1, $2}' info.01.t.dat > Wire/info/vde.dat
awk {'print $1, $3}' info.01.t.dat > Wire/info/vdh.dat  
awk {'print $1, $4}' info.01.t.dat > Wire/info/rho.e.dat  
awk {'print $1, $5}' info.01.t.dat > Wire/info/rho.h.dat
awk {'print $1, $6}' info.01.t.dat > Wire/info/Eng.e.dat
awk {'print $1, $7}' info.01.t.dat > Wire/info/Eng.h.dat
awk {'print $1, $8}' info.01.t.dat > Wire/info/Temp.e.dat
awk {'print $1, $9}' info.01.t.dat > Wire/info/Temp.h.dat
awk {'print $1, $10}' info.01.t.dat > Wire/info/rmax.e.dat
awk {'print $1, $11}' info.01.t.dat > Wire/info/rmax.h.dat
awk {'print $1, $12}' info.01.t.dat > Wire/info/d.e.dat  
awk {'print $1, $13}' info.01.t.dat > Wire/info/p.dat
awk {'print $1, $14}' info.01.t.dat > Wire/info/p.e.dat
awk {'print $1, $15}' info.01.t.dat > Wire/info/p.h.dat
awk {'print $1, $16}' info.01.t.dat > Wire/info/I0.dat

awk {'print $1, $17}' info.01.t.dat > Wire/info/I0e.dat
awk {'print $1, $18}' info.01.t.dat > Wire/info/I0h.dat

awk {'print $1, 2*$6+2*$7}' info.01.t.dat > Wire/info/En.t.dat
