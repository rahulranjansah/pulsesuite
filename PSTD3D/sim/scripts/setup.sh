#!/bin/bash


fd='fields'
qw='dataQW'

mkdir Backup
mkdir Backup/Wire
mkdir Backup/Fields

mkdir $fd
mkdir $fd/Ex
mkdir $fd/Ey
mkdir $fd/Px
mkdir $fd/Py
mkdir $fd/Dx
mkdir $fd/Dy
mkdir $fd/Hz
mkdir $fd/Jf
mkdir $fd/backup
mkdir $fd/host
mkdir $fd/host/nogam

mkdir $qw
mkdir $qw/Prop
mkdir $qw/Wire
mkdir $qw/backup
mkdir $qw/V

mkdir $qw/Prop/Ex
mkdir $qw/Prop/Ey
mkdir $qw/Prop/Ez
mkdir $qw/Prop/Vr
mkdir $qw/Prop/Px
mkdir $qw/Prop/Py
mkdir $qw/Prop/Pz
mkdir $qw/Prop/Re
mkdir $qw/Prop/Rh
mkdir $qw/Prop/Rho

mkdir $qw/Wire/PL
mkdir $qw/Wire/Jf
mkdir $qw/Wire/Pf
mkdir $qw/Wire/Xqw


mkdir $qw/Wire/Ex
mkdir $qw/Wire/Ey
mkdir $qw/Wire/Ez
mkdir $qw/Wire/Vr
mkdir $qw/Wire/Px
mkdir $qw/Wire/Py
mkdir $qw/Wire/Pz
mkdir $qw/Wire/Re
mkdir $qw/Wire/Rh
mkdir $qw/Wire/Rho
mkdir $qw/Wire/Ee
mkdir $qw/Wire/Eh
mkdir $qw/Wire/Win
mkdir $qw/Wire/Wout



mkdir $qw/Wire/Ge
mkdir $qw/Wire/Gh
mkdir $qw/Wire/Fe
mkdir $qw/Wire/Fh
mkdir $qw/Wire/ne
mkdir $qw/Wire/nh
mkdir $qw/Wire/C
mkdir $qw/Wire/D
mkdir $qw/Wire/P

mkdir $qw/Wire/info

cp scripts/output.sh $qw
