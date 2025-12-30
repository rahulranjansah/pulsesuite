#!/bin/bash

mkdir sim
cp -a stuff   sim/
cp -a scripts sim/
cp -a params  sim/
cp    SBETest     sim/


cd sim
./scripts/setup.sh
