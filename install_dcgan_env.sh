#!/usr/bin/env bash
rm -f ~/tensorflow_DCGAN_env 2>/dev/null
virtualenv --system-site-packages -p python3 ~/tensorflow_DCGAN_env
cd ~/tensorflow_DCGAN_env
source ~/tensorflow_DCGAN_env/bin/activate
pip3 install --upgrade pip
pip3 install pillow
pip3 install tensorflow==0.12.1
pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose
deactivate
