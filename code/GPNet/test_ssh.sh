#!/bin/bash

# SSH into the remote machine and execute the command
ssh haolu@172.21.7.51 '
echo "Hello World1" 
nohup echo "Hello World2" &
'