#!/bin/bash

file="ip.txt"
port="25"

if [[ -z "$1" ]]; then
  echo "]_No argument supplied, fall back to 'ip.txt'"
else
  file="$1"
  echo "]_Argument supplied, ip file is ${file}"
fi

nbIP="$(cat ${file} | wc -l)"
echo "]_Found ${nbIP} lines in ${file}"

i="0"

while read ip; do
  if [[ "$i" -eq "0" ]]; then
    tmux new -s client -d
  else
    tmux split-window -t client
    tmux select-layout -t client tiled
  fi

  command="openvpn\
  --remote ${ip}\
  --port ${port}\
  --comp-lzo no\
  --cipher no\
  --ncp-disable\
  --verb 3\
  --proto udp\
  --nobind\
  --dev tun${i}\
  --ifconfig 172.16.42.$((i*2)) 172.16.42.$((i*2+1))"

  tmux send-keys -t client:0.$i "${command}" C-m
  i=$((i+1))
done < $file

tmux attach -t client
