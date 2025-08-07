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

if [[ -z "$2" ]]; then
  echo "]_No argument supplied, fall back to port 25"
else
  port="$2"
  echo "]_Argument supplied, port is ${port}"
fi

nbIP="$(cat ${file} | wc -l)"
echo "]_Found ${nbIP} lines in ${file}"

i="0"

while read ip; do
  if [[ "$i" -eq "0" ]]; then
    tmux new -s server -d
  else
    tmux split-window -t server
    tmux select-layout -t server tiled
  fi
  tmux send-keys -t server:0.$i "ssh ${ip}" C-m
  sleep 1

  command="openvpn\
  --port ${port}\
  --comp-lzo no\
  --cipher no\
  --ncp-disable\
  --verb 3\
  --proto udp\
  --dev tun0\
  --ifconfig 172.16.42.$((i*2+1)) 172.16.42.$((i*2))"

  tmux send-keys -t server:0.$i "${command}" C-m
  i=$((i+1))
done < $file

tmux attach -t server
