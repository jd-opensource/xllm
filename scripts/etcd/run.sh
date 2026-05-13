#!/usr/bin/env bash

missing_bins=()

if [ ! -x "./etcd" ]; then
  missing_bins+=("etcd")
fi

if [ ! -x "./etcdctl" ]; then
  missing_bins+=("etcdctl")
fi

if [ ${#missing_bins[@]} -ne 0 ]; then
  echo "Error: Missing executable file(s) in current directory: ${missing_bins[*]}"
  echo "Please download and use version 3.6.8 from https://github.com/etcd-io/etcd/releases."
  echo "Move etcd and etcdctl to xllm/scripts/etcd directory."
  exit 1
fi

pkill -9 etcd

nohup ./etcd \
     --listen-client-urls http://0.0.0.0:8400 \
     --advertise-client-urls http://0.0.0.0:8400 > log.txt 2>&1 &


./etcdctl --endpoints=http://0.0.0.0:8400 del --prefix XLLM:CACHE
