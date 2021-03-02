#!/bin/bash
ps -ef|grep decoder_detector|awk '{print $2}'|xargs kill -9
rm /dev/shm/psm*   
