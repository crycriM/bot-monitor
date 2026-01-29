#!/bin/bash

# Trading session configuration
SCRIPTS=/home/ubuntu/scripts_new
DASHBOARD=streamlit LAUNCH_BACKEND=yes $SCRIPTS/launch_dashboard.sh >/home/ubuntu/output/monitor_launch.out 2>&1
