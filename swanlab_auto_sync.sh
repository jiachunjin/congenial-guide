#!/bin/bash

# 获取用户输入的 RUN_NAME 和 RUN_ID
read -p "请输入 RUN_NAME: " RUN_NAME
read -p "请输入 RUN_ID: " RUN_ID

# 每2秒运行一次同步命令
while true; do
    swanlab sync ./swanlog/${RUN_NAME} --id ${RUN_ID}
    sleep 2
done