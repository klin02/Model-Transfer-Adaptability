#!/bin/bash

# 指定本地源目录和远程目标目录
local_dir="/lustre/S/youkunlin/WorkSpace/ALL/"
remote_dir="/lustre/S/youkunlin/Model-Transfer-Adaptability/ykl/ALL"

# 读取.gitignore文件，排除指定的文件和文件夹
exclude_args=$(while read line; do echo "--exclude=$line"; done < ${local_dir}/.gitignore)

# 使用rsync同步文件夹
rsync -avz --delete ${exclude_args} ${local_dir}/ ${remote_dir}