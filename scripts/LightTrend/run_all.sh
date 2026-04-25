#!/bin/bash

# run_all.sh가 위치한 절대경로
script_dir="$(cd "$(dirname "$0")" && pwd)"
current_script="$(basename "$0")"

for file in "$script_dir"/*.sh; do
    # 매칭되는 파일이 없으면 건너뜀
    [ -e "$file" ] || continue

    # 자기 자신은 제외
    if [ "$(basename "$file")" != "$current_script" ]; then
        echo "Running $file ..."
        chmod +x "$file"
        "$file"
        echo "$file finished."
    fi
done
