#!/usr/bin/env bash
# Shared output guards. Source from a task wrapper; do not run directly.

prepare_eval_video() {
    local source="$1" destination="$2"
    if [[ -e "$source" || -L "$source" || -e "$destination" || -L "$destination" ]]; then
        printf '[evaluate] output already exists; archive it before rerunning:\n  %s\n  %s\n' \
            "$source" "$destination" >&2
        return 1
    fi
}

move_eval_video() {
    local source="$1" destination="$2"
    if [[ ! -d "$source" ]]; then
        printf '[evaluate] expected video output was not created: %s\n' "$source" >&2
        return 1
    fi
    if [[ -e "$destination" || -L "$destination" ]]; then
        printf '[evaluate] refusing to replace existing output: %s\n' "$destination" >&2
        return 1
    fi
    mv -- "$source" "$destination"
}
