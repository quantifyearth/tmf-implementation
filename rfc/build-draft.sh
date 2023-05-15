#!/bin/sh

file=$1
name="${file%%.md}"

inotifywait -m -e modify $file |
   while read file_path file_event file_name; do
       echo "Rebuilding $file_path $file_name."
       mmark "$name.md" > "$name.xml"
       xml2rfc --v3 "$name.xml" --html
   done