#!/bin/bash
# contact_sheet.sh out.jpg "t1 t2 t3" label1=video1.mp4 label2=video2.mp4 ...
# Rows = timestamps, columns = videos; each tile scaled to 240x426 (9:16) with a label.
out=$1; times=$2; shift 2
tmp=$(mktemp -d -p "$HOME/personal/ai-character-scope/.tmp"); rows=()
for t in $times; do
  cols=()
  for lv in "$@"; do
    label=${lv%%=*}; vid=${lv#*=}; f="$tmp/${label}_$t.png"
    ffmpeg -loglevel error -y -ss "$t" -i "$vid" -frames:v 1 \
      -vf "scale=240:426:flags=bicubic,drawtext=text='$label t=$t':x=4:y=4:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.6" "$f"
    cols+=(-i "$f")
  done
  n=$(( ${#cols[@]} / 2 ))
  ffmpeg -loglevel error -y "${cols[@]}" -filter_complex "hstack=$n" "$tmp/row_$t.png"
  rows+=(-i "$tmp/row_$t.png")
done
n=$(( ${#rows[@]} / 2 ))
if [ "$n" -gt 1 ]; then ffmpeg -loglevel error -y "${rows[@]}" -filter_complex "vstack=$n" "$out"; else cp "$tmp/row_$times.png" "$out"; fi
rm -rf "$tmp"
