#!/bin/bash
# Side-by-side review video + a 6-frame contact sheet: sbs.sh INPUT.mp4 OUTPUT.mp4 REF.png OUT_PREFIX [LABEL]
# (input | output | reference), all scaled to 480 px high.
set -e
IN=$1; OUT=$2; REF=$3; P=$4; LABEL=${5:-}
ffmpeg -loglevel error -y -i "$IN" -i "$OUT" -loop 1 -i "$REF" -filter_complex \
 "[0:v]scale=-2:480,setsar=1[a];[1:v]scale=-2:480,setsar=1,drawtext=text='${LABEL}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[b];[2:v]scale=-2:480,setsar=1[c];[a][b][c]hstack=3:shortest=1,scale='min(1920,iw)':-2" \
 -c:v libx264 -crf 22 -pix_fmt yuv420p -profile:v high -level 4.1 -movflags +faststart -shortest "$P.mp4"
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$P.mp4")
ffmpeg -loglevel error -y -i "$P.mp4" -vf "fps=6/${D%.*},scale=-2:240,tile=2x3" -frames:v 1 "$P.jpg"
