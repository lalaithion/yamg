if [ $1 != "" ]; then
  ffmpeg -framerate 7 -pattern_type glob -i "debug_$1*.png" -c:v libx264 -pix_fmt yuv420p $1.mp4
fi
