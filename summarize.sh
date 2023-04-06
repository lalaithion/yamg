yes | ./movify.sh heightmap 2> /dev/null
echo "done with heightmap.mp4"
yes | ./movify.sh erosion 2> /dev/null
echo "done with erosion.mp4"
yes | ./movify.sh gradient 2> /dev/null
echo "done with gradient.mp4"
rm debug*.png
