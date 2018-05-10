OUTPUT_FILE="`dirname $0`/multi_obj.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/sh/0itp9lx4jm3qycq/AAA3JPYmjV2R5u2cAT1cJiA7a/multi_obj.npz"
fi
