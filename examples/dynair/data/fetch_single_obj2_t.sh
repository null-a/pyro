OUTPUT_FILE="`dirname $0`/single_obj2_t.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/d9a6zsbknfutj9j/single_obj2_t.npz?dl=0"
fi
