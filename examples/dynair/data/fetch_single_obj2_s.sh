OUTPUT_FILE="`dirname $0`/single_obj2_s.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/cpes3c7rlxa4m09/single_obj2_s.npz"
fi
