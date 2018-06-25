OUTPUT_FILE="`dirname $0`/tr_obj13_tr.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/8lx2la48d3afj3l/tr_obj13_tr.npz"
fi
