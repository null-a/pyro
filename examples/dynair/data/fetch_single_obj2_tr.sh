OUTPUT_FILE="`dirname $0`/single_obj2_tr.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/y6jidy9r14asiw7/single_obj2_tr.npz"
fi
