OUTPUT_FILE="`dirname $0`/tr2_obj13_tr.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/aadnr2fd3ymknmm/tr2_obj13_tr.npz"
fi
