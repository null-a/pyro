OUTPUT_FILE="`dirname $0`/tr2_obj11_tr.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/iuy8v6ikogpxtrz/tr2_obj11_tr.npz"
fi
