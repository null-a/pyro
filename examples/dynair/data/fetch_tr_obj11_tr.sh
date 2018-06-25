OUTPUT_FILE="`dirname $0`/tr_obj11_tr.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/tk4bjglx1nwyodb/tr_obj11_tr.npz"
fi
