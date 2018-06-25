OUTPUT_FILE="`dirname $0`/tr_obj11_t.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/rwvfcm9kjl6aav5/tr_obj11_t.npz"
fi
