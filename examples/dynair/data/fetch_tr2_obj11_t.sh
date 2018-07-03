OUTPUT_FILE="`dirname $0`/tr2_obj11_t.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/qzqnwow8jewvsep/tr2_obj11_t.npz"
fi
