OUTPUT_FILE="`dirname $0`/tr_obj11_s.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/uci8iytf2qo02fa/tr_obj11_s.npz"
fi
