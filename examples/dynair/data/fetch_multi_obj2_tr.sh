OUTPUT_FILE="`dirname $0`/multi_obj2_tr.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/4soav3xfxq8cisx/multi_obj2_tr.npz"
fi
