OUTPUT_FILE="`dirname $0`/tr2_obj02_tr.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/jfhmehsghur9ni7/tr2_obj02_tr.npz"
fi
