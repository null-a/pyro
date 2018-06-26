OUTPUT_FILE="`dirname $0`/bkg_tr_obj13_tr.pytorch"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/sa20ofu9dpt70we/bkg_tr_obj13_tr.pytorch"
fi
