OUTPUT_FILE="`dirname $0`/tr_obj11_tr_no_bkg.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/2wae1qdm8bplvev/tr_obj11_tr_no_bkg.npz"
fi
