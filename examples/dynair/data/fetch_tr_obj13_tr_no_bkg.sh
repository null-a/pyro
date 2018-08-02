OUTPUT_FILE="`dirname $0`/tr_obj13_tr_no_bkg.npz"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/fsa03sy4diy1qw8/tr_obj13_tr_no_bkg.npz"
fi
