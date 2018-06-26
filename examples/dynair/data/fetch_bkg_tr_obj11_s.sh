OUTPUT_FILE="`dirname $0`/bkg_tr_obj11_s.pytorch"
if [ ! -f $OUTPUT_FILE ]; then
    curl -L -o $OUTPUT_FILE "https://www.dropbox.com/s/c1ov0ftl1pg658v/bkg_tr_obj11_s.pytorch"
fi
