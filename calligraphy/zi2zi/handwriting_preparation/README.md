# Convert your handwritings to training data

## 1. Prepare your handwriting images
* Scan the images using highest dpi as possible. Usually accuracy will be good when dpi is above 300
* Save the images to `images/`, we have a `test_image.jpg` as example

## 2. OCR your handwriting images
* Download / Install Tesseract according to the instructions here `https://github.com/tesseract-ocr/tesseract/wiki`
* Download Tesseract's Chinese trained model into folder `tessdata/` from `https://github.com/tesseract-ocr/tessdata/raw/master/chi_sim.traineddata`
* Download Tesseract's Orientation and script detection (OSD) model into folder `tessdata/` from`https://github.com/tesseract-ocr/tessdata/raw/master/osd.traineddata`
* (Optional) If you know the text of your handwriting image, you can save the text at `preprocessing/text.txt`, and run the `preprocessing/convert_text_to_word_list.py`, which will generate `tessdata/configs/wordlist` which will serve as a whitelist.
* Run command line `tesseract ./images/test_image_enhanced.jpg ./images/test_image_enhanced --tessdata-dir ./tessdata/ -l chi_sim --psm 1 --oem 0 wordlist makebox`. After that's done, you should see `test_image.box` file in `images/` folder.
* Download **jTessBoxEditor** from`http://vietocr.sourceforge.net/training.html`
    * Launch jTessBoxEditor, and load your .jpg along with your .box. Note that the basename of the two files have to be the same.
    * Correct your Box results, and save it. The next step will crop the image according to your corrected Box file, as training data.
    * A screenshot is here ![alt text](assets/jTessBoxEditor.png)

## 3. Generate paired_images
* Make sure the parameters in the scripts are right and then `cd ..; PYTHONPATH=/ python font2img_finetune.py`
* Follw the instructions of the Finetune section in `run.sh`