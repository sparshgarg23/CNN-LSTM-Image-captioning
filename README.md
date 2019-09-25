# CNN-LSTM-Image-captioning(part of Udacity's CVND program
Takes an image and generates a caption to describe it using an encoder-decoder architecture based on CNN-LSTM model as proposed in Show and tell a neural net image caption generation
Requirements:please refer to requiremtn.txt
The model is based on the paper discussed in "Show and tell : a neural net image captioning generator"
A few notes,the model doesn't generalise well on all images,for instance it generates a beach caption for a snow field.
Probable reasons can be smaller vocabulary ,small learning rate.
In my opinion,increasing the learning rate to 0.5 and  decreasing the vocab_threshold to 4 or 3 should do the trick.
If you have any suggestions please let me know.
